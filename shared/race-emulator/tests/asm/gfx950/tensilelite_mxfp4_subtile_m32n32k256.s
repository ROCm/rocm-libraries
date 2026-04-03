
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Alik_Bljk_F4SS_MXA32_MXB32_BH_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA32_GRVWB32_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT2_2_MO40_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
.globl Cijk_Alik_Bljk_F4SS_MXA32_MXB32_BH_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA32_GRVWB32_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT2_2_MO40_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
.p2align 8
.type Cijk_Alik_Bljk_F4SS_MXA32_MXB32_BH_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA32_GRVWB32_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT2_2_MO40_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_F4SS_MXA32_MXB32_BH_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA32_GRVWB32_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT2_2_MO40_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 152 // accvgpr offset
  .amdhsa_next_free_vgpr 168 // vgprs
  .amdhsa_next_free_sgpr 86 // sgprs
  .amdhsa_group_segment_fixed_size 49152 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_user_sgpr_count 13
  .amdhsa_user_sgpr_kernarg_preload_length 11
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.end_amdhsa_kernel
.text
/* Num VGPR   =151 */
/* Num AccVGPR=16 */
/* Num SGPR   =86 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 2 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=32, GlobalReadVectorWidthB=32 */
/* DirectToLdsA=True */
/* DirectToLdsB=True */
/* UseSgprForGRO=0 */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Cijk_Alik_Bljk_F4SS_MXA32_MXB32_BH_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA32_GRVWB32_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT2_2_MO40_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
    .symbol: 'Cijk_Alik_Bljk_F4SS_MXA32_MXB32_BH_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA32_GRVWB32_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT2_2_MO40_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1.kd'
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
        .value_type:      f32
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      fp4
        .address_space:   generic
      - .name:            MXSA
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          64
        .value_kind:      global_buffer
        .value_type:      fp4
        .address_space:   generic
      - .name:            MXSB
        .size:            8
        .offset:          72
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideMXSA0
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideMXSA1
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideMXSB0
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideMXSB1
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      f32
    .group_segment_fixed_size:   49152
    .kernarg_segment_align:      8
    .kernarg_segment_size:       136
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 86
    .sgpr_spill_count:           0
    .vgpr_count:                 151
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_F4SS_MXA32_MXB32_BH_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA32_GRVWB32_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV0_MIWT2_2_MO40_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1:
label_ASM_Start:  /// Main body of the asm kernel
.macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\vgprDstIdx+0], \dividend, \magicA
    v_add_u32 v[\vgprDstIdx+0], v[\vgprDstIdx+0], v[\vgprDstIdx+1]
    v_lshrrev_b32 v[\vgprDstIdx+0], \magicShift, v[\vgprDstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
.set vgprSerial, 0

/******************************************/
/* SGPR Assignments                       */
/******************************************/
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
.set sgprNumWorkGroups0, 14
.set sgprNumWorkGroups1, 15
.set sgprSizesFree, 16
.set sgprSizesSum, 19
.set sgprAddressD, 20
.set sgprAddressC, 22
.set sgprAddressA, 24
.set sgprAddressMXSA, 26
.set sgprAddressB, 28
.set sgprAddressMXSB, 30
.set sgprStridesD, 32
.set sgprStridesC, 34
.set sgprStridesA, 36
.set sgprStridesMXSA, 38
.set sgprStridesB, 40
.set sgprStridesMXSB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprLocalWriteBaseAddrA, 46
.set sgprLocalWriteBaseAddrB, 47
.set sgprLocalWriteSwapA, 48
.set sgprLocalWriteSwapB, 49
.set sgprLocalWriteBaseAddrMXSA, 50
.set sgprLocalWriteSwapMXSA, 51
.set sgprLocalWriteBaseAddrMXSB, 52
.set sgprLocalWriteSwapMXSB, 53
.set sgprSwapCommon, 54
.set sgprGSU, 55

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
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
.set constStrideMXSAL, 1
.set sgprStrideMXSA0I, sgprStridesMXSA+0
.set sgprStrideMXSAK, sgprStridesMXSA+1
.set constStrideMXSBL, 1
.set sgprStrideMXSB1J, sgprStridesMXSB+0
.set sgprStrideMXSBK, sgprStridesMXSB+1

.set MT0, 64
.set MT1, 64
.set DepthU, 256
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 0
.set SrdShiftLeftMXSA, 0
.set SrdShiftLeftB, 0
.set SrdShiftLeftMXSB, 0
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x20000                           */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* num_format (3b): 0                     */
/* data_format (4b): 4                    */
/* user_vm_enable (1b): 0                 */
/* user_vm_mode (1b): 0                   */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* _unusedA (3b): 0                       */
/* nv (1b): 0                             */
/* _unusedB (2b): 0                       */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x20000
/* 2 LDS Blocks for PGR 0 */
.set LdsOneBlockSize, 0
.set LdsBlockEndSize, 0

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_dword s56, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_dword s58, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_dword s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_dword s59, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s57, s56, 0x1e                          // Get arg type
s_and_b32 s56, 0x3fffffff, s56                     // Get nums of gemm
s_cmp_eq_u32 s57, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[16:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_dwordx8 s[32:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_waitcnt lgkmcnt(0)                               // preload
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_branch label_common_kernel_entry

/* pad 35 snops to satisfy 0x100 code size for Preload Backward Compatibility Prologue */
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
label_Preload_Offset_Start:
s_and_b32 s56, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s57, s2, 0x1e                           // Get arg type
s_mov_b32 s58, s3                                  // Preload internal args
s_cmp_eq_u32 s57, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dword s23, s[sgprKernArgAddress:sgprKernArgAddress+1], 28 // 28
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 32 // 32
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_mov_b64 s[16:17], s[6:7]                         // move preload data to correct sgpr
s_mov_b64 s[18:19], s[8:9]                         // move preload data to correct sgpr
s_mov_b64 s[20:21], s[10:11]                       // move preload data to correct sgpr
s_mov_b32 s22, s12                                 // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
s_mov_b32 s59, s5                                  // Load num of WGs
label_common_kernel_entry:  /// for both preload/non-preload common code
s_mov_b32 s[sgprWorkGroup0+0], s13                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+1], s14                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+2], s15                 // restore workgroup id
s_and_b32 s[sgprStaggerU], s58, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s58, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s57
s_mov_b32 m0, 0xc000                               // LDS clamp at 49152 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* remap workgroup to XCCs */
s_lshr_b32 s64, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s64, s64                             // Get log(WGMXCC)
s_lshr_b32 s65, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s64, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s61, s59, s64
s_lshl_b32 s61, s61, s64
s_cmp_ge_u32 s[sgprWorkGroup0], s61
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s65, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s61, s[sgprWorkGroup0], s64
s_bfm_b32 s62, s64, 0
s_and_b32 s62, s[sgprWorkGroup0], s62
s_lshr_b32 s63, s59, s64
s_mul_i32 s62, s62, s63
s_add_u32 s[sgprWorkGroup0], s61, s62
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[2:3], s65                          // s61 = s[sgprWorkGroup0] / s65
v_rcp_f64 v[2:3], v[2:3]                           // s61 = s[sgprWorkGroup0] / s65
v_cvt_f64_u32 v[4:5], s[sgprWorkGroup0]            // s61 = s[sgprWorkGroup0] / s65
v_mul_f64 v[2:3], v[2:3], v[4:5]                   // s61 = s[sgprWorkGroup0] / s65
v_cvt_u32_f64 v2, v[2:3]                           // s61 = s[sgprWorkGroup0] / s65
v_mul_lo_u32 v3, v2, s65                           // s61 = s[sgprWorkGroup0] / s65
v_sub_u32 v4, s[sgprWorkGroup0], v3                // s61 = s[sgprWorkGroup0] / s65
v_cmpx_ge_u32 exec, v4, s65                        // s61 = s[sgprWorkGroup0] / s65
v_add_u32 v2, v2, 1                                // s61 = s[sgprWorkGroup0] / s65
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v3, v2, s65                           // s61 = s[sgprWorkGroup0] / s65
v_sub_u32 v4, s[sgprWorkGroup0], v3                // s61 = s[sgprWorkGroup0] / s65
v_readfirstlane_b32 s61, v2                        // quotient
v_readfirstlane_b32 s62, v4                        // remainder
s_mul_i32 s61, s61, s65
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s62, s62, s64
/* temp0 = temp0 + temp1 */
s_add_u32 s61, s61, s62
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[2:3], s65                          // s62 = s59 / s65
v_rcp_f64 v[2:3], v[2:3]                           // s62 = s59 / s65
v_cvt_f64_u32 v[4:5], s59                          // s62 = s59 / s65
v_mul_f64 v[2:3], v[2:3], v[4:5]                   // s62 = s59 / s65
v_cvt_u32_f64 v2, v[2:3]                           // s62 = s59 / s65
v_mul_lo_u32 v3, v2, s65                           // s62 = s59 / s65
v_sub_u32 v4, s59, v3                              // s62 = s59 / s65
v_cmpx_ge_u32 exec, v4, s65                        // s62 = s59 / s65
v_add_u32 v2, v2, 1                                // s62 = s59 / s65
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s62, v2                        // quotient
s_mul_i32 s62, s62, s65
s_sub_u32 s63, s59, s62
s_cmp_gt_u32 s[sgprWorkGroup0], s62
s_cselect_b32 s62, s63, s65
s_lshr_b32 s62, s62, s64
s_bfm_b32 s63, s64, 0
s_and_b32 s63, s[sgprWorkGroup0], s63
s_mul_i32 s62, s62, s63
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s61, s62
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s57, 0
s_cbranch_scc0 label_MultiGemm
v_mov_b32 v3, MT0                                  // set MT0 into sgpr
v_mov_b32 v2, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v1, v3                               // v1 = ceil(v2 / v3)
v_rcp_iflag_f32 v1, v1                             // v1 = ceil(v2 / v3)
v_cvt_f32_u32 v4, v2                               // v1 = ceil(v2 / v3)
v_mul_f32 v1, v1, v4                               // v1 = ceil(v2 / v3)
v_cvt_u32_f32 v1, v1                               // v1 = ceil(v2 / v3)
v_mul_u32_u24 v4, v1, v3                           // v1 = ceil(v2 / v3)
v_sub_u32 v4, v2, v4                               // v1 = ceil(v2 / v3)
v_cmp_ne_u32 vcc, v4, 0                            // v1 = ceil(v2 / v3)
v_addc_co_u32 v1, vcc, v1, 0, vcc                  // ceil
v_mov_b32 v3, MT1                                  // set MT1 into sgpr
v_mov_b32 v2, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v1      // set back to numWorkGroup0
v_cvt_f32_u32 v1, v3                               // v1 = ceil(v2 / v3)
v_rcp_iflag_f32 v1, v1                             // v1 = ceil(v2 / v3)
v_cvt_f32_u32 v4, v2                               // v1 = ceil(v2 / v3)
v_mul_f32 v1, v1, v4                               // v1 = ceil(v2 / v3)
v_cvt_u32_f32 v1, v1                               // v1 = ceil(v2 / v3)
v_mul_u32_u24 v4, v1, v3                           // v1 = ceil(v2 / v3)
v_sub_u32 v4, v2, v4                               // v1 = ceil(v2 / v3)
v_cmp_ne_u32 vcc, v4, 0                            // v1 = ceil(v2 / v3)
v_addc_co_u32 v1, vcc, v1, 0, vcc                  // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v1      // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 76/0 bytes of kern args

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s58, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s59, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s58, s58, s59
v_cvt_f32_u32 v1, s58                              // s58 = s[sgprWorkGroup0] / s58
v_rcp_iflag_f32 v1, v1                             // s58 = s[sgprWorkGroup0] / s58
v_cvt_f32_u32 v2, s[sgprWorkGroup0]                // s58 = s[sgprWorkGroup0] / s58
v_mul_f32 v1, v1, v2                               // s58 = s[sgprWorkGroup0] / s58
v_cvt_u32_f32 v1, v1                               // s58 = s[sgprWorkGroup0] / s58
v_mul_u32_u24 v2, v1, s58                          // s58 = s[sgprWorkGroup0] / s58
v_sub_u32 v2, s[sgprWorkGroup0], v2                // s58 = s[sgprWorkGroup0] / s58
v_cmpx_eq_u32 exec, v2, s58                        // s58 = s[sgprWorkGroup0] / s58
v_add_u32 v1, 1, v1                                // s58 = s[sgprWorkGroup0] / s58
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s58                        // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s58, v1                        // quotient
s_mov_b32 s[sgprWorkGroup2], s58
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s58, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s58, s58, s[sgprWorkGroup2]
s_mul_i32 s58, s58, s59
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s58
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v1, s[sgprNumWorkGroups0]            // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v1, v1                             // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v2, s[sgprWorkGroup0]                // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v1, v1, v2                               // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v1, v1                               // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v2, v1, s[sgprNumWorkGroups0]        // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v2, s[sgprWorkGroup0], v2                // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v2, s[sgprNumWorkGroups0]      // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v1, 1, v1                                // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s[sgprNumWorkGroups0]      // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s58, v1                        // quotient
s_mov_b32 s[sgprWorkGroup1], s58
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s58, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s58
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 120
s_mul_i32 s64, s56, 4
s_mov_b64 s[58:59], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 228
s_mov_b32 s64, 0
s_mov_b64 s[58:59], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s65, 0
s_load_dwordx4 s[16:19], s[58:59], s64
s_cmpk_eq_u32 s56, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s62, s16, 6                             // s62 = s16 / 64
s_and_b32 s60, 63, s16                             // s60 = s16 % 64
s_addc_u32 s62, s62, 0
s_lshr_b32 s63, s17, 6                             // s63 = s17 / 64
s_and_b32 s60, 63, s17                             // s60 = s17 % 64
s_addc_u32 s63, s63, 0
s_mul_i32 s62, s62, s63
s_mul_i32 s62, s62, s18
s_and_b32 s63, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s62, s62, s63
s_add_u32 s65, s65, s62
s_cmp_lt_u32 s[sgprWorkGroup0], s65
s_cbranch_scc1 label_FOUND
s_add_u32 s64, s64, s15
s_load_dwordx4 s[16:19], s[58:59], s64
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s56
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s62, s16, 6                             // s62 = s16 / 64
s_and_b32 s60, 63, s16                             // s60 = s16 % 64
s_addc_u32 s62, s62, 0
s_lshr_b32 s63, s17, 6                             // s63 = s17 / 64
s_and_b32 s60, 63, s17                             // s60 = s17 % 64
s_addc_u32 s63, s63, 0
s_mul_i32 s62, s62, s63
s_mul_i32 s62, s62, s18
s_and_b32 s58, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s62, s62, s58
s_add_u32 s65, s65, s62

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s59, s14, 1
s_sub_u32 s58, s65, s62
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s58
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s56, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s59, s59, 120
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s59
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx8 s[36:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s59, s59, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s59
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_dwordx16 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx8 s[36:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_dword s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
// Read Beta
s_load_dword s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 128 // 128
label_LoadExternalStructEnd:
v_mov_b32 v3, MT0                                  // set MT0 into sgpr
v_mov_b32 v2, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v1, v3                               // v1 = ceil(v2 / v3)
v_rcp_iflag_f32 v1, v1                             // v1 = ceil(v2 / v3)
v_cvt_f32_u32 v4, v2                               // v1 = ceil(v2 / v3)
v_mul_f32 v1, v1, v4                               // v1 = ceil(v2 / v3)
v_cvt_u32_f32 v1, v1                               // v1 = ceil(v2 / v3)
v_mul_u32_u24 v4, v1, v3                           // v1 = ceil(v2 / v3)
v_sub_u32 v4, v2, v4                               // v1 = ceil(v2 / v3)
v_cmp_ne_u32 vcc, v4, 0                            // v1 = ceil(v2 / v3)
v_addc_co_u32 v1, vcc, v1, 0, vcc                  // ceil
v_mov_b32 v3, MT1                                  // set MT1 into sgpr
v_mov_b32 v2, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v1      // set back to numWorkGroup0
v_cvt_f32_u32 v1, v3                               // v1 = ceil(v2 / v3)
v_rcp_iflag_f32 v1, v1                             // v1 = ceil(v2 / v3)
v_cvt_f32_u32 v4, v2                               // v1 = ceil(v2 / v3)
v_mul_f32 v1, v1, v4                               // v1 = ceil(v2 / v3)
v_cvt_u32_f32 v1, v1                               // v1 = ceil(v2 / v3)
v_mul_u32_u24 v4, v1, v3                           // v1 = ceil(v2 / v3)
v_sub_u32 v4, v2, v4                               // v1 = ceil(v2 / v3)
v_cmp_ne_u32 vcc, v4, 0                            // v1 = ceil(v2 / v3)
v_addc_co_u32 v1, vcc, v1, 0, vcc                  // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v1      // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 76/0 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s58, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s59, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s58, s58, s59
v_cvt_f32_u32 v1, s58                              // s58 = s[sgprWorkGroup0] / s58
v_rcp_iflag_f32 v1, v1                             // s58 = s[sgprWorkGroup0] / s58
v_cvt_f32_u32 v2, s[sgprWorkGroup0]                // s58 = s[sgprWorkGroup0] / s58
v_mul_f32 v1, v1, v2                               // s58 = s[sgprWorkGroup0] / s58
v_cvt_u32_f32 v1, v1                               // s58 = s[sgprWorkGroup0] / s58
v_mul_u32_u24 v2, v1, s58                          // s58 = s[sgprWorkGroup0] / s58
v_sub_u32 v2, s[sgprWorkGroup0], v2                // s58 = s[sgprWorkGroup0] / s58
v_cmpx_eq_u32 exec, v2, s58                        // s58 = s[sgprWorkGroup0] / s58
v_add_u32 v1, 1, v1                                // s58 = s[sgprWorkGroup0] / s58
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s58                        // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s58, v1                        // quotient
s_mov_b32 s[sgprWorkGroup2], s58
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s58, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s58, s58, s[sgprWorkGroup2]
s_mul_i32 s58, s58, s59
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s58
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v1, s[sgprNumWorkGroups0]            // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v1, v1                             // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v2, s[sgprWorkGroup0]                // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v1, v1, v2                               // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v1, v1                               // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v2, v1, s[sgprNumWorkGroups0]        // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v2, s[sgprWorkGroup0], v2                // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v2, s[sgprNumWorkGroups0]      // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v1, 1, v1                                // s58 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s[sgprNumWorkGroups0]      // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s58, v1                        // quotient
s_mov_b32 s[sgprWorkGroup1], s58
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s58, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s58

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprSrdA, 56
.set sgprSrdMXSA, 60
.set sgprSrdB, 64
.set sgprSrdMXSB, 68
.set sgprShadowLimitA, 72
.set sgprShadowLimitB, 74
.set sgprGlobalReadIncsA, 76
.set sgprGlobalReadIncsMXSA, 77
.set sgprGlobalReadIncsB, 78
.set sgprGlobalReadIncsMXSB, 79

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:
// Scale StridesMXSA by 32
s_lshl_b32 s[sgprStridesMXSA], s[sgprStridesMXSA], 5
s_lshl_b32 s[sgprStridesMXSB], s[sgprStridesMXSB], 5
/* Number of subtiles for A: 2 */
/* Number of subtiles for B: 2 */

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_and_b32 s56, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s56, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s56, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s56, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v1, s56                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_rcp_iflag_f32 v1, v1                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_cvt_f32_u32 v2, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_mul_f32 v1, v1, v2                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_cvt_u32_f32 v1, v1                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_mul_u32_u24 v2, v1, s56                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_sub_u32 v2, s[sgprWorkGroup1], v2                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_cmpx_eq_u32 exec, v2, s56                        // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_add_u32 v1, 1, v1                                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s56
v_mov_b32 v2, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s56
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s56                        // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
v_mul_u32_u24 v2, v1, s56                          // re-calculate remainder
v_sub_u32 v2, s[sgprWorkGroup1], v2                // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v1          // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v2           // remainder
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
v_cvt_f32_u32 v1, s[sgprNumWorkGroups1]            // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v1, v1                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_f32_u32 v2, s[sgprWorkGroup1]                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v1, v1, v2                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_u32_f32 v1, v1                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v2, v1, s[sgprNumWorkGroups1]        // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_sub_u32 v2, s[sgprWorkGroup1], v2                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmpx_eq_u32 exec, v2, s[sgprNumWorkGroups1]      // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_u32 v1, 1, v1                                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v2, 0                                    // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s[sgprNumWorkGroups1]      // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
v_mul_u32_u24 v2, v1, s[sgprNumWorkGroups1]        // re-calculate remainder
v_sub_u32 v2, s[sgprWorkGroup1], v2                // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprGSUSumIdx], v1           // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // remainder
label_GSUWGMRR_End:
s_mov_b32 s[sgprGSULog2BpeC], 2
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 2
s_mov_b32 s[sgprGSULog2BpeD], 2
label_GSU_End:
/* WGM Calculation */
s_mov_b32 s56, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s56, s56                            // Restore WGM
s_cmp_gt_i32 s56, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s56, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s56, s56                                 // abs(WGM)
v_cvt_f64_u32 v[2:3], s56                          // s57 = s[sgprWorkGroup0] / s56
v_rcp_f64 v[2:3], v[2:3]                           // s57 = s[sgprWorkGroup0] / s56
v_cvt_f64_u32 v[4:5], s[sgprWorkGroup0]            // s57 = s[sgprWorkGroup0] / s56
v_mul_f64 v[2:3], v[2:3], v[4:5]                   // s57 = s[sgprWorkGroup0] / s56
v_cvt_u32_f64 v2, v[2:3]                           // s57 = s[sgprWorkGroup0] / s56
v_mul_lo_u32 v3, v2, s56                           // s57 = s[sgprWorkGroup0] / s56
v_sub_u32 v4, s[sgprWorkGroup0], v3                // s57 = s[sgprWorkGroup0] / s56
v_cmpx_ge_u32 exec, v4, s56                        // s57 = s[sgprWorkGroup0] / s56
v_add_u32 v2, v2, 1                                // s57 = s[sgprWorkGroup0] / s56
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s57, v2                        // quotient
s_mul_i32 s64, s57, s56                            // quotient * non-magic divisor
s_sub_u32 s64, s[sgprWorkGroup0], s64              // WorkGroup0=remainder
s_mul_i32 s64, s64, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s64, s64, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[2:3], s56                          // s58 = s[sgprNumWorkGroups0] / s56
v_rcp_f64 v[2:3], v[2:3]                           // s58 = s[sgprNumWorkGroups0] / s56
v_cvt_f64_u32 v[4:5], s[sgprNumWorkGroups0]        // s58 = s[sgprNumWorkGroups0] / s56
v_mul_f64 v[2:3], v[2:3], v[4:5]                   // s58 = s[sgprNumWorkGroups0] / s56
v_cvt_u32_f64 v2, v[2:3]                           // s58 = s[sgprNumWorkGroups0] / s56
v_mul_lo_u32 v3, v2, s56                           // s58 = s[sgprNumWorkGroups0] / s56
v_sub_u32 v4, s[sgprNumWorkGroups0], v3            // s58 = s[sgprNumWorkGroups0] / s56
v_cmpx_ge_u32 exec, v4, s56                        // s58 = s[sgprNumWorkGroups0] / s56
v_add_u32 v2, v2, 1                                // s58 = s[sgprNumWorkGroups0] / s56
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s58, v2                        // quotient
s_mul_i32 s59, s56, s58                            // quotient * non-magic divisor
s_sub_u32 s59, s[sgprNumWorkGroups0], s59          // NumWorkGroups0=remainder
s_cmp_eq_u32 s59, 0                                // remainder == 0 ?
s_cmov_b32 s59, s56                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s57, s58                              // blockId >= numFullBlocks ?
s_cselect_b32 s58, s59, s56
v_cvt_f64_u32 v[2:3], s58                          // s[sgprWorkGroup1] = s64 / s58
v_rcp_f64 v[2:3], v[2:3]                           // s[sgprWorkGroup1] = s64 / s58
v_cvt_f64_u32 v[4:5], s64                          // s[sgprWorkGroup1] = s64 / s58
v_mul_f64 v[2:3], v[2:3], v[4:5]                   // s[sgprWorkGroup1] = s64 / s58
v_cvt_u32_f64 v2, v[2:3]                           // s[sgprWorkGroup1] = s64 / s58
v_mul_lo_u32 v3, v2, s58                           // s[sgprWorkGroup1] = s64 / s58
v_sub_u32 v4, s64, v3                              // s[sgprWorkGroup1] = s64 / s58
v_cmpx_ge_u32 exec, v4, s58                        // s[sgprWorkGroup1] = s64 / s58
v_add_u32 v2, v2, 1                                // s[sgprWorkGroup1] = s64 / s58
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v3, v2, s58                           // s[sgprWorkGroup1] = s64 / s58
v_sub_u32 v4, s64, v3                              // s[sgprWorkGroup1] = s64 / s58
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v4          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s58 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s64, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s57, s57, s56                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s57 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s56, s56                                 // WGM
v_cvt_f64_u32 v[2:3], s56                          // s57 = s[sgprWorkGroup1] / s56
v_rcp_f64 v[2:3], v[2:3]                           // s57 = s[sgprWorkGroup1] / s56
v_cvt_f64_u32 v[4:5], s[sgprWorkGroup1]            // s57 = s[sgprWorkGroup1] / s56
v_mul_f64 v[2:3], v[2:3], v[4:5]                   // s57 = s[sgprWorkGroup1] / s56
v_cvt_u32_f64 v2, v[2:3]                           // s57 = s[sgprWorkGroup1] / s56
v_mul_lo_u32 v3, v2, s56                           // s57 = s[sgprWorkGroup1] / s56
v_sub_u32 v4, s[sgprWorkGroup1], v3                // s57 = s[sgprWorkGroup1] / s56
v_cmpx_ge_u32 exec, v4, s56                        // s57 = s[sgprWorkGroup1] / s56
v_add_u32 v2, v2, 1                                // s57 = s[sgprWorkGroup1] / s56
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s57, v2                        // quotient
s_mul_i32 s64, s57, s56                            // quotient * non-magic divisor
s_sub_u32 s64, s[sgprWorkGroup1], s64              // WorkGroup1=remainder
s_mul_i32 s64, s64, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s64, s64, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[2:3], s56                          // s58 = s[sgprNumWorkGroups1] / s56
v_rcp_f64 v[2:3], v[2:3]                           // s58 = s[sgprNumWorkGroups1] / s56
v_cvt_f64_u32 v[4:5], s[sgprNumWorkGroups1]        // s58 = s[sgprNumWorkGroups1] / s56
v_mul_f64 v[2:3], v[2:3], v[4:5]                   // s58 = s[sgprNumWorkGroups1] / s56
v_cvt_u32_f64 v2, v[2:3]                           // s58 = s[sgprNumWorkGroups1] / s56
v_mul_lo_u32 v3, v2, s56                           // s58 = s[sgprNumWorkGroups1] / s56
v_sub_u32 v4, s[sgprNumWorkGroups1], v3            // s58 = s[sgprNumWorkGroups1] / s56
v_cmpx_ge_u32 exec, v4, s56                        // s58 = s[sgprNumWorkGroups1] / s56
v_add_u32 v2, v2, 1                                // s58 = s[sgprNumWorkGroups1] / s56
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s58, v2                        // quotient
s_mul_i32 s59, s56, s58                            // quotient * non-magic divisor
s_sub_u32 s59, s[sgprNumWorkGroups1], s59          // NumWorkGroups1=remainder
s_cmp_eq_u32 s59, 0                                // remainder == 0 ?
s_cmov_b32 s59, s56                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s57, s58                              // blockId >= numFullBlocks ?
s_cselect_b32 s58, s59, s56
v_cvt_f64_u32 v[2:3], s58                          // s[sgprWorkGroup0] = s64 / s58
v_rcp_f64 v[2:3], v[2:3]                           // s[sgprWorkGroup0] = s64 / s58
v_cvt_f64_u32 v[4:5], s64                          // s[sgprWorkGroup0] = s64 / s58
v_mul_f64 v[2:3], v[2:3], v[4:5]                   // s[sgprWorkGroup0] = s64 / s58
v_cvt_u32_f64 v2, v[2:3]                           // s[sgprWorkGroup0] = s64 / s58
v_mul_lo_u32 v3, v2, s58                           // s[sgprWorkGroup0] = s64 / s58
v_sub_u32 v4, s64, v3                              // s[sgprWorkGroup0] = s64 / s58
v_cmpx_ge_u32 exec, v4, s58                        // s[sgprWorkGroup0] = s64 / s58
v_add_u32 v2, v2, 1                                // s[sgprWorkGroup0] = s64 / s58
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v3, v2, s58                           // s[sgprWorkGroup0] = s64 / s58
v_sub_u32 v4, s64, v3                              // s[sgprWorkGroup0] = s64 / s58
v_readfirstlane_b32 s[sgprWorkGroup0], v2          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v4          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s58 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s64, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s57, s57, s56                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s57 // wg1 += blockId * WGM
label_WGM:

/******************************************/
/* End setupNewTile                       */
/******************************************/
/* Compute shared offsets used by m0 in DTL loads */
v_lshrrev_b32 v1, 0x6, v[vgprSerial]               // Wave Id
s_mov_b32 s80, 32                                  // A: row offset
v_and_b32 v4, 0x1, v1                              // A: waveId % 2
v_lshrrev_b32 v5, 0x1, v1                          // A: waveId / 2
v_lshlrev_b32 v4, 0x3, v4                          // A: local row offset
v_mul_lo_u32 v5, s80, v5                           // A: wave row offset
v_add_u32 v2, v4, v5                               // A: row offset
s_mov_b32 s80, 32                                  // B: row offset
v_and_b32 v4, 0x1, v1                              // B: waveId % 2
v_lshrrev_b32 v5, 0x1, v1                          // B: waveId / 2
v_lshlrev_b32 v4, 0x3, v4                          // B: local row offset
v_mul_lo_u32 v5, s80, v5                           // B: wave row offset
v_add_u32 v3, v4, v5                               // B: row offset
v_lshlrev_b32 v2, 0x7, v2                          // Apply wave-specific offset for A
v_lshlrev_b32 v3, 0x7, v3                          // Apply wave-specific offset for B
s_nop 0                                            // Wait for VGPR to be ready
v_readfirstlane_b32 s[sgprLocalWriteBaseAddrA], v2 // Store base LDS offset, will be modified
v_readfirstlane_b32 s[sgprLocalWriteBaseAddrB], v3 // Store base LDS offset, will be modified
s_add_u32 s[sgprLocalWriteBaseAddrB], s[sgprLocalWriteBaseAddrB], 0x2000
s_add_u32 s[sgprLocalWriteSwapA], s[sgprLocalWriteBaseAddrA], 24576
s_xor_b32 s[sgprLocalWriteSwapA], s[sgprLocalWriteBaseAddrA], s[sgprLocalWriteSwapA]
s_add_u32 s[sgprLocalWriteSwapB], s[sgprLocalWriteBaseAddrB], 24576
s_xor_b32 s[sgprLocalWriteSwapB], s[sgprLocalWriteBaseAddrB], s[sgprLocalWriteSwapB]
/* Compute shared offsets used by m0 in DTL loads */
v_lshrrev_b32 v1, 0x6, v[vgprSerial]               // Wave Id
v_lshlrev_b32 v1, 0xa, v1                          // Apply wave-specific common offset (1024) for A/B
s_nop 0                                            // Wait for VGPR to be ready
v_readfirstlane_b32 s[sgprLocalWriteBaseAddrMXSA], v1 // Store base LDS offset, will be modified
v_readfirstlane_b32 s[sgprLocalWriteBaseAddrMXSB], v1 // Store base LDS offset, will be modified
s_add_u32 s[sgprLocalWriteBaseAddrMXSA], s[sgprLocalWriteBaseAddrMXSA], 0x4000
s_add_u32 s[sgprLocalWriteBaseAddrMXSB], s[sgprLocalWriteBaseAddrMXSB], 0x5000
s_add_u32 s[sgprLocalWriteSwapMXSA], s[sgprLocalWriteBaseAddrMXSA], 24576
s_xor_b32 s[sgprLocalWriteSwapMXSA], s[sgprLocalWriteBaseAddrMXSA], s[sgprLocalWriteSwapMXSA]
s_add_u32 s[sgprLocalWriteSwapMXSB], s[sgprLocalWriteBaseAddrMXSB], 24576
s_xor_b32 s[sgprLocalWriteSwapMXSB], s[sgprLocalWriteBaseAddrMXSB], s[sgprLocalWriteSwapMXSB]

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s83, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s82, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
s_mul_hi_u32 s83, s82, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s82, s82, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_and_b32 s80, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s81, 256, s[sgprGSUSumIdx]            // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s80, 256, s[sgprGSUSumIdx]               // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 8 // s[LoopCounterL] = s[sgprSizesSum] / 256
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v1, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v1, v1                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v2, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v1, v1, v2                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v1, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v2, v1, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v2, s[sgprLoopCounterL], v2              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v2, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v1, 1, v1                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v2, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s[sgprGSUSumIdx+1]         // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
v_mul_u32_u24 v2, v1, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_u32 v2, s[sgprLoopCounterL], v2              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v1        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v2         // remainder
s_mul_i32 s81, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s80, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s81, s81, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s80, s80, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s80, s80, s81                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s81, s80, 256                         // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s80, s80, 256                            // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_add_u32 s82, s82, s80                            // accum GsuOffset term to tilestart
s_addc_u32 s83, s83, s81                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s80, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s81, constStrideAL, s80               // stride x (size-1)
s_mul_i32 s80, constStrideAL, s80                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s80 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s81 // sum tensor size
s_sub_u32 s80, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s81, s[sgprStrideA0I], s80            // stride x (size-1)
s_mul_i32 s80, s[sgprStrideA0I], s80               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s80 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s81 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s82 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s83 // sub tileStart
s_lshr_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiply bpe 0.500000)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s81, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s80, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s82, s82, s80                            // accum wg term to tilestart
s_addc_u32 s83, s83, s81                           // accum wg term to tilestart
s_lshr_b64 s[82:83], s[82:83], 1                   // tileStart (multiply bpe 0.500000)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s82    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s83   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses mxsa */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s83, s[sgprWorkGroup0], 2             // WorkGroup[01] * roundup(MT/32)
s_mul_i32 s82, s[sgprWorkGroup0], 2                // WorkGroup[01] * roundup(MT/32)
s_add_u32 s80, s[sgprSizeI], 31                    // size + 32 - 1
s_lshr_b32 s80, s80, 5                             // roundup(size/32)
s_sub_u32 s80, s80, s82                            // numBlkToEnd = roundUp(size/32) - (WorkGroup[01] * roundup(MT/32))
s_min_u32 s80, s80, 2                              // min (numBlkToEnd, roundup(MT/32))
s_sub_u32 s80, s80, 1                              // numLine = min (numBlkToEnd, roundup(MT/32)) - 1
s_mul_hi_u32 s81, s80, s[sgprStrideMXSA0I]         // scaled by stride
s_mul_i32 s80, s80, s[sgprStrideMXSA0I]            // scaled by stride
s_add_u32 s[sgprSrdMXSA+2], s80, 256               // buffer_load limit for MXSA
s_mul_hi_u32 s83, s82, s[sgprStrideMXSA0I]         // tlu=0, scaled tile-offset by stride
s_mul_i32 s82, s82, s[sgprStrideMXSA0I]            // tlu=0, scaled tile-offset by stride
s_and_b32 s80, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_M                        // branch if GSUC == 1
s_mul_hi_u32 s81, 8, s[sgprGSUSumIdx]              // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s80, 8, s[sgprGSUSumIdx]                 // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_M_End
label_GSUC_M:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 8 // s[LoopCounterL] = s[sgprSizesSum] / 256
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v1, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v1, v1                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v2, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v1, v1, v2                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v1, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v2, v1, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v2, s[sgprLoopCounterL], v2              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v2, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v1, 1, v1                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v2, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s[sgprGSUSumIdx+1]         // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
v_mul_u32_u24 v2, v1, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_u32 v2, s[sgprLoopCounterL], v2              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v1        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v2         // remainder
s_mul_i32 s81, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s80, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s81, s81, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s80, s80, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s80, s80, s81                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s81, s80, 8                           // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s80, s80, 8                              // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_M_End:
s_add_u32 s82, s82, s80                            // accum GsuOffset term to tilestart
s_addc_u32 s83, s83, s81                           // accum GsuOffset term to tilestart
s_mul_hi_u32 s81, s[sgprStrideMXSAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s80, s[sgprStrideMXSAK], s[sgprWorkGroup2] // Stride*WG
s_add_u32 s82, s82, s80                            // accum wg term to tilestart
s_addc_u32 s83, s83, s81                           // accum wg term to tilestart
                                                   // tileStart (multiply bpe 1.000000) (bpe is 1, do nothing)
s_add_u32 s[sgprSrdMXSA+0], s[sgprAddressMXSA+0], s82 // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdMXSA+1], s[sgprAddressMXSA+1], s83 // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdMXSA+3], Srd127_96              // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s83, s[sgprWorkGroup1], 64            // WorkGroup[01] * MT
s_mul_i32 s82, s[sgprWorkGroup1], 64               // WorkGroup[01] * MT
s_mul_hi_u32 s83, s82, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s82, s82, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s80, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s81, 256, s[sgprGSUSumIdx]            // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s80, 256, s[sgprGSUSumIdx]               // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 8 // s[LoopCounterL] = s[sgprSizesSum] / 256
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v1, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v1, v1                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v2, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v1, v1, v2                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v1, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v2, v1, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v2, s[sgprLoopCounterL], v2              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v2, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v1, 1, v1                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v2, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s[sgprGSUSumIdx+1]         // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
v_mul_u32_u24 v2, v1, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_u32 v2, s[sgprLoopCounterL], v2              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v1        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v2         // remainder
s_mul_i32 s81, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s80, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s81, s81, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s80, s80, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s80, s80, s81                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s81, s80, 256                         // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s80, s80, 256                            // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_B_End:
s_add_u32 s82, s82, s80                            // accum GsuOffset term to tilestart
s_addc_u32 s83, s83, s81                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s80, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s81, constStrideBL, s80               // stride x (size-1)
s_mul_i32 s80, constStrideBL, s80                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s80 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s81 // sum tensor size
s_sub_u32 s80, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s81, s[sgprStrideB1J], s80            // stride x (size-1)
s_mul_i32 s80, s[sgprStrideB1J], s80               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s80 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s81 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s82 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s83 // sub tileStart
s_lshr_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiply bpe 0.500000)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s81, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s80, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s82, s82, s80                            // accum wg term to tilestart
s_addc_u32 s83, s83, s81                           // accum wg term to tilestart
s_lshr_b64 s[82:83], s[82:83], 1                   // tileStart (multiply bpe 0.500000)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s82    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s83   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses mxsb */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s83, s[sgprWorkGroup1], 2             // WorkGroup[01] * roundup(MT/32)
s_mul_i32 s82, s[sgprWorkGroup1], 2                // WorkGroup[01] * roundup(MT/32)
s_add_u32 s80, s[sgprSizeJ], 31                    // size + 32 - 1
s_lshr_b32 s80, s80, 5                             // roundup(size/32)
s_sub_u32 s80, s80, s82                            // numBlkToEnd = roundUp(size/32) - (WorkGroup[01] * roundup(MT/32))
s_min_u32 s80, s80, 2                              // min (numBlkToEnd, roundup(MT/32))
s_sub_u32 s80, s80, 1                              // numLine = min (numBlkToEnd, roundup(MT/32)) - 1
s_mul_hi_u32 s81, s80, s[sgprStrideMXSB1J]         // scaled by stride
s_mul_i32 s80, s80, s[sgprStrideMXSB1J]            // scaled by stride
s_add_u32 s[sgprSrdMXSB+2], s80, 256               // buffer_load limit for MXSB
s_mul_hi_u32 s83, s82, s[sgprStrideMXSB1J]         // tlu=0, scaled tile-offset by stride
s_mul_i32 s82, s82, s[sgprStrideMXSB1J]            // tlu=0, scaled tile-offset by stride
s_and_b32 s80, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_M_1                      // branch if GSUC == 1
s_mul_hi_u32 s81, 8, s[sgprGSUSumIdx]              // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s80, 8, s[sgprGSUSumIdx]                 // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_M_End_1
label_GSUC_M_1:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 8 // s[LoopCounterL] = s[sgprSizesSum] / 256
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v1, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v1, v1                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v2, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v1, v1, v2                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v1, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v2, v1, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v2, s[sgprLoopCounterL], v2              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v2, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v1, 1, v1                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v2, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v2, s[sgprGSUSumIdx+1]         // overflow happened in remainder
v_sub_u32 v1, v1, 1                                // quotient - 1
v_mul_u32_u24 v2, v1, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_u32 v2, s[sgprLoopCounterL], v2              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v1        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v2         // remainder
s_mul_i32 s81, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s80, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s81, s81, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s80, s80, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s80, s80, s81                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s81, s80, 8                           // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s80, s80, 8                              // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_M_End_1:
s_add_u32 s82, s82, s80                            // accum GsuOffset term to tilestart
s_addc_u32 s83, s83, s81                           // accum GsuOffset term to tilestart
s_mul_hi_u32 s81, s[sgprStrideMXSBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s80, s[sgprStrideMXSBK], s[sgprWorkGroup2] // Stride*WG
s_add_u32 s82, s82, s80                            // accum wg term to tilestart
s_addc_u32 s83, s83, s81                           // accum wg term to tilestart
                                                   // tileStart (multiply bpe 1.000000) (bpe is 1, do nothing)
s_add_u32 s[sgprSrdMXSB+0], s[sgprAddressMXSB+0], s82 // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdMXSB+1], s[sgprAddressMXSB+1], s83 // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdMXSB+3], Srd127_96              // Set bits 127_96 in SRD
/* TileInfo(tc=A)
  mmaTileShape:           [16, 128]
  mmaTileSize:            1024 bytes
  mmaTileRegCount:        4.0
  mmaTileLocalTotalCount: 4
  subtileShape:           [1, 2]
  subtileSize:            2048 bytes
  subtileLocalTotalCount: 2
  globalMMATileGrid:      [4, 2]
  globalSubtileGrid:      [4, 1]
  localMMATileGrid:       [2, 2]
  localSubtileGrid:       [2, 1]
  mxBlock:                32
  loadRatioGR:            1.0
  numGRPerSubtile:        1
  numGRTotal:             2
  loadRatioLR:            0.5
  numLRPerSubtile:        2
  numLRTotal:             4
  vgprTileFactor:         1.0
  vgprTiles:              not allocated
  sharedVgprGROffset:     not allocated
  sharedVgprLROffset:     not allocated */
/* TileInfo(tc=B)
  mmaTileShape:           [16, 128]
  mmaTileSize:            1024 bytes
  mmaTileRegCount:        4.0
  mmaTileLocalTotalCount: 4
  subtileShape:           [1, 2]
  subtileSize:            2048 bytes
  subtileLocalTotalCount: 2
  globalMMATileGrid:      [4, 2]
  globalSubtileGrid:      [4, 1]
  localMMATileGrid:       [2, 2]
  localSubtileGrid:       [2, 1]
  mxBlock:                32
  loadRatioGR:            1.0
  numGRPerSubtile:        1
  numGRTotal:             2
  loadRatioLR:            0.5
  numLRPerSubtile:        2
  numLRTotal:             4
  vgprTileFactor:         1.0
  vgprTiles:              not allocated
  sharedVgprGROffset:     not allocated
  sharedVgprLROffset:     not allocated */
/* TileInfo(tc=MXSA)
  mmaTileShape:           [16, 4]
  mmaTileSize:            64 bytes
  mmaTileRegCount:        0.25
  mmaTileLocalTotalCount: 4
  subtileShape:           [2, 2]
  subtileSize:            256 bytes
  subtileLocalTotalCount: 1
  globalMMATileGrid:      [4, 2]
  globalSubtileGrid:      [2, 1]
  localMMATileGrid:       [2, 2]
  localSubtileGrid:       [1, 1]
  mxBlock:                32
  loadRatioGR:            8.0
  numGRPerSubtile:        1
  numGRTotal:             0
  loadRatioLR:            4.0
  numLRPerSubtile:        1
  numLRTotal:             0
  vgprTileFactor:         1.0
  vgprTiles:              not allocated
  sharedVgprGROffset:     not allocated
  sharedVgprLROffset:     not allocated */
/* TileInfo(tc=MXSB)
  mmaTileShape:           [16, 4]
  mmaTileSize:            64 bytes
  mmaTileRegCount:        0.25
  mmaTileLocalTotalCount: 4
  subtileShape:           [2, 2]
  subtileSize:            256 bytes
  subtileLocalTotalCount: 1
  globalMMATileGrid:      [4, 2]
  globalSubtileGrid:      [2, 1]
  localMMATileGrid:       [2, 2]
  localSubtileGrid:       [1, 1]
  mxBlock:                32
  loadRatioGR:            8.0
  numGRPerSubtile:        1
  numGRTotal:             0
  loadRatioLR:            4.0
  numLRPerSubtile:        1
  numLRTotal:             0
  vgprTileFactor:         1.0
  vgprTiles:              not allocated
  sharedVgprGROffset:     not allocated
  sharedVgprLROffset:     not allocated */
/* TileInfo(tc=D)
  mmaTileShape:           [16, 16]
  mmaTileSize:            1024 bytes
  mmaTileRegCount:        4.0
  mmaTileLocalTotalCount: 4
  subtileShape:           [1, 2]
  subtileSize:            1024 bytes
  subtileLocalTotalCount: 4
  globalMMATileGrid:      [4, 4]
  globalSubtileGrid:      [4, 4]
  localMMATileGrid:       [2, 2]
  localSubtileGrid:       [2, 2]
  mxBlock:                0
  loadRatioGR:            0
  numGRPerSubtile:        0
  numGRTotal:             0
  loadRatioLR:            0
  numLRPerSubtile:        0
  numLRTotal:             0
  vgprTileFactor:         1.0
  vgprTiles:              not allocated
  sharedVgprGROffset:     not allocated
  sharedVgprLROffset:     not allocated */
// Allocating v[1] for A GR
// Allocating v[2, 4] for A LR
// Allocating v[3, 5] for A LR Swap
// Allocating v[6] for B GR
// Allocating v[7, 9] for B LR
// Allocating v[8, 10] for B LR Swap
// Allocating v[11] for MXSA GR
// Allocating v[12] for MXSA LR
// Allocating v[13] for MXSA LR Swap
// Allocating v[14] for MXSB GR
// Allocating v[15] for MXSB LR
// Allocating v[16] for MXSB LR Swap
/* Using As for [] GR, subtile: [0, 0] */
/* Using As for [80] GR, subtile: [1, 0] */
/* Using Bs for [] GR, subtile: [0, 0] */
/* Using Bs for [81] GR, subtile: [1, 0] */
/* GR Offset Calculation for Subtile Based Tiling */
v_lshrrev_b32 v22, 0x6, v[vgprSerial]              // Wave Id
v_and_b32 v23, v[vgprSerial], 63
v_and_b32 v17, v[vgprSerial], 7                    // colIdA base
v_and_b32 v18, v[vgprSerial], 7                    // colIdB base
v_lshrrev_b32 v19, 0x3, v23                        // row id within wave
/* Swizzling colIdA,B */
v_lshrrev_b32 v24, 0x3, v23                        // row id within wave
v_lshrrev_b32 v24, 0x1, v24                        // lds row id
v_and_b32 v25, v24, 0x1                            // lds row id % 2
v_cmpx_eq_u32 vcc, 0, v25                          // lds row id % 2 == 0 ?
v_mov_b32 v17, v17 quad_perm:[1,0,3,2]             // swap colId pairs for swizzling
s_mov_b64 exec, -1
v_mov_b32 v18, v17                                 // copy swizzled colId
/* Rotation within a single wave */
v_lshrrev_b32 v25, 0x1, v24
v_lshlrev_b32 v25, 0x1, v25                        // (ldsRowId //2) * 2
v_sub_u32 v25, 0x8, v25                            // rotation offset : blockSize - (ldsRowId//2)*2
/* Rotation per wave */
v_and_b32 v26, v22, 0x1
v_lshlrev_b32 v26, 0x2, v26
v_sub_u32 v26, v25, v26
v_add_u32 v17, v26, v17
/* Rotation per wave */
v_and_b32 v26, v22, 0x1
v_lshlrev_b32 v26, 0x2, v26
v_sub_u32 v26, v25, v26
v_add_u32 v18, v26, v18
v_and_b32 v17, v17, 0x7                            // (col + offset) % block_size
v_and_b32 v18, v18, 0x7                            // (col + offset) % block_size
s_mov_b32 s82, 32                                  // A: row offset
v_and_b32 v24, 0x1, v22                            // A: waveId % 2
v_lshrrev_b32 v25, 0x1, v22                        // A: waveId / 2
v_lshlrev_b32 v24, 0x3, v24                        // A: local row offset
v_mul_lo_u32 v25, s82, v25                         // A: wave row offset
v_add_u32 v20, v24, v25                            // A: row offset
s_mov_b32 s82, 32                                  // B: row offset
v_and_b32 v24, 0x1, v22                            // B: waveId % 2
v_lshrrev_b32 v25, 0x1, v22                        // B: waveId / 2
v_lshlrev_b32 v24, 0x3, v24                        // B: local row offset
v_mul_lo_u32 v25, s82, v25                         // B: wave row offset
v_add_u32 v21, v24, v25                            // B: row offset
v_add_u32 v20, v19, v20                            // A: row offset
v_lshlrev_b32 v25, 0x4, v17                        // scale col_id by load_width
v_mul_lo_u32 v24, s[sgprStrideA0I], v20            // A: rowId * stride
v_lshlrev_b32 v24, 0x2, v24                        // A: rowId*stride*bpe
v_lshrrev_b32 v24, 0x3, v24                        // to bytes
v_add_u32 v1, v25, v24                             // A: GR row_offset
v_add_u32 v21, v19, v21                            // B: row offset
v_lshlrev_b32 v25, 0x4, v18                        // scale col_id by load_width
v_mul_lo_u32 v24, s[sgprStrideB1J], v21            // B: rowId * stride
v_lshlrev_b32 v24, 0x2, v24                        // B: rowId*stride*bpe
v_lshrrev_b32 v24, 0x3, v24                        // to bytes
v_add_u32 v6, v25, v24                             // B: GR row_offset
s_mul_i32 s80, 0x8, s[sgprStrideA0I]               // A: group 1 offset = 8 * StrideA0I
s_mul_i32 s81, 0x8, s[sgprStrideB1J]               // B: group 1 offset = 8 * StrideB1J
/* LR Offset Calculation for Subtile Based Tiling */
v_and_b32 v18, v[vgprSerial], 63                   // laneId
v_lshrrev_b32 v18, 0x4, v18                        // lane16Group
v_and_b32 v17, v[vgprSerial], 15                   // laneId % 16
v_lshrrev_b32 v19, 0x1, v17                        // lds_row_id
v_lshrrev_b32 v19, 0x1, v19                        // (lds_row_id //2 )
v_lshlrev_b32 v19, 0x1, v19                        // rotation=(lds_row_id //2) * 2
v_add_u32 v21, v19, v18                            // colOffset = rotation + lane16Group
v_and_b32 v21, v21, 0x7                            // colOffset = colOffset % blockSize
s_mov_b32 s82, 0x33333333                          // exec mask lo
s_mov_b32 s83, 0x33333333                          // exec mask hi
s_mov_b64 exec, s[82:83]                           // Set exec mask
v_permlane16_swap_b32 v21, v21                     // apply swizzling
s_mov_b32 s82, -0x1                                // exec mask lo
s_mov_b32 s83, -0x1                                // exec mask hi
s_mov_b64 exec, s[82:83]                           // Set exec mask
v_lshlrev_b32 v20, 0x7, v17                        // offsetRow = depthUBytes*lane16
v_mov_b32 v2, v21                                  // A: laneId
v_add_u32 v4, v2, 0x4                              // A: colOffset for MFMA 1 of subtile
v_and_b32 v4, v4, 0x7                              // A: colOffset = colOffset % block_size
v_lshlrev_b32 v2, 0x4, v2                          // A: colOffset*loadWidth
v_add_u32 v2, v2, v20                              // A: row + col
v_lshlrev_b32 v4, 0x4, v4                          // A: colOffset*loadWidth
v_add_u32 v4, v4, v20                              // A: row + col
v_mov_b32 v7, v21                                  // B: laneId
v_add_u32 v9, v7, 0x4                              // B: colOffset for MFMA 1 of subtile
v_and_b32 v9, v9, 0x7                              // B: colOffset = colOffset % block_size
v_lshlrev_b32 v7, 0x4, v7                          // B: colOffset*loadWidth
v_add_u32 v7, v7, v20                              // B: row + col
v_lshlrev_b32 v9, 0x4, v9                          // B: colOffset*loadWidth
v_add_u32 v9, v9, v20                              // B: row + col
v_lshrrev_b32 v17, 0x6, v[vgprSerial]              // waveId
v_and_b32 v17, 0x1, v17                            // A: waveId % 2
s_mov_b32 s82, 0x1000                              // A: interleave stride
v_mul_lo_u32 v17, s82, v17
v_add_u32 v2, v2, v17                              // A: wave partition LR offset
v_add_u32 v4, v4, v17                              // A: wave partition LR offset
v_lshrrev_b32 v17, 0x6, v[vgprSerial]              // waveId
v_lshrrev_b32 v17, 0x1, v17                        // B: waveId / 2
s_mov_b32 s82, 0x1000                              // B: interleave stride
v_mul_lo_u32 v17, s82, v17
v_add_u32 v7, v7, v17                              // B: wave partition LR offset
v_add_u32 v9, v9, v17                              // B: wave partition LR offset
v_add_u32 v7, 8192, v7                             // B matrix offset in LDS
v_add_u32 v9, 8192, v9                             // B matrix offset in LDS
s_mov_b32 s82, 24576                               // Store Total Lds Size for one buffer
v_add_u32 v3, v2, s82
v_xor_b32 v3, v2, v3
v_add_u32 v5, v4, s82
v_xor_b32 v5, v4, v5
v_add_u32 v8, v7, s82
v_xor_b32 v8, v7, v8
v_add_u32 v10, v9, s82
v_xor_b32 v10, v9, v10
// Computing GR Offset for MXSA
v_lshrrev_b32 v17, 0x4, v[vgprSerial]              // MXSA: grOffset = serial / 16
s_lshl_b32 s82, s[sgprStridesMXSA], 0              // *= bpe (1)
v_mul_lo_u32 v17, s82, v17                         // Apply scaleMXSA stride to each group
v_and_b32 v11, 0xf, v[vgprSerial]                  // MXSA: grOffset = serial % 16
v_lshlrev_b32 v11, 0x4, v11                        // Scale by load width for each thread in group
v_add_u32 v11, v11, v17                            // Final offset calc
// Computing GR Offset for MXSB
v_lshrrev_b32 v17, 0x4, v[vgprSerial]              // MXSB: grOffset = serial / 16
s_lshl_b32 s82, s[sgprStridesMXSB], 0              // *= bpe (1)
v_mul_lo_u32 v17, s82, v17                         // Apply scaleMXSB stride to each group
v_and_b32 v14, 0xf, v[vgprSerial]                  // MXSB: grOffset = serial % 16
v_lshlrev_b32 v14, 0x4, v14                        // Scale by load width for each thread in group
v_add_u32 v14, v14, v17                            // Final offset calc
/* LR Offset Calculation for Scale Tensors */
v_lshrrev_b32 v17, 0x6, v[vgprSerial]              // scale: waveId
v_and_b32 v18, 1, v17                              // scaleMXSA: waveId % 2
s_mov_b32 s82, 256                                 // scaleMXSA: scale region
v_mul_lo_u32 v12, s82, v18                         // scaleMXSA: partition offset
v_lshrrev_b32 v18, 1, v17                          // scaleMXSB: waveId / numWavesM
s_mov_b32 s82, 256                                 // scaleMXSB: scale region
v_mul_lo_u32 v15, s82, v18                         // scaleMXSB: partition offset
v_and_b32 v17, v[vgprSerial], 63                   // scale: laneId
v_lshlrev_b32 v17, 0x2, v17                        // scale: laneId * 4
v_add_u32 v12, v17, v12                            // scaleA: lrOffset = laneId * 4
v_add_u32 v15, v17, v15                            // scaleB: lrOffset = laneId * 4
s_mov_b32 s82, 0x4000                              // scale: LDS offset for A scale
v_add_u32 v12, v12, s82                            // scaleA: +=LDS offset
s_mov_b32 s82, 0x5000                              // scale: LDS offset for B scale
v_add_u32 v15, v15, s82                            // scaleB: +=LDS offset
s_mov_b32 s82, 24576                               // scale: total LDS size for swap
v_add_u32 v13, v12, s82                            // scaleMXSA: LR swap
v_xor_b32 v13, v12, v13                            // scaleMXSA: LR swap
v_add_u32 v16, v15, s82                            // scaleMXSB: LR swap
v_xor_b32 v16, v15, v16                            // scaleMXSB: LR swap
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 8 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 256
s_and_b32 s82, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s82, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v17, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v17, v17                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v18, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v17, v17, v18                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v17, v17                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v18, v17, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v18, s[sgprLoopCounterL], v18            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v18, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v17, 1, v17                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v18, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v18, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v17, v17, 1                              // quotient - 1
v_mul_u32_u24 v18, v17, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v18, s[sgprLoopCounterL], v18            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v17       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v18        // remainder
s_add_u32 s82, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s82                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
/* Init D vgprTiles to zero */
v_mov_b64 v[52:53], 0
s_nop 1                                            // wait for vgpr to be ready before MFMA
v_mfma_i32_32x32x16_i8 acc[0:15], v[52:53], v[52:53], 0 // initD: [0:15]
// Vgprs used for A mma tile 0: [20, 21, 22, 23]
// Vgprs used for A mma tile 1: [24, 25, 26, 27]
// Vgprs used for A mma tile 2: [28, 29, 30, 31]
// Vgprs used for A mma tile 3: [32, 33, 34, 35]
// Vgprs used for B mma tile 0: [36, 37, 38, 39]
// Vgprs used for B mma tile 1: [40, 41, 42, 43]
// Vgprs used for B mma tile 2: [44, 45, 46, 47]
// Vgprs used for B mma tile 3: [48, 49, 50, 51]
// Vgprs used for MXSA mma tile 0: [17]
// Vgprs used for MXSA mma tile 1: []
// Vgprs used for MXSA mma tile 2: []
// Vgprs used for MXSA mma tile 3: []
// Vgprs used for MXSB mma tile 0: [18]
// Vgprs used for MXSB mma tile 1: []
// Vgprs used for MXSB mma tile 2: []
// Vgprs used for MXSB mma tile 3: []
// Agprs used for D mma tile 0: [0, 1, 2, 3]
// Agprs used for D mma tile 1: [4, 5, 6, 7]
// Agprs used for D mma tile 2: [8, 9, 10, 11]
// Agprs used for D mma tile 3: [12, 13, 14, 15]
// Checking out 19
/* MAINLOOP */
/* REMOVE WHEN IMPLEMNTED: Placeholder for subtile based main loop impl */
label_start:
/* Emit load for A subtile: [0, 0] */
s_add_u32 m0, s[sgprLocalWriteBaseAddrA], 0
buffer_load_dwordx4 v1, s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // grBaseId = 0, i= 0
/* Emit load for A subtile: [1, 0] */
s_add_u32 m0, s[sgprLocalWriteBaseAddrA], 2048
buffer_load_dwordx4 v1, s[sgprSrdA:sgprSrdA+3], s80 offen offset:0 lds // grBaseId = 1, i= 0
/* Emit load for B subtile: [0, 0] */
s_add_u32 m0, s[sgprLocalWriteBaseAddrB], 0
buffer_load_dwordx4 v6, s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // grBaseId = 0, i= 0
/* Emit load for B subtile: [1, 0] */
s_add_u32 m0, s[sgprLocalWriteBaseAddrB], 2048
buffer_load_dwordx4 v6, s[sgprSrdB:sgprSrdB+3], s81 offen offset:0 lds // grBaseId = 1, i= 0
/* Scale GR: MXSA (DTL: BufferLoadB128 -> LDS) */
s_mov_b32 m0, s[sgprLocalWriteBaseAddrMXSA]        // scaleMXSA: M0 = scaleLdsBase
buffer_load_dwordx4 v11, s[sgprSrdMXSA:sgprSrdMXSA+3], 0 offen offset:0 lds // scaleMXSA: DTL b128 load
/* Scale GR: MXSB (DTL: BufferLoadB128 -> LDS) */
s_mov_b32 m0, s[sgprLocalWriteBaseAddrMXSB]        // scaleMXSB: M0 = scaleLdsBase
buffer_load_dwordx4 v14, s[sgprSrdMXSB:sgprSrdMXSB+3], 0 offen offset:0 lds // scaleMXSB: DTL b128 load
s_waitcnt vmcnt(0)                                 // Wait for all subtile GRs to complete
s_barrier
ds_read_b128 v[20:23], v2 offset:0                 // SubtileA[0,0] mfmaId=[0,0]
ds_read_b128 v[24:27], v4 offset:0                 // SubtileA[0,0] mfmaId=[0,1]
ds_read_b128 v[28:31], v2 offset:2048              // SubtileA[1,0] mfmaId=[0,0]
ds_read_b128 v[32:35], v4 offset:2048              // SubtileA[1,0] mfmaId=[0,1]
ds_read_b128 v[36:39], v7 offset:0                 // SubtileB[0,0] mfmaId=[0,0]
ds_read_b128 v[40:43], v9 offset:0                 // SubtileB[0,0] mfmaId=[0,1]
ds_read_b128 v[44:47], v7 offset:2048              // SubtileB[1,0] mfmaId=[0,0]
ds_read_b128 v[48:51], v9 offset:2048              // SubtileB[1,0] mfmaId=[0,1]
ds_read_b32 v17, v12 offset:0                      // scaleMXSA[0]: load 4B from LDS
ds_read_b32 v18, v15 offset:0                      // scaleMXSB[0]: load 4B from LDS
s_waitcnt lgkmcnt(0)                               // Wait for all subtile LRs to complete
v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[20:23], v[36:39], acc[0:3], v17, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[0, 0] += A[0, 0] * B[0, 0] sA = 0, sB = 0
v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[28:31], v[36:39], acc[4:7], v17, v18 op_sel:[1,0] op_sel_hi:[0,0] cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[1, 0] += A[1, 0] * B[0, 0] sA = 1, sB = 0
v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[20:23], v[44:47], acc[8:11], v17, v18 op_sel:[0,1] op_sel_hi:[0,0] cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[0, 1] += A[0, 0] * B[0, 1] sA = 0, sB = 1
v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[28:31], v[44:47], acc[12:15], v17, v18 op_sel:[1,1] op_sel_hi:[0,0] cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[1, 1] += A[1, 0] * B[0, 1] sA = 1, sB = 1
v_mfma_scale_f32_16x16x128_f8f6f4 acc[0:3], v[24:27], v[40:43], acc[0:3], v17, v18 op_sel:[0,0] op_sel_hi:[1,1] cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[0, 0] += A[0, 1] * B[1, 0] sA = 2, sB = 2
v_mfma_scale_f32_16x16x128_f8f6f4 acc[4:7], v[32:35], v[40:43], acc[4:7], v17, v18 op_sel:[1,0] op_sel_hi:[1,1] cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[1, 0] += A[1, 1] * B[1, 0] sA = 3, sB = 2
v_mfma_scale_f32_16x16x128_f8f6f4 acc[8:11], v[24:27], v[48:51], acc[8:11], v17, v18 op_sel:[0,1] op_sel_hi:[1,1] cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[0, 1] += A[0, 1] * B[1, 1] sA = 2, sB = 3
v_mfma_scale_f32_16x16x128_f8f6f4 acc[12:15], v[32:35], v[48:51], acc[12:15], v17, v18 op_sel:[1,1] op_sel_hi:[1,1] cbsz:4 blgp:4 // Emit MMFA code for MMA tiles C[1, 1] += A[1, 1] * B[1, 1] sA = 3, sB = 3
/* Emit code to swap A GR m0 offsets */
s_xor_b32 s[sgprLocalWriteBaseAddrA], s[sgprLocalWriteBaseAddrA], s[sgprLocalWriteSwapA]
/* Emit code to swap B GR m0 offsets */
s_xor_b32 s[sgprLocalWriteBaseAddrB], s[sgprLocalWriteBaseAddrB], s[sgprLocalWriteSwapB]
/* Emit code to swap MXSA GR m0 offsets */
s_xor_b32 s[sgprLocalWriteBaseAddrMXSA], s[sgprLocalWriteBaseAddrMXSA], s[sgprLocalWriteSwapMXSA]
/* Emit code to swap MXSB GR m0 offsets */
s_xor_b32 s[sgprLocalWriteBaseAddrMXSB], s[sgprLocalWriteBaseAddrMXSB], s[sgprLocalWriteSwapMXSB]
/* Emit code to swap A LR vgpr offsets */
v_xor_b32 v2, v2, v3
v_xor_b32 v4, v4, v5
/* Emit code to swap B LR vgpr offsets */
v_xor_b32 v7, v7, v8
v_xor_b32 v9, v9, v10
/* Emit code to swap MXSA LR vgpr offsets */
v_xor_b32 v12, v12, v13
/* Emit code to swap MXSB LR vgpr offsets */
v_xor_b32 v15, v15, v16
s_add_u32 s[sgprSrdA], s[sgprSrdA], 128
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0
s_sub_u32 s[sgprSrdA+2], s[sgprSrdA+2], 128
s_add_u32 s[sgprSrdB], s[sgprSrdB], 128
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0
s_sub_u32 s[sgprSrdB+2], s[sgprSrdB+2], 128
/* Scale SRD update: MXSA += 256 */
s_add_u32 s[sgprSrdMXSA], s[sgprSrdMXSA], 256
s_addc_u32 s[sgprSrdMXSA+1], s[sgprSrdMXSA+1], 0
s_sub_u32 s[sgprSrdMXSA+2], s[sgprSrdMXSA+2], 256
/* Scale SRD update: MXSB += 256 */
s_add_u32 s[sgprSrdMXSB], s[sgprSrdMXSB], 256
s_addc_u32 s[sgprSrdMXSB+1], s[sgprSrdMXSB+1], 0
s_sub_u32 s[sgprSrdMXSB+2], s[sgprSrdMXSB+2], 256
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_u32 s[sgprLoopCounterL], 0
s_cbranch_scc0 label_start
//
/*  ===============================================================  */
/*  =================== Start of post-loop code ===================  */
/*  ===============================================================  */
/* ValuC range: [4-20),  */
.set vgprValuC, 4
label_Summation_End_T8JHFHKM7BO5OHXW:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressMXSA, UNDEF
.set sgprAddressB, UNDEF
.set sgprAddressMXSB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesMXSA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStridesMXSB, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdMXSA, UNDEF
.set sgprSrdB, UNDEF
.set sgprSrdMXSB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsMXSA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprGlobalReadIncsMXSB, UNDEF
/* load store sgprs */
.set sgprSrdC, 28
.set sgprSrdD, 24

/* Mapping of Acc register -> C Vgpr register */
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s38, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s37, s38, s[sgprStrideC1J]            // ScaleC s38 by Stride
s_mul_i32 s36, s38, s[sgprStrideC1J]               // ScaleC s38 by Stride
s_lshl_b64 s[36:37], s[36:37], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s36    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s37   // add hi to SRD
s_mul_hi_u32 s37, s38, s[sgprStrideD1J]            // ScaleD s38 by Stride
s_mul_i32 s36, s38, s[sgprStrideD1J]               // ScaleD s38 by Stride
s_lshl_b64 s[36:37], s[36:37], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s36    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s37   // add hi to SRD

s_mul_hi_u32 s37, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s36, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[36:37], s[36:37], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s36        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s37       // add hi to SRD
s_mul_hi_u32 s37, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s36, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[36:37], s[36:37], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s36        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s37       // add hi to SRD

s_and_b32 s11, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s11, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s37, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s36, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s38, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s38, s38, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s39, s38, s[sgprStrideC1J]            // Free1
s_mul_i32 s38, s38, s[sgprStrideC1J]               // Free1
s_add_u32 s36, s36, s38                            // Free1
s_addc_u32 s37, s37, s39                           // Free1
s_sub_u32 s38, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s38, s38, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s39, s38, s[sgprStrideCK]             // Free2
s_mul_i32 s38, s38, s[sgprStrideCK]                // Free2
s_add_u32 s36, s36, s38                            // Free2
s_addc_u32 s37, s37, s39                           // Free2
s_lshl_b64 s[36:37], s[36:37], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s36        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s37       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v6, 6, v[vgprSerial]                 // 6 = Serial / 64
v_lshrrev_b32 v7, 1, v6                            // 7 = 6 / 2
v_lshlrev_b32 v2, 5, v7                            // wave coordination offset 1
v_and_b32 v7, 15, v[vgprSerial]                    // v7 = v[vgprSerial] % 16
v_add_lshl_u32 v2, v7, v2, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v3, v2, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v5, v2, s[sgprStrideD1J]              //  offset 1
v_and_b32 v7, 1, v6                                // v7 = v6 % 2
v_lshlrev_b32 v7, 5, v7                            // wave coordination offset 0
v_and_b32 v1, 63, v[vgprSerial]                    // v1 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v1                            // 1 = 1 / 16
v_lshlrev_b32 v1, 2, v1                            // thread0 * continuous_output
v_add_lshl_u32 v1, v7, v1, 0                       // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s8, 64, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_u32 v1, s8, v1                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 64, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v2, s8, v2                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_3                         // branch if GSU == 1
label_GW_B0:
label_GW_B0_FD0:
s_and_b32 s36, 63, s[sgprSizeI]                    // s36 = s[sgprSizeI] % 64
s_add_u32 s37, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s37                // wg0 >= nwg0-1 ?
s_cselect_b32 s36, s36, 0                          // set rem
s_cmpk_gt_u32 s36, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_Else            // jump if edges required
s_and_b32 s36, 63, s[sgprSizeJ]                    // s36 = s[sgprSizeJ] % 64
s_add_u32 s37, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s37                // wg1 >= nwg1-1
s_cselect_b32 s36, s36, 0                          // set rem
s_cmpk_gt_u32 s36, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_Then            // jump if edges required
label_GW_B0_FD0_VW4_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=35 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_lshl_u32 v9, v5, v1, 0x2                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=1, coord0Vgpr=1
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc3           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[16:19], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_mul_i32 s12, s[sgprStrideD1J], 64                // scale StrideD *= numRows(16) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[20:23], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[24:27], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_NonEdgeEnd:
label_GW_B0_FD0_VW4_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=28 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v8, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v9, v5, v1, 0x2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v9, v8, v9, s[40:41]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v10, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v8, v10, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v2, vcc, v2, 16                       // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s36, s[sgprStrideC1J], 16                // scale stride
v_add_i32 v3, v3, s36                              // ROWINC- Move cinRowPtr to next row
s_mul_i32 s36, s[sgprStrideD1J], 16                // scale stride
v_add_i32 v5, v5, s36                              // Move coutRowPtrD to next row
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v11, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v11, v8, v11, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v28, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v8, v28, s[40:41]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc3           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[16:19], v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[20:23], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[24:27], v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_Else:
label_GW_B0_FD0_VW1_Else:
label_GW_B0_FD0_VW1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=71 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (1,0,0,0:vw1); (1,0,0,1:vw1); (1,0,0,2:vw1); (1,0,0,3:vw1); (1,1,0,0:vw1); (1,1,0,1:vw1); (1,1,0,2:vw1); (1,1,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v8, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v25, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v25, v8, v25, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v6, vcc, v1, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v26, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v8, v26, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v6, vcc, v1, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v27, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v8, v27, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v6, vcc, v1, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v28, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v8, v28, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v29, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v29, v8, v29, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v6, vcc, v1, 17                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v30, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v8, v30, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
v_add_co_u32 v6, vcc, v1, 18                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v31, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v8, v31, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
v_add_co_u32 v6, vcc, v1, 19                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v32, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v8, v32, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v2, vcc, v2, 16                       // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s36, s[sgprStrideC1J], 16                // scale stride
v_add_i32 v3, v3, s36                              // ROWINC- Move cinRowPtr to next row
s_mul_i32 s36, s[sgprStrideD1J], 16                // scale stride
v_add_i32 v5, v5, s36                              // Move coutRowPtrD to next row
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v33, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v33, v8, v33, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
v_add_co_u32 v6, vcc, v1, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v34, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v8, v34, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,2) */
v_add_co_u32 v6, vcc, v1, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v35, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v35, v8, v35, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,3) */
v_add_co_u32 v6, vcc, v1, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v36, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v8, v36, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v37, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, v8, v37, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,1) */
v_add_co_u32 v6, vcc, v1, 17                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v38, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v8, v38, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,2) */
v_add_co_u32 v6, vcc, v1, 18                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v39, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v8, v39, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,3) */
v_add_co_u32 v6, vcc, v1, 19                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v40, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v8, v40, s[40:41]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+5], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+6], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+7], acc2            // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+8], acc3            // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+10], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+11], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+12], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+13], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+14], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+15], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+16], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+17], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+18], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+19], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+20], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 0, 3)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v9, v25, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v10, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v11, v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v12, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v13, v29, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v14, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v15, v31, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v16, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v17, v33, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v18, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v19, v35, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v20, v36, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v21, v37, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[36:37]                               // addr of next instr
s_add_i32 s38, label_KernelEnd, 4                  // target branch offset
s_add_u32 s36, s36, s38                            // add target branch offset
s_addc_u32 s37, s37, 0                             // add high and carry
s_setpc_b64 s[36:37]                               // branch to label_KernelEnd
label_GSU_3:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1                         // Branch if Beta is not zero

label_GW_B0_1:
label_GW_B0_FD0_1:
s_and_b32 s36, 63, s[sgprSizeI]                    // s36 = s[sgprSizeI] % 64
s_add_u32 s37, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s37                // wg0 >= nwg0-1 ?
s_cselect_b32 s36, s36, 0                          // set rem
s_cmpk_gt_u32 s36, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_Else_1          // jump if edges required
s_and_b32 s36, 63, s[sgprSizeJ]                    // s36 = s[sgprSizeJ] % 64
s_add_u32 s37, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s37                // wg1 >= nwg1-1
s_cselect_b32 s36, s36, 0                          // set rem
s_cmpk_gt_u32 s36, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_Then_1          // jump if edges required
label_GW_B0_FD0_VW4_NonEdge_1:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=35 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_lshl_u32 v9, v5, v1, 0x2                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=1, coord0Vgpr=1
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc3           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[16:19], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_mul_i32 s12, s[sgprStrideD1J], 64                // scale StrideD *= numRows(16) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[20:23], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[24:27], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_NonEdgeEnd_1:
label_GW_B0_FD0_VW4_Then_1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=28 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v8, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v9, v5, v1, 0x2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v9, v8, v9, s[40:41]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v10, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v8, v10, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v2, vcc, v2, 16                       // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s36, s[sgprStrideC1J], 16                // scale stride
v_add_i32 v3, v3, s36                              // ROWINC- Move cinRowPtr to next row
s_mul_i32 s36, s[sgprStrideD1J], 16                // scale stride
v_add_i32 v5, v5, s36                              // Move coutRowPtrD to next row
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v11, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v11, v8, v11, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v28, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v8, v28, s[40:41]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc3           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[16:19], v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[20:23], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[24:27], v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_Else_1:
label_GW_B0_FD0_VW1_Else_1:
label_GW_B0_FD0_VW1_Then_1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=71 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (1,0,0,0:vw1); (1,0,0,1:vw1); (1,0,0,2:vw1); (1,0,0,3:vw1); (1,1,0,0:vw1); (1,1,0,1:vw1); (1,1,0,2:vw1); (1,1,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v8, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v25, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v25, v8, v25, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v6, vcc, v1, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v26, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v8, v26, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v6, vcc, v1, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v27, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v8, v27, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v6, vcc, v1, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v28, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v8, v28, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v29, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v29, v8, v29, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v6, vcc, v1, 17                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v30, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v8, v30, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
v_add_co_u32 v6, vcc, v1, 18                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v31, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v8, v31, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
v_add_co_u32 v6, vcc, v1, 19                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v32, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v8, v32, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v2, vcc, v2, 16                       // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s36, s[sgprStrideC1J], 16                // scale stride
v_add_i32 v3, v3, s36                              // ROWINC- Move cinRowPtr to next row
s_mul_i32 s36, s[sgprStrideD1J], 16                // scale stride
v_add_i32 v5, v5, s36                              // Move coutRowPtrD to next row
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v33, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v33, v8, v33, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
v_add_co_u32 v6, vcc, v1, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v34, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v8, v34, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,2) */
v_add_co_u32 v6, vcc, v1, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v35, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v35, v8, v35, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,3) */
v_add_co_u32 v6, vcc, v1, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v36, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v8, v36, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v37, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, v8, v37, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,1) */
v_add_co_u32 v6, vcc, v1, 17                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v38, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v8, v38, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,2) */
v_add_co_u32 v6, vcc, v1, 18                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v39, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v8, v39, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,3) */
v_add_co_u32 v6, vcc, v1, 19                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v40, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v8, v40, s[40:41]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+5], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+6], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+7], acc2            // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+8], acc3            // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+10], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+11], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+12], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+13], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+14], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+15], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+16], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+17], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+18], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+19], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+20], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 0, 3)] */
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_pk_mul_f32 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+6:vgprValuC+6+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha

/* apply mask, calc new C and issue writes */
buffer_store_dword v9, v25, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v10, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v11, v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v12, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v13, v29, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v14, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v15, v31, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v16, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v17, v33, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v18, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v19, v35, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v20, v36, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v21, v37, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1:
label_GW_B1_FD0:
s_and_b32 s36, 63, s[sgprSizeI]                    // s36 = s[sgprSizeI] % 64
s_add_u32 s37, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s37                // wg0 >= nwg0-1 ?
s_cselect_b32 s36, s36, 0                          // set rem
s_cmpk_gt_u32 s36, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_Else            // jump if edges required
s_and_b32 s36, 63, s[sgprSizeJ]                    // s36 = s[sgprSizeJ] % 64
s_add_u32 s37, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s37                // wg1 >= nwg1-1
s_cselect_b32 s36, s36, 0                          // set rem
s_cmpk_gt_u32 s36, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_Then            // jump if edges required
label_GW_B1_FD0_VW4_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=17 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v10, v3, v1, 0x2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=1, coord0Vgpr=1
buffer_load_dwordx4 v[28:31], v10, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_dwordx4 v[32:35], v10, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s12, s[sgprStrideC1J], 64                // scale StrideC *= numRows(16) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[36:39], v10, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_dwordx4 v[40:43], v10, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
v_add_lshl_u32 v9, v5, v1, 0x2                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=1, coord0Vgpr=1
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc3           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 4 - 1 (beta) vscnt(0) (interleaved)
v_fmac_f32 v[vgprValuC+8], v28, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+9], v29, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+10], v30, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+11], v31, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[12:15], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(2) = 4 - 2 (beta) vscnt(1) (interleaved)
v_fmac_f32 v[vgprValuC+12], v32, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+13], v33, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+14], v34, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+15], v35, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[16:19], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(1) = 4 - 3 (beta) vscnt(2) (interleaved)
v_fmac_f32 v[vgprValuC+16], v36, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+17], v37, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+18], v38, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+19], v39, s[sgprBeta]       // finalSum = sum*alpha + C*beta
s_mul_i32 s12, s[sgprStrideD1J], 64                // scale StrideD *= numRows(16) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[20:23], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(0) = 4 - 4 (beta) vscnt(3) (interleaved)
v_fmac_f32 v[vgprValuC+20], v40, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+21], v41, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+22], v42, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+23], v43, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[24:27], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_NonEdgeEnd:
label_GW_B1_FD0_VW4_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=15 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (1,0,0,0:vw4); (1,1,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v8, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v9, v3, v1, 0x2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v9, v8, v9, s[40:41]                 // LDC clip if OOB. offset
buffer_load_dwordx4 v[28:31], v9, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v9, v5, v1, 0x2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v9, v8, v9, s[40:41]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v10, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v8, v10, s[40:41]               // LDC clip if OOB. offset
buffer_load_dwordx4 v[32:35], v10, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v10, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v8, v10, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v2, vcc, v2, 16                       // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s36, s[sgprStrideC1J], 16                // scale stride
v_add_i32 v3, v3, s36                              // ROWINC- Move cinRowPtr to next row
s_mul_i32 s36, s[sgprStrideD1J], 16                // scale stride
v_add_i32 v5, v5, s36                              // Move coutRowPtrD to next row
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v11, v3, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v11, v8, v11, s[40:41]               // LDC clip if OOB. offset
buffer_load_dwordx4 v[36:39], v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v11, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v11, v8, v11, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v44, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v8, v44, s[40:41]               // LDC clip if OOB. offset
buffer_load_dwordx4 v[40:43], v44, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v44, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v8, v44, s[40:41]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc3           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+8], v28, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+9], v29, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+10], v30, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+11], v31, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[12:15], v9, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+12], v32, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+13], v33, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+14], v34, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+15], v35, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[16:19], v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+16], v36, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+17], v37, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+18], v38, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+19], v39, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[20:23], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+20], v40, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+21], v41, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+22], v42, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+23], v43, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[24:27], v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_Else:
label_GW_B1_FD0_VW1_Else:
label_GW_B1_FD0_VW1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=47 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (1,0,0,0:vw1); (1,0,0,1:vw1); (1,0,0,2:vw1); (1,0,0,3:vw1); (1,1,0,0:vw1); (1,1,0,1:vw1); (1,1,0,2:vw1); (1,1,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v8, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v26, v3, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v8, v26, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v25, v26, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v26, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v8, v26, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v6, vcc, v1, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v28, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v8, v28, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v27, v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v28, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v8, v28, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v6, vcc, v1, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v30, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v8, v30, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v29, v30, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v30, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v8, v30, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v6, vcc, v1, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v32, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v8, v32, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v31, v32, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v32, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v8, v32, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v34, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v8, v34, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v33, v34, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v34, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v8, v34, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v6, vcc, v1, 17                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v36, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v8, v36, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v35, v36, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v36, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v8, v36, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
v_add_co_u32 v6, vcc, v1, 18                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v38, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v8, v38, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v37, v38, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v38, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v8, v38, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
v_add_co_u32 v6, vcc, v1, 19                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v40, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v8, v40, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v39, v40, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v40, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v8, v40, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v2, vcc, v2, 16                       // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s36, s[sgprStrideC1J], 16                // scale stride
v_add_i32 v3, v3, s36                              // ROWINC- Move cinRowPtr to next row
s_mul_i32 s36, s[sgprStrideD1J], 16                // scale stride
v_add_i32 v5, v5, s36                              // Move coutRowPtrD to next row
v_cmp_lt_u32 s[36:37], v1, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v42, v3, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v8, v42, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v41, v42, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v42, v5, v1, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v8, v42, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
v_add_co_u32 v6, vcc, v1, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v44, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v8, v44, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v43, v44, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v44, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v8, v44, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,2) */
v_add_co_u32 v6, vcc, v1, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v46, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, v8, v46, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v45, v46, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v46, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, v8, v46, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,3) */
v_add_co_u32 v6, vcc, v1, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v48, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, v8, v48, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v47, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v48, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, v8, v48, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v6, vcc, v1, 16                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v50, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v8, v50, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v49, v50, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v50, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v8, v50, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,1) */
v_add_co_u32 v6, vcc, v1, 17                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v52, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v52, v8, v52, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v51, v52, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v52, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v52, v8, v52, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,2) */
v_add_co_u32 v6, vcc, v1, 18                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v54, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v54, v8, v54, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v53, v54, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v54, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v54, v8, v54, s[40:41]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,3) */
v_add_co_u32 v6, vcc, v1, 19                       // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[36:37], v6, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[40:41], v2, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[40:41], s[36:37], s[40:41]             // in0 && in1
v_add_lshl_u32 v56, v3, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v56, v8, v56, s[40:41]               // LDC clip if OOB. offset
buffer_load_dword v55, v56, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v56, v5, v6, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v56, v8, v56, s[40:41]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+5], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+6], acc1            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+7], acc2            // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+8], acc3            // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+10], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+11], acc6           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+12], acc7           // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+13], acc8           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+14], acc9           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+15], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+16], acc11          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+17], acc12          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+18], acc13          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+19], acc14          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+20], acc15          // copy acc to vreg[15]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 0, 3)] */
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_pk_mul_f32 v[vgprValuC+6:vgprValuC+6+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+6:vgprValuC+6+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+5], v25, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v9, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+6], v27, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v10, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+7], v29, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v11, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+8], v31, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v12, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+9], v33, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v13, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+10], v35, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v14, v36, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+11], v37, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v15, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+12], v39, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v16, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+13], v41, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v17, v42, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+14], v43, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v18, v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+15], v45, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v19, v46, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+16], v47, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v20, v48, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+17], v49, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v21, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+18], v51, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v22, v52, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+19], v53, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v23, v54, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+20], v55, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v24, v56, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
