
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x128x32_MI16x16x1_CMS_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA950_IU1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG32_8_1
.globl Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x128x32_MI16x16x1_CMS_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA950_IU1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG32_8_1
.p2align 8
.type Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x128x32_MI16x16x1_CMS_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA950_IU1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG32_8_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x128x32_MI16x16x1_CMS_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA950_IU1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG32_8_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 256 // accvgpr offset
  .amdhsa_next_free_vgpr 320 // vgprs
  .amdhsa_next_free_sgpr 72 // sgprs
  .amdhsa_group_segment_fixed_size 99328 // lds bytes
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
/* Num VGPR   =255 */
/* Num AccVGPR=64 */
/* Num SGPR   =72 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 4 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=4 */
/* VectorWidthB=4 */
/* GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=4 */
/* DirectToLdsA=True */
/* DirectToLdsB=True */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.version:
  - 1
amdhsa.kernels:
  - .name: Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x128x32_MI16x16x1_CMS_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA950_IU1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG32_8_1
    .symbol: 'Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x128x32_MI16x16x1_CMS_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA950_IU1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG32_8_1.kd'
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
        .value_type:      f32
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      f32
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
    .group_segment_fixed_size:   99328
    .kernarg_segment_align:      8
    .kernarg_segment_size:       104
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 72
    .sgpr_spill_count:           0
    .vgpr_count:                 255
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x128x32_MI16x16x1_CMS_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA950_IU1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WG32_8_1:
label_ASM_Start:  /// Main body of the asm kernel
.macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\vgprDstIdx+0], \dividend, \magicA
    v_add_u32 v[\vgprDstIdx+0], v[\vgprDstIdx+0], v[\vgprDstIdx+1]
    v_lshrrev_b32 v[\vgprDstIdx+0], \magicShift, v[\vgprDstIdx+0]
.endm

/******************************************/
/* VGPR Assignments for MX                */
/******************************************/
.set vgprMXSBase, 0

/******************************************/
/* VGPR Macro Assignments for MX          */
/******************************************/

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-0), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 10
.set vgprGlobalReadOffsetA, 0
.set vgprGlobalReadOffsetB, 4
.set vgprLocalReadAddrA, 8
.set vgprLocalReadAddrB, 9
.set vgprSerial, 108

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+32
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_T0_I0, 76
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_T0_I0, 92
.set IdentityMatrix, 74

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
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprNumWorkGroups0, 14
.set sgprNumWorkGroups1, 15
.set sgprSizesFree, 24
.set sgprSizesSum, 27
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprLocalWriteAddrA, 46
.set sgprLocalWriteAddrB, 47
.set sgprGSU, 48

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

.set MT0, 128
.set MT1, 128
.set DepthU, 32
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 4
.set SrdShiftLeftB, 4
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0xfffff000

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

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req, vgprOffsetL:req, vgprOffset0I:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideA0I], v[\vgprOffset0I] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0], 0x4, v[\vgprAddr+0]      // add prepad for pointer shift
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0], 0x4, v[\vgprAddr+0]      // add prepad for pointer shift
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_dword s20, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_dword s22, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_dword s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_dword s23, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s21, s20, 0x1e                          // Get arg type
s_and_b32 s20, 0x3fffffff, s20                     // Get nums of gemm
s_cmp_eq_u32 s21, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s21, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
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
s_and_b32 s20, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s21, s2, 0x1e                           // Get arg type
s_mov_b32 s22, s3                                  // Preload internal args
s_cmp_eq_u32 s21, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance2
s_cmp_eq_u32 s21, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance2:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dword s31, s[sgprKernArgAddress:sgprKernArgAddress+1], 28 // 28
s_load_dwordx8 s[32:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 32 // 32
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_mov_b64 s[24:25], s[6:7]                         // move preload data to correct sgpr
s_mov_b64 s[26:27], s[8:9]                         // move preload data to correct sgpr
s_mov_b64 s[28:29], s[10:11]                       // move preload data to correct sgpr
s_mov_b32 s30, s12                                 // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
s_mov_b32 s23, s5                                  // Load num of WGs
label_common_kernel_entry:  /// for both preload/non-preload common code
s_mov_b32 s[sgprWorkGroup0+0], s13                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+1], s14                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+2], s15                 // restore workgroup id
s_and_b32 s[sgprStaggerU], s22, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s22, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s21
s_mov_b32 m0, 0x18400                              // LDS clamp at 99328 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* remap workgroup to XCCs */
s_lshr_b32 s54, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s54, s54                             // Get log(WGMXCC)
s_lshr_b32 s55, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s54, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s51, s23, s54
s_lshl_b32 s51, s51, s54
s_cmp_ge_u32 s[sgprWorkGroup0], s51
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s55, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s51, s[sgprWorkGroup0], s54
s_bfm_b32 s52, s54, 0
s_and_b32 s52, s[sgprWorkGroup0], s52
s_lshr_b32 s53, s23, s54
s_mul_i32 s52, s52, s53
s_add_u32 s[sgprWorkGroup0], s51, s52
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[16:17], s55                        // s51 = s[sgprWorkGroup0] / s55
v_rcp_f64 v[16:17], v[16:17]                       // s51 = s[sgprWorkGroup0] / s55
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup0]          // s51 = s[sgprWorkGroup0] / s55
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s51 = s[sgprWorkGroup0] / s55
v_cvt_u32_f64 v16, v[16:17]                        // s51 = s[sgprWorkGroup0] / s55
v_mul_lo_u32 v17, v16, s55                         // s51 = s[sgprWorkGroup0] / s55
v_sub_u32 v18, s[sgprWorkGroup0], v17              // s51 = s[sgprWorkGroup0] / s55
v_cmpx_ge_u32 exec, v18, s55                       // s51 = s[sgprWorkGroup0] / s55
v_add_u32 v16, v16, 1                              // s51 = s[sgprWorkGroup0] / s55
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v17, v16, s55                         // s51 = s[sgprWorkGroup0] / s55
v_sub_u32 v18, s[sgprWorkGroup0], v17              // s51 = s[sgprWorkGroup0] / s55
v_readfirstlane_b32 s51, v16                       // quotient
v_readfirstlane_b32 s52, v18                       // remainder
s_mul_i32 s51, s51, s55
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s52, s52, s54
/* temp0 = temp0 + temp1 */
s_add_u32 s51, s51, s52
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[16:17], s55                        // s52 = s23 / s55
v_rcp_f64 v[16:17], v[16:17]                       // s52 = s23 / s55
v_cvt_f64_u32 v[18:19], s23                        // s52 = s23 / s55
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s52 = s23 / s55
v_cvt_u32_f64 v16, v[16:17]                        // s52 = s23 / s55
v_mul_lo_u32 v17, v16, s55                         // s52 = s23 / s55
v_sub_u32 v18, s23, v17                            // s52 = s23 / s55
v_cmpx_ge_u32 exec, v18, s55                       // s52 = s23 / s55
v_add_u32 v16, v16, 1                              // s52 = s23 / s55
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s52, v16                       // quotient
s_mul_i32 s52, s52, s55
s_sub_u32 s53, s23, s52
s_cmp_gt_u32 s[sgprWorkGroup0], s52
s_cselect_b32 s52, s53, s55
s_lshr_b32 s52, s52, s54
s_bfm_b32 s53, s54, 0
s_and_b32 s53, s[sgprWorkGroup0], s53
s_mul_i32 s52, s52, s53
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s51, s52
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [10...84) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...64) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v11, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v10, 15, v11                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v10, 5, v10                          // 1. N offset: nOffset = nIdx * nStride(32)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v10, 2, v10                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v11, 4, v11                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v10, v11, 2, v10                    // 5. K offset: lrKOffset = kIdx * mStride(4); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v14, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v14, 1, v14                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v10, v14, 11, v10                   // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v12, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v11, 15, v12                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v11, 5, v11                          // 1. N offset: nOffset = nIdx * nStride(32)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v11, 2, v11                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v12, 4, v12                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v11, v12, 2, v11                    // 5. K offset: lrKOffset = kIdx * mStride(4); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v13, 7, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v13, 1, v13                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v11, v13, 11, v11                   // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v12, 6, v[vgprSerial]                // 12 = Serial / 64
v_lshrrev_b32 v12, 2, v12                          // LSU offset: Get LSU wave_id
s_mov_b32 s16, 32                                  // LSU offset: stride = lsuStride(32) when umlds==True
v_mul_lo_u32 v12, s16, v12                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v12, v10          // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 2, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v13, 10, v[vgprLocalReadAddrA]       // Final Offset: padding 32 per block 1024
v_lshl_add_u32 v[vgprLocalReadAddrA], v13, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 1024

/* local read addresses: final offsets b */
v_lshrrev_b32 v10, 6, v[vgprSerial]                // 10 = Serial / 64
v_lshrrev_b32 v10, 2, v10                          // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(32) when umlds==True (dup assign opt.)
v_mul_lo_u32 v10, s16, v10                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v10, v11          // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 2, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v12, 10, v[vgprLocalReadAddrB]       // Final Offset: padding 32 per block 1024
v_lshl_add_u32 v[vgprLocalReadAddrB], v12, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 1024

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x4200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v11 = A-unroll = serial%LVCA */
v_lshrrev_b32 v10, 3, v[vgprSerial]                // 10 = Serial / 8
v_and_b32 v11, 7, v[vgprSerial]                    // 11 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v11, 2, v11                          // v11 = v11 * 4
v_mov_b32 v14, v11                                 // copy for GlobalSplitU
/* LVCB = 8 */
/* v13 = B-unroll = serial%LVCB */
v_lshrrev_b32 v12, 3, v[vgprSerial]                // 12 = Serial / 8
v_and_b32 v13, 7, v[vgprSerial]                    // 13 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v13, 2, v13                          // v13 = v13 * 4
v_mov_b32 v15, v13                                 // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v14 */
/* lwaUnrollAssignmentB = v15 */

/* local write addresses: first offset a */
v_mul_u32_u24 v16, 0x20, v10                       // lwAL**(DepthU_Compute + PAD)
v_add_u32 v16, v14, v16                            // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v16, 2, v16                          //  (multiple bpe)
v_lshrrev_b32 v18, 10, v16                         // padding 32 per block 1024
v_lshl_add_u32 v16, v18, 5, v16                    // padding 32 per block 1024
v_lshrrev_b32 v17, 6, v[vgprSerial]                // Compute waveID
s_nop 0                                            // 1 wait states required before reading vgpr by lane
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v17    // Copy lds write address VGPR to SGPR
s_mul_i32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], 1056

/* local write addresses: first offset b */
v_mul_u32_u24 v16, 0x20, v12                       // lwBL**(DepthU_Compute + PAD)
v_add_u32 v16, v15, v16                            // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v16, 2, v16                          //  (multiple bpe)
v_lshrrev_b32 v18, 10, v16                         // padding 32 per block 1024
v_lshl_add_u32 v16, v18, 5, v16                    // padding 32 per block 1024
v_add_co_u32 v16, vcc, 0x4200, v16                 // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=16896
v_lshrrev_b32 v17, 6, v[vgprSerial]                // Compute waveID
s_nop 0                                            // 1 wait states required before reading vgpr by lane
v_readfirstlane_b32 s[sgprLocalWriteAddrB], v17    // Copy lds write address VGPR to SGPR
s_mul_i32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], 1056
s_add_u32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], 16896
v_mov_b32 v18, MT0                                 // set MT0 into sgpr
v_mov_b32 v17, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v16, v18                             // v16 = ceil(v17 / v18)
v_rcp_iflag_f32 v16, v16                           // v16 = ceil(v17 / v18)
v_cvt_f32_u32 v19, v17                             // v16 = ceil(v17 / v18)
v_mul_f32 v16, v16, v19                            // v16 = ceil(v17 / v18)
v_cvt_u32_f32 v16, v16                             // v16 = ceil(v17 / v18)
v_mul_u32_u24 v19, v16, v18                        // v16 = ceil(v17 / v18)
v_sub_u32 v19, v17, v19                            // v16 = ceil(v17 / v18)
v_cmp_ne_u32 vcc, v19, 0                           // v16 = ceil(v17 / v18)
v_addc_co_u32 v16, vcc, v16, 0, vcc                // ceil
v_mov_b32 v18, MT1                                 // set MT1 into sgpr
v_mov_b32 v17, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v16     // set back to numWorkGroup0
v_cvt_f32_u32 v16, v18                             // v16 = ceil(v17 / v18)
v_rcp_iflag_f32 v16, v16                           // v16 = ceil(v17 / v18)
v_cvt_f32_u32 v19, v17                             // v16 = ceil(v17 / v18)
v_mul_f32 v16, v16, v19                            // v16 = ceil(v17 / v18)
v_cvt_u32_f32 v16, v16                             // v16 = ceil(v17 / v18)
v_mul_u32_u24 v19, v16, v18                        // v16 = ceil(v17 / v18)
v_sub_u32 v19, v17, v19                            // v16 = ceil(v17 / v18)
v_cmp_ne_u32 vcc, v19, 0                           // v16 = ceil(v17 / v18)
v_addc_co_u32 v16, vcc, v16, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v16     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 44/0 bytes of kern args

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v16, s16                             // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v16, v16                           // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v17, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v16, v16, v17                            // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v16, v16                             // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v17, v16, s16                        // s16 = s[sgprWorkGroup0] / s16
v_sub_u32 v17, s[sgprWorkGroup0], v17              // s16 = s[sgprWorkGroup0] / s16
v_cmpx_eq_u32 exec, v17, s16                       // s16 = s[sgprWorkGroup0] / s16
v_add_u32 v16, 1, v16                              // s16 = s[sgprWorkGroup0] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v17, s16                       // overflow happened in remainder
v_sub_u32 v16, v16, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s16, v16                       // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v16, s[sgprNumWorkGroups0]           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v16, v16                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v17, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v16, v16, v17                            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v16, v16                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v17, v16, s[sgprNumWorkGroups0]      // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v17, s[sgprWorkGroup0], v17              // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v17, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v16, 1, v16                              // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v17, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v16, v16, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s16, v16                       // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 88                                  // KernArgAddressOffset
s_mul_i32 s56, s20, 4
s_mov_b64 s[50:51], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 196
s_mov_b32 s56, 0
s_mov_b64 s[50:51], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s57, 0
s_load_dwordx4 s[24:27], s[50:51], s56
s_cmpk_eq_u32 s20, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s54, s24, 7                             // s54 = s24 / 128
s_and_b32 s52, 127, s24                            // s52 = s24 % 128
s_addc_u32 s54, s54, 0
s_lshr_b32 s55, s25, 7                             // s55 = s25 / 128
s_and_b32 s52, 127, s25                            // s52 = s25 % 128
s_addc_u32 s55, s55, 0
s_mul_i32 s54, s54, s55
s_mul_i32 s54, s54, s26
s_and_b32 s55, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s54, s54, s55
s_add_u32 s57, s57, s54
s_cmp_lt_u32 s[sgprWorkGroup0], s57
s_cbranch_scc1 label_FOUND
s_add_u32 s56, s56, s15
s_load_dwordx4 s[24:27], s[50:51], s56
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s20
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s54, s24, 7                             // s54 = s24 / 128
s_and_b32 s52, 127, s24                            // s52 = s24 % 128
s_addc_u32 s54, s54, 0
s_lshr_b32 s55, s25, 7                             // s55 = s25 / 128
s_and_b32 s52, 127, s25                            // s52 = s25 % 128
s_addc_u32 s55, s55, 0
s_mul_i32 s54, s54, s55
s_mul_i32 s54, s54, s26
s_and_b32 s50, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s54, s54, s50
s_add_u32 s57, s57, s54

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s51, s14, 1
s_sub_u32 s50, s57, s54
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s50
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s51, s51, 88                             // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s51
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s51, s51, 196
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s51
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dword s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
// Read Beta
s_load_dword s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
label_LoadExternalStructEnd:
/* init: add vgpr [10...84) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...64) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v11, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v10, 15, v11                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v10, 5, v10                          // 1. N offset: nOffset = nIdx * nStride(32)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v10, 2, v10                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v11, 4, v11                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v10, v11, 2, v10                    // 5. K offset: lrKOffset = kIdx * mStride(4); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v14, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v14, 1, v14                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v10, v14, 11, v10                   // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v12, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v11, 15, v12                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v11, 5, v11                          // 1. N offset: nOffset = nIdx * nStride(32)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v11, 2, v11                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v12, 4, v12                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v11, v12, 2, v11                    // 5. K offset: lrKOffset = kIdx * mStride(4); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v13, 7, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v13, 1, v13                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v11, v13, 11, v11                   // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v12, 6, v[vgprSerial]                // 12 = Serial / 64
v_lshrrev_b32 v12, 2, v12                          // LSU offset: Get LSU wave_id
s_mov_b32 s16, 32                                  // LSU offset: stride = lsuStride(32) when umlds==True
v_mul_lo_u32 v12, s16, v12                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v12, v10          // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 2, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v13, 10, v[vgprLocalReadAddrA]       // Final Offset: padding 32 per block 1024
v_lshl_add_u32 v[vgprLocalReadAddrA], v13, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 1024

/* local read addresses: final offsets b */
v_lshrrev_b32 v10, 6, v[vgprSerial]                // 10 = Serial / 64
v_lshrrev_b32 v10, 2, v10                          // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(32) when umlds==True (dup assign opt.)
v_mul_lo_u32 v10, s16, v10                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v10, v11          // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 2, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v12, 10, v[vgprLocalReadAddrB]       // Final Offset: padding 32 per block 1024
v_lshl_add_u32 v[vgprLocalReadAddrB], v12, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 1024

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x4200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v11 = A-unroll = serial%LVCA */
v_lshrrev_b32 v10, 3, v[vgprSerial]                // 10 = Serial / 8
v_and_b32 v11, 7, v[vgprSerial]                    // 11 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v11, 2, v11                          // v11 = v11 * 4
v_mov_b32 v14, v11                                 // copy for GlobalSplitU
/* LVCB = 8 */
/* v13 = B-unroll = serial%LVCB */
v_lshrrev_b32 v12, 3, v[vgprSerial]                // 12 = Serial / 8
v_and_b32 v13, 7, v[vgprSerial]                    // 13 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v13, 2, v13                          // v13 = v13 * 4
v_mov_b32 v15, v13                                 // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v14 */
/* lwaUnrollAssignmentB = v15 */

/* local write addresses: first offset a */
v_mul_u32_u24 v16, 0x20, v10                       // lwAL**(DepthU_Compute + PAD)
v_add_u32 v16, v14, v16                            // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v16, 2, v16                          //  (multiple bpe)
v_lshrrev_b32 v18, 10, v16                         // padding 32 per block 1024
v_lshl_add_u32 v16, v18, 5, v16                    // padding 32 per block 1024
v_lshrrev_b32 v17, 6, v[vgprSerial]                // Compute waveID
s_nop 0                                            // 1 wait states required before reading vgpr by lane
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v17    // Copy lds write address VGPR to SGPR
s_mul_i32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], 1056

/* local write addresses: first offset b */
v_mul_u32_u24 v16, 0x20, v12                       // lwBL**(DepthU_Compute + PAD)
v_add_u32 v16, v15, v16                            // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v16, 2, v16                          //  (multiple bpe)
v_lshrrev_b32 v18, 10, v16                         // padding 32 per block 1024
v_lshl_add_u32 v16, v18, 5, v16                    // padding 32 per block 1024
v_add_co_u32 v16, vcc, 0x4200, v16                 // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=16896
v_lshrrev_b32 v17, 6, v[vgprSerial]                // Compute waveID
s_nop 0                                            // 1 wait states required before reading vgpr by lane
v_readfirstlane_b32 s[sgprLocalWriteAddrB], v17    // Copy lds write address VGPR to SGPR
s_mul_i32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], 1056
s_add_u32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], 16896
v_mov_b32 v18, MT0                                 // set MT0 into sgpr
v_mov_b32 v17, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v16, v18                             // v16 = ceil(v17 / v18)
v_rcp_iflag_f32 v16, v16                           // v16 = ceil(v17 / v18)
v_cvt_f32_u32 v19, v17                             // v16 = ceil(v17 / v18)
v_mul_f32 v16, v16, v19                            // v16 = ceil(v17 / v18)
v_cvt_u32_f32 v16, v16                             // v16 = ceil(v17 / v18)
v_mul_u32_u24 v19, v16, v18                        // v16 = ceil(v17 / v18)
v_sub_u32 v19, v17, v19                            // v16 = ceil(v17 / v18)
v_cmp_ne_u32 vcc, v19, 0                           // v16 = ceil(v17 / v18)
v_addc_co_u32 v16, vcc, v16, 0, vcc                // ceil
v_mov_b32 v18, MT1                                 // set MT1 into sgpr
v_mov_b32 v17, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v16     // set back to numWorkGroup0
v_cvt_f32_u32 v16, v18                             // v16 = ceil(v17 / v18)
v_rcp_iflag_f32 v16, v16                           // v16 = ceil(v17 / v18)
v_cvt_f32_u32 v19, v17                             // v16 = ceil(v17 / v18)
v_mul_f32 v16, v16, v19                            // v16 = ceil(v17 / v18)
v_cvt_u32_f32 v16, v16                             // v16 = ceil(v17 / v18)
v_mul_u32_u24 v19, v16, v18                        // v16 = ceil(v17 / v18)
v_sub_u32 v19, v17, v19                            // v16 = ceil(v17 / v18)
v_cmp_ne_u32 vcc, v19, 0                           // v16 = ceil(v17 / v18)
v_addc_co_u32 v16, vcc, v16, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v16     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 44/0 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v16, s16                             // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v16, v16                           // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v17, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v16, v16, v17                            // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v16, v16                             // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v17, v16, s16                        // s16 = s[sgprWorkGroup0] / s16
v_sub_u32 v17, s[sgprWorkGroup0], v17              // s16 = s[sgprWorkGroup0] / s16
v_cmpx_eq_u32 exec, v17, s16                       // s16 = s[sgprWorkGroup0] / s16
v_add_u32 v16, 1, v16                              // s16 = s[sgprWorkGroup0] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v17, s16                       // overflow happened in remainder
v_sub_u32 v16, v16, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s16, v16                       // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v16, s[sgprNumWorkGroups0]           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v16, v16                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v17, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v16, v16, v17                            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v16, v16                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v17, v16, s[sgprNumWorkGroups0]      // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v17, s[sgprWorkGroup0], v17              // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v17, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v16, 1, v16                              // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v17, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v16, v16, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s16, v16                       // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprSrdA, 52
.set sgprSrdB, 56
.set sgprShadowLimitA, 50
.set sgprShadowLimitB, 60
.set sgprStaggerUIter, 49
.set sgprWrapUA, 62
.set sgprWrapUB, 64
.set sgprGlobalReadIncsA, 66
.set sgprGlobalReadIncsB, 67
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift
label_Skip_Address_Prepad_For_Pointer_Array:  /// Skip pre-padding of address for pointer array case

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:
/* Create a negative identity matrix used by TF32 MFMA emulation. */
v_and_b32 v17, 3, v[vgprSerial]                    // lane % 4
v_mov_b64 v[74:75], 0
v_mov_b32 v16, 0xbf80
v_cmp_eq_u32 vcc, 0, v17                           // Lane %4 == 0 ?
s_nop 1
v_cndmask_b32 v74, v74, v16, vcc
v_cmp_eq_u32 vcc, 2, v17                           // Lane %4 == 2 ?
s_nop 1
v_cndmask_b32 v75, v75, v16, vcc
v_mov_b32 v16, 0xbf800000
v_cmp_eq_u32 vcc, 1, v17                           // Lane %4 == 1 ?
s_nop 1
v_cndmask_b32 v74, v74, v16, vcc
v_cmp_eq_u32 vcc, 3, v17                           // Lane %4 == 3 ?
s_nop 1
v_cndmask_b32 v75, v75, v16, vcc

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s16, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v16, s16                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_rcp_iflag_f32 v16, v16                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_f32_u32 v17, s[sgprWorkGroup1]               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_f32 v16, v16, v17                            // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_u32_f32 v16, v16                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_u32_u24 v17, v16, s16                        // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_sub_u32 v17, s[sgprWorkGroup1], v17              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cmpx_eq_u32 exec, v17, s16                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_add_u32 v16, 1, v16                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mov_b32 v17, 0                                   // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s16
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v17, s16                       // overflow happened in remainder
v_sub_u32 v16, v16, 1                              // quotient - 1
v_mul_u32_u24 v17, v16, s16                        // re-calculate remainder
v_sub_u32 v17, s[sgprWorkGroup1], v17              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v16         // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v17          // remainder
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
v_cvt_f32_u32 v16, s[sgprNumWorkGroups1]           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v16, v16                           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_f32_u32 v17, s[sgprWorkGroup1]               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v16, v16, v17                            // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_u32_f32 v16, v16                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v17, v16, s[sgprNumWorkGroups1]      // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_sub_u32 v17, s[sgprWorkGroup1], v17              // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmpx_eq_u32 exec, v17, s[sgprNumWorkGroups1]     // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_u32 v16, 1, v16                              // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v17, 0                                   // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v17, s[sgprNumWorkGroups1]     // overflow happened in remainder
v_sub_u32 v16, v16, 1                              // quotient - 1
v_mul_u32_u24 v17, v16, s[sgprNumWorkGroups1]      // re-calculate remainder
v_sub_u32 v17, s[sgprWorkGroup1], v17              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprGSUSumIdx], v16          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v17         // remainder
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
s_mov_b32 s16, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s16, s16                            // Restore WGM
s_cmp_gt_i32 s16, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s16, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s16, s16                                 // abs(WGM)
v_cvt_f64_u32 v[16:17], s16                        // s17 = s[sgprWorkGroup0] / s16
v_rcp_f64 v[16:17], v[16:17]                       // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup0]          // s17 = s[sgprWorkGroup0] / s16
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s17 = s[sgprWorkGroup0] / s16
v_cvt_u32_f64 v16, v[16:17]                        // s17 = s[sgprWorkGroup0] / s16
v_mul_lo_u32 v17, v16, s16                         // s17 = s[sgprWorkGroup0] / s16
v_sub_u32 v18, s[sgprWorkGroup0], v17              // s17 = s[sgprWorkGroup0] / s16
v_cmpx_ge_u32 exec, v18, s16                       // s17 = s[sgprWorkGroup0] / s16
v_add_u32 v16, v16, 1                              // s17 = s[sgprWorkGroup0] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s17, v16                       // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup0], s20              // WorkGroup0=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s20, s20, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[16:17], s16                        // s18 = s[sgprNumWorkGroups0] / s16
v_rcp_f64 v[16:17], v[16:17]                       // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_f64_u32 v[18:19], s[sgprNumWorkGroups0]      // s18 = s[sgprNumWorkGroups0] / s16
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_u32_f64 v16, v[16:17]                        // s18 = s[sgprNumWorkGroups0] / s16
v_mul_lo_u32 v17, v16, s16                         // s18 = s[sgprNumWorkGroups0] / s16
v_sub_u32 v18, s[sgprNumWorkGroups0], v17          // s18 = s[sgprNumWorkGroups0] / s16
v_cmpx_ge_u32 exec, v18, s16                       // s18 = s[sgprNumWorkGroups0] / s16
v_add_u32 v16, v16, 1                              // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s18, v16                       // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups0], s19          // NumWorkGroups0=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[16:17], s18                        // s[sgprWorkGroup1] = s20 / s18
v_rcp_f64 v[16:17], v[16:17]                       // s[sgprWorkGroup1] = s20 / s18
v_cvt_f64_u32 v[18:19], s20                        // s[sgprWorkGroup1] = s20 / s18
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s[sgprWorkGroup1] = s20 / s18
v_cvt_u32_f64 v16, v[16:17]                        // s[sgprWorkGroup1] = s20 / s18
v_mul_lo_u32 v17, v16, s18                         // s[sgprWorkGroup1] = s20 / s18
v_sub_u32 v18, s20, v17                            // s[sgprWorkGroup1] = s20 / s18
v_cmpx_ge_u32 exec, v18, s18                       // s[sgprWorkGroup1] = s20 / s18
v_add_u32 v16, v16, 1                              // s[sgprWorkGroup1] = s20 / s18
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v17, v16, s18                         // s[sgprWorkGroup1] = s20 / s18
v_sub_u32 v18, s20, v17                            // s[sgprWorkGroup1] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup1], v16         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v18         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s16, s16                                 // WGM
v_cvt_f64_u32 v[16:17], s16                        // s17 = s[sgprWorkGroup1] / s16
v_rcp_f64 v[16:17], v[16:17]                       // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup1]          // s17 = s[sgprWorkGroup1] / s16
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s17 = s[sgprWorkGroup1] / s16
v_cvt_u32_f64 v16, v[16:17]                        // s17 = s[sgprWorkGroup1] / s16
v_mul_lo_u32 v17, v16, s16                         // s17 = s[sgprWorkGroup1] / s16
v_sub_u32 v18, s[sgprWorkGroup1], v17              // s17 = s[sgprWorkGroup1] / s16
v_cmpx_ge_u32 exec, v18, s16                       // s17 = s[sgprWorkGroup1] / s16
v_add_u32 v16, v16, 1                              // s17 = s[sgprWorkGroup1] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s17, v16                       // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup1], s20              // WorkGroup1=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s20, s20, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[16:17], s16                        // s18 = s[sgprNumWorkGroups1] / s16
v_rcp_f64 v[16:17], v[16:17]                       // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_f64_u32 v[18:19], s[sgprNumWorkGroups1]      // s18 = s[sgprNumWorkGroups1] / s16
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_u32_f64 v16, v[16:17]                        // s18 = s[sgprNumWorkGroups1] / s16
v_mul_lo_u32 v17, v16, s16                         // s18 = s[sgprNumWorkGroups1] / s16
v_sub_u32 v18, s[sgprNumWorkGroups1], v17          // s18 = s[sgprNumWorkGroups1] / s16
v_cmpx_ge_u32 exec, v18, s16                       // s18 = s[sgprNumWorkGroups1] / s16
v_add_u32 v16, v16, 1                              // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s18, v16                       // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups1], s19          // NumWorkGroups1=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[16:17], s18                        // s[sgprWorkGroup0] = s20 / s18
v_rcp_f64 v[16:17], v[16:17]                       // s[sgprWorkGroup0] = s20 / s18
v_cvt_f64_u32 v[18:19], s20                        // s[sgprWorkGroup0] = s20 / s18
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s[sgprWorkGroup0] = s20 / s18
v_cvt_u32_f64 v16, v[16:17]                        // s[sgprWorkGroup0] = s20 / s18
v_mul_lo_u32 v17, v16, s18                         // s[sgprWorkGroup0] = s20 / s18
v_sub_u32 v18, s20, v17                            // s[sgprWorkGroup0] = s20 / s18
v_cmpx_ge_u32 exec, v18, s18                       // s[sgprWorkGroup0] = s20 / s18
v_add_u32 v16, v16, 1                              // s[sgprWorkGroup0] = s20 / s18
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v17, v16, s18                         // s[sgprWorkGroup0] = s20 / s18
v_sub_u32 v18, s20, v17                            // s[sgprWorkGroup0] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup0], v16         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v18         // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s20, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s17 // wg1 += blockId * WGM
label_WGM:

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v10 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v12 */

/* global read addresses: unroll assignment a */
/* v11 */

/* global read addresses: unroll assignment b */
/* v13 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */
v_mov_b32 v16, v10                                 // groA0I_0
v_add_co_u32 v17, vcc, 32, v16                     // groA0I_1 += LSPA
v_add_co_u32 v18, vcc, 32, v17                     // groA0I_2 += LSPA
v_add_co_u32 v19, vcc, 32, v18                     // groA0I_3 += LSPA

/* global read addresses: tile offsets b */
v_mov_b32 v20, v12                                 // groB1J_0
v_add_co_u32 v21, vcc, 32, v20                     // groB1J_1 += LSPB
v_add_co_u32 v22, vcc, 32, v21                     // groB1J_2 += LSPB
v_add_co_u32 v23, vcc, 32, v22                     // groB1J_3 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v24, v11                                 // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v25, v13                                 // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s17, 32, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 32, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 5 // s[LoopCounterL] = s[sgprSizesSum] / 32
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v26, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v26, v26                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v27, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v26, v26, v27                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v26, v26                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v27, v26, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v27, s[sgprLoopCounterL], v27            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v27, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v26, 1, v26                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v27, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v27, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v26, v26, 1                              // quotient - 1
v_mul_u32_u24 v27, v26, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v27, s[sgprLoopCounterL], v27            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v26       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v27        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 32                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 32                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_add_u32 s18, s18, s16                            // accum GsuOffset term to tilestart
s_addc_u32 s19, s19, s17                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s16, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s17, constStrideAL, s16               // stride x (size-1)
s_mul_i32 s16, constStrideAL, s16                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // sum tensor size
s_sub_u32 s16, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s17, s[sgprStrideA0I], s16            // stride x (size-1)
s_mul_i32 s16, s[sgprStrideA0I], s16               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s18 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s19 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 2 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s16, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_add_u32 s16, s16, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s17, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdA:sgprSrdA+1], s[16:17], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s18, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s19, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s17, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s19   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s17, 32, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 32, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 5 // s[LoopCounterL] = s[sgprSizesSum] / 32
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v26, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v26, v26                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v27, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v26, v26, v27                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v26, v26                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v27, v26, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v27, s[sgprLoopCounterL], v27            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v27, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v26, 1, v26                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v27, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v27, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v26, v26, 1                              // quotient - 1
v_mul_u32_u24 v27, v26, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v27, s[sgprLoopCounterL], v27            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v26       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v27        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 32                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 32                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_B_End:
s_add_u32 s18, s18, s16                            // accum GsuOffset term to tilestart
s_addc_u32 s19, s19, s17                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s16, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s17, constStrideBL, s16               // stride x (size-1)
s_mul_i32 s16, constStrideBL, s16                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17 // sum tensor size
s_sub_u32 s16, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s17, s[sgprStrideB1J], s16            // stride x (size-1)
s_mul_i32 s16, s[sgprStrideB1J], s16               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s18 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s19 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 2 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s16, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_add_u32 s16, s16, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s17, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdB:sgprSrdB+1], s[16:17], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s18, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s19, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s17, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s19   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
// Using GLNC for A
/* NumThreadsCoalescedA = 8, 256 total threads, 1 thread groups */
v_mov_b32 v[vgprGlobalReadOffsetA+0], v[vgprSerial]
v_add_u32 v[vgprGlobalReadOffsetA+1], 256, v[vgprGlobalReadOffsetA+0] //  = vgprSerial + 1 * 256
v_add_u32 v[vgprGlobalReadOffsetA+2], 256, v[vgprGlobalReadOffsetA+1] //  = vgprSerial + 2 * 256
v_add_u32 v[vgprGlobalReadOffsetA+3], 256, v[vgprGlobalReadOffsetA+2] //  = vgprSerial + 3 * 256
v_lshrrev_b32 v30, 3, v[vgprGlobalReadOffsetA+0]   // division
v_and_b32 v29, 0x7, v[vgprGlobalReadOffsetA+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 2, v29
v_mul_lo_u32 v30, s[sgprStridesA], v30
v_add_u32 v[vgprGlobalReadOffsetA+0], v30, v[vgprGlobalReadOffsetA+0] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 2, v[vgprGlobalReadOffsetA+0]
v_add_u32 v[vgprGlobalReadOffsetA+0], 16, v[vgprGlobalReadOffsetA+0] // ptr-shift
v_lshrrev_b32 v30, 3, v[vgprGlobalReadOffsetA+1]   // division
v_and_b32 v29, 0x7, v[vgprGlobalReadOffsetA+1]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+1], 2, v29
v_mul_lo_u32 v30, s[sgprStridesA], v30
v_add_u32 v[vgprGlobalReadOffsetA+1], v30, v[vgprGlobalReadOffsetA+1] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+1], 2, v[vgprGlobalReadOffsetA+1]
v_add_u32 v[vgprGlobalReadOffsetA+1], 16, v[vgprGlobalReadOffsetA+1] // ptr-shift
v_lshrrev_b32 v30, 3, v[vgprGlobalReadOffsetA+2]   // division
v_and_b32 v29, 0x7, v[vgprGlobalReadOffsetA+2]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+2], 2, v29
v_mul_lo_u32 v30, s[sgprStridesA], v30
v_add_u32 v[vgprGlobalReadOffsetA+2], v30, v[vgprGlobalReadOffsetA+2] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+2], 2, v[vgprGlobalReadOffsetA+2]
v_add_u32 v[vgprGlobalReadOffsetA+2], 16, v[vgprGlobalReadOffsetA+2] // ptr-shift
v_lshrrev_b32 v30, 3, v[vgprGlobalReadOffsetA+3]   // division
v_and_b32 v29, 0x7, v[vgprGlobalReadOffsetA+3]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+3], 2, v29
v_mul_lo_u32 v30, s[sgprStridesA], v30
v_add_u32 v[vgprGlobalReadOffsetA+3], v30, v[vgprGlobalReadOffsetA+3] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+3], 2, v[vgprGlobalReadOffsetA+3]
v_add_u32 v[vgprGlobalReadOffsetA+3], 16, v[vgprGlobalReadOffsetA+3] // ptr-shift

/* global read addresses: final offsets b */
// Using GLNC for B
/* NumThreadsCoalescedB = 8, 256 total threads, 1 thread groups */
v_mov_b32 v[vgprGlobalReadOffsetB+0], v[vgprSerial]
v_add_u32 v[vgprGlobalReadOffsetB+1], 256, v[vgprGlobalReadOffsetB+0] //  = vgprSerial + 1 * 256
v_add_u32 v[vgprGlobalReadOffsetB+2], 256, v[vgprGlobalReadOffsetB+1] //  = vgprSerial + 2 * 256
v_add_u32 v[vgprGlobalReadOffsetB+3], 256, v[vgprGlobalReadOffsetB+2] //  = vgprSerial + 3 * 256
v_lshrrev_b32 v10, 3, v[vgprGlobalReadOffsetB+0]   // division
v_and_b32 v14, 0x7, v[vgprGlobalReadOffsetB+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 2, v14
v_mul_lo_u32 v10, s[sgprStridesB], v10
v_add_u32 v[vgprGlobalReadOffsetB+0], v10, v[vgprGlobalReadOffsetB+0] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 2, v[vgprGlobalReadOffsetB+0]
v_add_u32 v[vgprGlobalReadOffsetB+0], 16, v[vgprGlobalReadOffsetB+0] // ptr-shift
v_lshrrev_b32 v10, 3, v[vgprGlobalReadOffsetB+1]   // division
v_and_b32 v14, 0x7, v[vgprGlobalReadOffsetB+1]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 2, v14
v_mul_lo_u32 v10, s[sgprStridesB], v10
v_add_u32 v[vgprGlobalReadOffsetB+1], v10, v[vgprGlobalReadOffsetB+1] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 2, v[vgprGlobalReadOffsetB+1]
v_add_u32 v[vgprGlobalReadOffsetB+1], 16, v[vgprGlobalReadOffsetB+1] // ptr-shift
v_lshrrev_b32 v10, 3, v[vgprGlobalReadOffsetB+2]   // division
v_and_b32 v14, 0x7, v[vgprGlobalReadOffsetB+2]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 2, v14
v_mul_lo_u32 v10, s[sgprStridesB], v10
v_add_u32 v[vgprGlobalReadOffsetB+2], v10, v[vgprGlobalReadOffsetB+2] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 2, v[vgprGlobalReadOffsetB+2]
v_add_u32 v[vgprGlobalReadOffsetB+2], 16, v[vgprGlobalReadOffsetB+2] // ptr-shift
v_lshrrev_b32 v10, 3, v[vgprGlobalReadOffsetB+3]   // division
v_and_b32 v14, 0x7, v[vgprGlobalReadOffsetB+3]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 2, v14
v_mul_lo_u32 v10, s[sgprStridesB], v10
v_add_u32 v[vgprGlobalReadOffsetB+3], v10, v[vgprGlobalReadOffsetB+3] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 2, v[vgprGlobalReadOffsetB+3]
v_add_u32 v[vgprGlobalReadOffsetB+3], 16, v[vgprGlobalReadOffsetB+3] // ptr-shift

/* global read addresses: increments a */
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s17, s17, 128                            // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], 128, s17   // incrA (unrollIdx)

/* global read addresses: increments b */
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s17, s17, 128                            // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], 128, s17   // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 5 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 32
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v10, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v10, v10                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v11, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v10, v10, v11                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v10, v10                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v11, v10, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v11, s[sgprLoopCounterL], v11            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v11, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v10, 1, v10                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v11, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v11, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v10, v10, 1                              // quotient - 1
v_mul_u32_u24 v11, v10, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v11, s[sgprLoopCounterL], v11            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v10       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v11        // remainder
s_add_u32 s16, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s16                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s18, s[sgprStaggerU], 0x1f00
s_lshr_b32 s18, s18, 0x8
s_and_b32 s19, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s16, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s17, s16, s18                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s17           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s16, s16, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s17, s16, 1                              // staggerU mask
s_cmp_ge_u32 s16, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s17, 0          // set Mask
s_cmp_eq_u32 s19, 0x0
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s19, 0x2000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s16, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s19, 0x4000
s_cbranch_scc0 label_StaggerUMapping_3
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s19, 0x6000
s_cbranch_scc0 label_StaggerUMapping_4
s_mul_i32 s17, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s16, s16, s17
s_add_u32 s16, s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s19, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s16 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s18 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

/* localReadInitPointers */

/* prefetch: global -> local */
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
/* before DirectToLds load, ensure prior ds_reads have finished */
s_waitcnt lgkmcnt(0)
s_barrier
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0

/* global read inc A loopL */
s_add_u32 s18, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s18              // Is this wrapIter? (pf)
s_cselect_b32 s16, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s17, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s18, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s18              // Is this wrapIter? (pf)
s_cselect_b32 s16, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s17, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_and_b32 s68, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s68, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmSrdInitiationD
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_GeneralBatchedGemmSrdInitiationD:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationD_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_and_b32 s68, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s68, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmSrdInitiationC
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_GeneralBatchedGemmSrdInitiationC:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationC_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s70, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s69, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_hi_u32 s69, s70, s[sgprStrideC1J]            // ScaleC s70 by Stride
s_mul_i32 s68, s70, s[sgprStrideC1J]               // ScaleC s70 by Stride
s_lshl_b64 s[68:69], s[68:69], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s69       // add hi to SRD
s_and_b32 s69, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_hi_u32 s69, s70, s[sgprStrideD1J]            // ScaleD s70 by Stride
s_mul_i32 s68, s70, s[sgprStrideD1J]               // ScaleD s70 by Stride
s_lshl_b64 s[68:69], s[68:69], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // add hi to SRD

s_and_b32 s69, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s69, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s69, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s68, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[68:69], s[68:69], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s69       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s68, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s68, s68, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s69, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[68:69], s[68:69], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s68        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s69       // Offsetting within the Batch Matrix [Higher half of address]
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s69, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s69, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s69, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s68, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[68:69], s[68:69], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s68, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s68, s68, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s69, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[68:69], s[68:69], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // Offsetting within the Batch Matrix [Higher half of address]
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_and_b32 s68, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s68, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s69, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s68, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s70, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s70, s70, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s71, s70, s[sgprStrideC1J]            // Free1
s_mul_i32 s70, s70, s[sgprStrideC1J]               // Free1
s_add_u32 s68, s68, s70                            // Free1
s_addc_u32 s69, s69, s71                           // Free1
s_sub_u32 s70, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s70, s70, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s71, s70, s[sgprStrideCK]             // Free2
s_mul_i32 s70, s70, s[sgprStrideCK]                // Free2
s_add_u32 s68, s68, s70                            // Free2
s_addc_u32 s69, s69, s71                           // Free2
s_lshl_b64 s[68:69], s[68:69], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...64) from pool */

/* initC: remove ValuA/B vgpr buffer [10...74) from pool */
v_mov_b64 v[110:111], 0                            // A/B=0
v_accvgpr_write acc0, 0                            // initC
v_accvgpr_write acc1, 0                            // initC
v_accvgpr_write acc2, 0                            // initC
v_accvgpr_write acc3, 0                            // initC
v_accvgpr_write acc4, 0                            // initC
v_accvgpr_write acc5, 0                            // initC
v_accvgpr_write acc6, 0                            // initC
v_accvgpr_write acc7, 0                            // initC
v_accvgpr_write acc8, 0                            // initC
v_accvgpr_write acc9, 0                            // initC
v_accvgpr_write acc10, 0                           // initC
v_accvgpr_write acc11, 0                           // initC
v_accvgpr_write acc12, 0                           // initC
v_accvgpr_write acc13, 0                           // initC
v_accvgpr_write acc14, 0                           // initC
v_accvgpr_write acc15, 0                           // initC
v_mfma_i32_32x32x16_i8 acc[16:31], v[110:111], v[110:111], acc[0:15] // initC: [16, 31]
v_mfma_i32_32x32x16_i8 acc[32:47], v[110:111], v[110:111], acc[0:15] // initC: [32, 47]
v_mfma_i32_32x32x16_i8 acc[48:63], v[110:111], v[110:111], acc[0:15] // initC: [48, 63]
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc1
s_getpc_b64 s[68:69]                               // addr of next instr
s_add_i32 s70, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s68, s68, s70                            // add target branch offset
s_addc_u32 s69, s69, 0                             // add high and carry
s_setpc_b64 s[68:69]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_T8JHFHKM7BO5OHXW:

/* local write swap a */
s_xor_b32 s[sgprLocalWriteAddrA], 0x10000, s[sgprLocalWriteAddrA] // swap Red Blk SGPR

/* local write swap b */
s_xor_b32 s[sgprLocalWriteAddrB], 0x10000, s[sgprLocalWriteAddrB] // swap Red Blk SGPR
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0

/* local write swap a */
s_xor_b32 s[sgprLocalWriteAddrA], 0x10000, s[sgprLocalWriteAddrA] // swap Red Blk SGPR

/* local write swap b */
s_xor_b32 s[sgprLocalWriteAddrB], 0x10000, s[sgprLocalWriteAddrB] // swap Red Blk SGPR
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
s_waitcnt vmcnt(0)                                 // wait for global reads with lds (for early exit)
label_skipPGR2_2:
s_waitcnt vmcnt(8)                                 // wait for global reads with lds
// Skip force waitcnt0
s_barrier                                          // LW to PLR, sync LDS0

/* local read prefetch a */
ds_read_b128 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_T0_I0+0:vgprValuB_T0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_T0_I0+4:vgprValuB_T0_I0+4+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_toPGR1                        // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* subiter 0 */

/* subiter 1 */

/* subiter 2 */

/* subiter 3 */
.macro MAINLOOP ID, useGR=1, usePLR=1, useGRInc=1, useLoop=1
    /* mfmaIndex:0 */
    v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
    .if \useGRInc == 1
    s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
    .endif                                             // EndIf \useGRInc == 1
    ds_read_b128 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
    ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
    /* mfmaIndex:1 */
    v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
    .if \useGRInc == 1
    s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
    .endif                                             // EndIf \useGRInc == 1
    ds_read_b128 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
    ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
    /* mfmaIndex:2 */
    v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
    .if \useGRInc == 1
    s_cselect_b32 s69, s[sgprWrapUA+1], 0              // incUpper <- ?
    .endif                                             // EndIf \useGRInc == 1
    .if \useGRInc == 1
    s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
    .endif                                             // EndIf \useGRInc == 1
    .if \useGRInc == 1
    s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s69       // gra SRD += inc(upper)
    .endif                                             // EndIf \useGRInc == 1
    .if \useGRInc == 1
    s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
    .endif                                             // EndIf \useGRInc == 1
    .if \useGRInc == 1
    s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
    .endif                                             // EndIf \useGRInc == 1
    /* mfmaIndex:3 */
    v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
    s_waitcnt lgkmcnt(2)                               // Wait for the first 2 LRA0 to complete before pack
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+16], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+17], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
    /* mfmaIndex:4 */
    v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
    .if \useGRInc == 1
    s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+18], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+19], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23]
    /* mfmaIndex:5 */
    v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
    s_waitcnt lgkmcnt(0)                               // Wait for the rest    LRA0 to complete before pack
    .if \useGRInc == 1
    s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+24], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+25], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
    /* mfmaIndex:6 */
    v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
    .if \useGRInc == 1
    s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
    .endif                                             // EndIf \useGRInc == 1
    ds_read_b128 v[vgprValuB_T0_I0+8:vgprValuB_T0_I0+8+3], v[vgprLocalReadAddrB+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
    ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+26], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+27], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31]
    /* mfmaIndex:7 */
    v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
    ds_read_b128 v[vgprValuB_T0_I0+12:vgprValuB_T0_I0+12+3], v[vgprLocalReadAddrB+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
    ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[74:75], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+1], v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3] // Calculate low bits for TF32 emulation
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[74:75], v[vgprValuA_X0_I0+18:vgprValuA_X0_I0+18+1], v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[74:75], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+1], v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3] // Calculate low bits for TF32 emulation
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[74:75], v[vgprValuA_X0_I0+26:vgprValuA_X0_I0+26+1], v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
    /* mfmaIndex:8 */
    v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
    .if \useGRInc == 1
    s_cselect_b32 s68, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+23], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23] // pack final begin
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
    /* mfmaIndex:9 */
    v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
    .if \useGRInc == 1
    s_cselect_b32 s69, s[sgprWrapUB+1], 0              // incUpper <- ?
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+21], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+20], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9] // __TF32_2_A_2 pack final end
    /* mfmaIndex:10 */
    v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
    .if \useGRInc == 1
    s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+31], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31] // pack final begin
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
    /* mfmaIndex:11 */
    v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
    s_waitcnt lgkmcnt(0)                               // Wait for LRB0 to complete before pack
    s_barrier                                          // Wait for all waves to finish LRs before GRs
    .if \useGRInc == 1
    s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s69       // gra SRD += inc(upper)
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+29], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+28], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13] // __TF32_2_A_3 pack final end
    /* mfmaIndex:12 */
    v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
    .if \useGRInc == 1
    s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+16], v[vgprValuB_T0_I0+8], v[vgprValuB_T0_I0+9]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+17], v[vgprValuB_T0_I0+10], v[vgprValuB_T0_I0+11]
    /* mfmaIndex:13 */
    v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
    .if \useGRInc == 1
    s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+19], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23]
    .if \usePLR == 1
    v_xor_b32 v[vgprLocalReadAddrA], 0x10000, v[vgprLocalReadAddrA] // swap Red Blk
    .endif                                             // EndIf \usePLR == 1
    /* mfmaIndex:14 */
    v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
    .if \useGRInc == 1
    s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+24], v[vgprValuB_T0_I0+12], v[vgprValuB_T0_I0+13]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+25], v[vgprValuB_T0_I0+14], v[vgprValuB_T0_I0+15]
    .if \usePLR == 1
    v_xor_b32 v[vgprLocalReadAddrB], 0x10000, v[vgprLocalReadAddrB] // swap Red Blk
    .endif                                             // EndIf \usePLR == 1
    /* mfmaIndex:15 */
    v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
    .if \useGRInc == 1
    s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
    .endif                                             // EndIf \useGRInc == 1
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+27], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31]
    .if \useGR == 1
    s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
    .endif                                             // EndIf \useGR == 1
    /* mfmaIndex:16 */
    v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+8:vgprValuB_T0_I0+8+3], v[74:75], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1], v[vgprValuB_T0_I0+8:vgprValuB_T0_I0+8+3] // Calculate low bits for TF32 emulation
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[74:75], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1], v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+12:vgprValuB_T0_I0+12+3], v[74:75], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1], v[vgprValuB_T0_I0+12:vgprValuB_T0_I0+12+3] // Calculate low bits for TF32 emulation
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[74:75], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1], v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
    /* mfmaIndex:17 */
    v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+23], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23] // pack final begin
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
    .if \useGR == 1
    buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
    .endif                                             // EndIf \useGR == 1
    /* mfmaIndex:18 */
    v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+21], v[vgprValuB_T0_I0+10], v[vgprValuB_T0_I0+11]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+20], v[vgprValuB_T0_I0+8], v[vgprValuB_T0_I0+9] // __TF32_2_B_2 pack final end
    .if \useGR == 1
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    .endif                                             // EndIf \useGR == 1
    /* mfmaIndex:19 */
    v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+31], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31] // pack final begin
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
    .if \useGR == 1
    buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
    .endif                                             // EndIf \useGR == 1
    /* mfmaIndex:20 */
    v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+29], v[vgprValuB_T0_I0+14], v[vgprValuB_T0_I0+15]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+28], v[vgprValuB_T0_I0+12], v[vgprValuB_T0_I0+13] // __TF32_2_B_3 pack final end
    .if \useGR == 1
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    .endif                                             // EndIf \useGR == 1
    /* mfmaIndex:21 */
    v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
    .if \useGR == 1
    buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
    .endif                                             // EndIf \useGR == 1
    /* mfmaIndex:22 */
    v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
    /* mfmaIndex:23 */
    v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
    .if \useGR == 1 && \usePLR == 1
    /* vmcnt used in main loop */
    s_waitcnt vmcnt(3)                                 // Wait for previous GRA&B
    .elseif \useGR == 0 && \usePLR == 1
    /* vmcnt used in ngl, applying 8 shift */
    s_waitcnt vmcnt(0)                                 // Wait for previous GRA&B
    .elseif \useGR == 0 && \usePLR == 0
    /* vmcnt used in nll, applying 8 shift */
    s_waitcnt vmcnt(0)                                 // Wait for previous GRA&B
    .endif
    s_barrier
    /* mfmaIndex:24 */
    v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
    .if \usePLR == 1
    ds_read_b128 v[vgprValuB_T0_I0+0:vgprValuB_T0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
    .endif                                             // EndIf \usePLR == 1
    .if \usePLR == 1
    ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
    .endif                                             // EndIf \usePLR == 1
    /* mfmaIndex:25 */
    v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
    .if \useGR == 1
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    .endif                                             // EndIf \useGR == 1
    .if \usePLR == 1
    ds_read_b128 v[vgprValuB_T0_I0+4:vgprValuB_T0_I0+4+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
    .endif                                             // EndIf \usePLR == 1
    .if \usePLR == 1
    ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
    .endif                                             // EndIf \usePLR == 1
    /* mfmaIndex:26 */
    v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
    .if \useGR == 1
    buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
    .endif                                             // EndIf \useGR == 1
    /* mfmaIndex:27 */
    v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
    .if \useGR == 1
    s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
    .endif                                             // EndIf \useGR == 1
    /* mfmaIndex:28 */
    v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
    s_waitcnt lgkmcnt(2)                               // Wait for the first 2 LRB3 to complete
    .if \useGR == 1
    buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
    .endif                                             // EndIf \useGR == 1
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], v[vgprValuB_T0_I0+0], v[vgprValuB_T0_I0+1]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], v[vgprValuB_T0_I0+2], v[vgprValuB_T0_I0+3]
    /* mfmaIndex:29 */
    v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]
    /* mfmaIndex:30 */
    v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
    s_waitcnt lgkmcnt(0)                               // Wait for the rest    LRB3 to complete
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+8], v[vgprValuB_T0_I0+4], v[vgprValuB_T0_I0+5]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+9], v[vgprValuB_T0_I0+6], v[vgprValuB_T0_I0+7]
    /* mfmaIndex:31 */
    v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
    .if \useGR == 1
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    .endif                                             // EndIf \useGR == 1
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+11], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15]
    /* mfmaIndex:32 */
    v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+0:vgprValuB_T0_I0+0+3], v[74:75], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprValuB_T0_I0+0:vgprValuB_T0_I0+0+3] // Calculate low bits for TF32 emulation
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[74:75], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+4:vgprValuB_T0_I0+4+3], v[74:75], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1], v[vgprValuB_T0_I0+4:vgprValuB_T0_I0+4+3] // Calculate low bits for TF32 emulation
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[74:75], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1], v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
    /* mfmaIndex:33 */
    v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
    .if \useGR == 1
    buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
    .endif                                             // EndIf \useGR == 1
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7] // pack final begin
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
    /* mfmaIndex:34 */
    v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+5], v[vgprValuB_T0_I0+2], v[vgprValuB_T0_I0+3]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+4], v[vgprValuB_T0_I0+0], v[vgprValuB_T0_I0+1] // __TF32_2_B_0 pack final end
    /* mfmaIndex:35 */
    v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+15], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15] // pack final begin
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
    /* mfmaIndex:36 */
    v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
    .if \useGR == 1
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    .endif                                             // EndIf \useGR == 1
    .if \usePLR == 1
    ds_read_b128 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
    .endif                                             // EndIf \usePLR == 1
    .if \usePLR == 1
    ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
    .endif                                             // EndIf \usePLR == 1
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+13], v[vgprValuB_T0_I0+6], v[vgprValuB_T0_I0+7]
    v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+12], v[vgprValuB_T0_I0+4], v[vgprValuB_T0_I0+5] // __TF32_2_B_1 pack final end
    /* mfmaIndex:37 */
    v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
    .if \useGR == 1
    buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
    .endif                                             // EndIf \useGR == 1
    .if \usePLR == 1
    ds_read_b128 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
    .endif                                             // EndIf \usePLR == 1
    .if \usePLR == 1
    ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
    .endif                                             // EndIf \usePLR == 1
    /* mfmaIndex:38 */
    v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
    /* mfmaIndex:39 */
    v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
    s_waitcnt lgkmcnt(2)                               // Wait for the first 2 LRA3 to complete
    .if \useGR == 1
    s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
    .endif                                             // EndIf \useGR == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+1], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
    /* mfmaIndex:40 */
    v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
    .if \useGR == 1
    buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
    .endif                                             // EndIf \useGR == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7]
    /* mfmaIndex:41 */
    v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
    s_waitcnt lgkmcnt(0)                               // Wait for the rest    LRA3 to complete
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+8], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+9], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
    /* mfmaIndex:42 */
    v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+10], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+11], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]
    /* mfmaIndex:43 */
    v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[74:75], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3] // Calculate low bits for TF32 emulation
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[74:75], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[74:75], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3] // Calculate low bits for TF32 emulation
    v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[74:75], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
    /* mfmaIndex:44 */
    v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7] // pack final begin
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
    /* mfmaIndex:45 */
    v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
    .if \useGR == 1
    s_xor_b32 s[sgprLocalWriteAddrA], 0x10000, s[sgprLocalWriteAddrA] // swap Red Blk SGPR
    .endif                                             // EndIf \useGR == 1
    .if \useGR == 1
    s_xor_b32 s[sgprLocalWriteAddrB], 0x10000, s[sgprLocalWriteAddrB] // swap Red Blk SGPR
    .endif                                             // EndIf \useGR == 1
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+5], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+4], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1] // __TF32_2_A_0 pack final end
    /* mfmaIndex:46 */
    v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15] // pack final begin
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
    .if \useLoop == 1
    s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
    .endif                                             // EndIf \useLoop == 1
    /* mfmaIndex:47 */
    v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+13], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
    v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+12], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5] // __TF32_2_A_1 pack final end
    .if \useLoop == 1
    s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
    .endif                                             // EndIf \useLoop == 1
.endm
label_LoopBeginL_0:
MAINLOOP 0
s_cbranch_scc0 label_LoopBeginL_0
label_LoopEndL:

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop_1 - Begin        */
/******************************************/
/* Code-path 0, useGR=0, usePLR=1, useGRInc=1, useLoop = 0 */
MAINLOOP 0, 0, 1, 1, 0
label_toPGR1:
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1

/******************************************/
/* Opt. NoLoadLoop - Begin                */
/******************************************/
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_OptNLL_End                    // Branch if Beta is not zero

s_cmp_eq_u32 s[sgprAlpha], 1.0                     // Alpha == 1.0 ?
s_cbranch_scc0 label_OptNLL_End                    // branch if alpha != 1

s_and_b32 s68, 127, s[sgprSizeI]                   // s68 = s[sgprSizeI] % 128
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s68, 127, s[sgprSizeJ]                   // s68 = s[sgprSizeJ] % 128
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required

s_and_b32 s69, 31, s[sgprSizesSum+0]               // s69 = s[sgprSizesSum+0] % 32
s_cmp_eq_u32 s69, 0                                // numIterL == 0
s_cbranch_scc0 label_OptNLL_End                    // skip if tail loop required
/* Code-path 0, useGR=0, usePLR=0, useGRInc=0, useLoop = 0 */
MAINLOOP 0, 0, 0, 0, 0
label_toPGR1end_OptNLL:
/* Stores for OptNLL */
label_Summation_End_OptNLL:
/* endSummation: add vgpr [0...74) to pool */
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // 4 = Serial / 64
v_lshrrev_b32 v5, 1, v4                            // 5 = 4 / 2
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 1
v_and_b32 v1, 63, v[vgprSerial]                    // v1 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v1                            // 1 = 1 / 16
v_lshlrev_b32 v1, 2, v1                            // thread0 * continuous_output
v_add_lshl_u32 v1, v5, v1, 2                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v0, 1, v4                                // v0 = v4 % 2
v_mul_lo_u32 v0, 0x10, v0                          // wave coordination offset 0
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v0, v5, v0, 2                       // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v0, s8, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 128, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v1, s8, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/******************************************/
/* Global Write Elements                  */
/******************************************/
label_GW_B0_OptNLL_MB:
label_GW_B0_FD0_OptNLL_MB:
label_GW_B0_FD0_VW4_OptNLL_MB_Then:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=51 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_lshl_u32 v7, v3, v0, 2                       // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+16], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+17], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+18], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+19], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+20], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+21], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+22], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+23], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+24], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+25], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+26], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+27], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+28], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+29], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+30], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+31], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+32], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+33], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+34], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+35], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+36], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+37], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+38], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+39], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+40], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+41], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+42], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+43], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+44], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+45], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+46], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+47], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+48], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+49], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+50], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+51], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+52], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+53], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+54], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+55], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+56], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+57], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+58], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+59], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+60], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+61], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+62], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+63], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+64], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+65], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+66], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+67], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+68], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+69], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+70], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+71], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+112], acc51         // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+113], acc55         // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+114], acc59         // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+115], acc63         // copy acc to vreg[63]

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[16:19], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[20:23], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[28:31], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[36:39], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[44:47], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[52:55], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[60:63], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[68:71], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[112:115], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:

s_endpgm                                           // Kernel End
label_OptNLL_End:
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/
/* Code-path 0, useGR=0, usePLR=0, useGRInc=0, useLoop = 0 */
MAINLOOP 0, 0, 0, 0, 0
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [10...74) to pool */

/* Tail: add address/G2L vgpr [74...74) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */
s_and_b32 s[sgprLocalWriteAddrA], 0xf0ffff, s[sgprLocalWriteAddrA] // reset to Red

/* local write reset offsets b */
s_and_b32 s[sgprLocalWriteAddrB], 0xf0ffff, s[sgprLocalWriteAddrB] // reset to Red
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (16,16,0) */
.set vgprG2LA_BASE, 10
.set vgprG2LB_BASE, 26

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 31, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 32
s_and_b32 s68, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s69, s[sgprSizesSum], 5                 // s69 = s[sgprSizesSum] / 32
s_and_b32 s70, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v42, s70                             // s68 = s69 / s70
v_rcp_iflag_f32 v42, v42                           // s68 = s69 / s70
v_cvt_f32_u32 v43, s69                             // s68 = s69 / s70
v_mul_f32 v42, v42, v43                            // s68 = s69 / s70
v_cvt_u32_f32 v42, v42                             // s68 = s69 / s70
v_mul_u32_u24 v43, v42, s70                        // s68 = s69 / s70
v_sub_u32 v43, s69, v43                            // s68 = s69 / s70
v_cmpx_eq_u32 exec, v43, s70                       // s68 = s69 / s70
v_add_u32 v42, 1, v42                              // s68 = s69 / s70
v_mov_b32 v43, 0                                   // s[sgprGSUSumIdx+1] = s69 % s70
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v43, s70                       // overflow happened in remainder
v_sub_u32 v42, v42, 1                              // quotient - 1
v_mul_u32_u24 v43, v42, s70                        // re-calculate remainder
v_sub_u32 v43, s69, v43                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s68, v42                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v43        // remainder
s_sub_u32 s69, s70, 1                              // GSU-1
s_cmp_eq_u32 s68, 0                                // quotient == 0
s_cselect_b32 s68, s[sgprGSUSumIdx+1], s69         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s68                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
//  removeStagger A
s_sub_i32 s68, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s68, 0
s_cbranch_scc0 label_Negative_S4FDBQ587JJL6NOU
s_mul_hi_u32 s69, s68, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s68, s68, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_L43KTIIJOOEN7J6P
label_Negative_S4FDBQ587JJL6NOU:
s_abs_i32 s68, s68
s_mul_hi_u32 s69, s68, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s68, s68, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s68, s68, 0xffffffff
s_xor_b32 s69, s69, 0xffffffff
s_add_u32 s68, s68, 0x1
s_addc_u32 s69, s69, 0
label_MultiplyDone_L43KTIIJOOEN7J6P:
s_sub_u32 s68, s68, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s69, s69, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s69       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
//  removeStagger B
s_sub_i32 s68, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s68, 0
s_cbranch_scc0 label_Negative_UR8VN3A1SJCPC6PO
s_mul_hi_u32 s69, s68, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s68, s68, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_HYY06MPL0TYYIAT2
label_Negative_UR8VN3A1SJCPC6PO:
s_abs_i32 s68, s68
s_mul_hi_u32 s69, s68, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s68, s68, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s68, s68, 0xffffffff
s_xor_b32 s69, s69, 0xffffffff
s_add_u32 s68, s68, 0x1
s_addc_u32 s69, s69, 0
label_MultiplyDone_HYY06MPL0TYYIAT2:
s_sub_u32 s68, s68, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s69, s69, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s69       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */
s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
/* before DirectToLds load, ensure prior ds_reads have finished */
s_waitcnt lgkmcnt(0)
s_barrier

/* Tail global read A */
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0

/* Update M0 for DTLDS */
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address

/* Tail global read B */
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip force waitcnt0
s_barrier

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip force waitcnt0
s_barrier                                          // Tail loop LW->LR, sync LDS0
.set vgprG2LA_BASE, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprValuA_X0_I0_BASE, 10
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_T0_I0, 76
.set vgprValuB_X0_I0_BASE, 42
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_T0_I0, 92
.set IdentityMatrix, 74

/* Tail: local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA+0], 0xffff, v[vgprLocalReadAddrA+0] // reset Red,Blk -> Red

/* Tail: local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB+0], 0xffff, v[vgprLocalReadAddrB+0] // reset Red,Blk -> Red

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_read_b128 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_read_b128 v[vgprValuB_T0_I0+0:vgprValuB_T0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_T0_I0+4:vgprValuB_T0_I0+4+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_T0_I0+8:vgprValuB_T0_I0+8+3], v[vgprLocalReadAddrB+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_T0_I0+12:vgprValuB_T0_I0+12+3], v[vgprLocalReadAddrB+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
s_mov_b32 s8, 128                                  // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s8, v[vgprLocalReadAddrA+0] // lrA += 128 (bpeDS)

/* local read inc b */
                                                   // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s8, v[vgprLocalReadAddrB+0] // lrB += 128 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_and_b32 v109, 63, v[vgprSerial]                  // v109 = v[vgprSerial] % 64
v_lshrrev_b32 v109, 4, v109                        // 109 = 109 / 16
v_lshlrev_b32 v109, 2, v109                        // v109 = v109 * 4
v_add_u32 v110, v109, 0
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+0+0+0], v[vgprValuA_T0_I0+0+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+4+0+0], v[vgprValuA_T0_I0+4+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+8+0+0], v[vgprValuA_T0_I0+8+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+12+0+0], v[vgprValuA_T0_I0+12+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+1+0+0], v[vgprValuA_T0_I0+1+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+5+0+0], v[vgprValuA_T0_I0+5+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+9+0+0], v[vgprValuA_T0_I0+9+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+13+0+0], v[vgprValuA_T0_I0+13+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+2+0+0], v[vgprValuA_T0_I0+2+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+6+0+0], v[vgprValuA_T0_I0+6+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+10+0+0], v[vgprValuA_T0_I0+10+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+14+0+0], v[vgprValuA_T0_I0+14+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+3+0+0], v[vgprValuA_T0_I0+3+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+7+0+0], v[vgprValuA_T0_I0+7+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+11+0+0], v[vgprValuA_T0_I0+11+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+15+0+0], v[vgprValuA_T0_I0+15+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0], v[vgprValuA_X0_I0+4+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0], v[vgprValuA_X0_I0+12+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0], v[vgprValuA_X0_I0+20+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0], v[vgprValuA_X0_I0+28+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+5+0+0], v[vgprValuA_X0_I0+5+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+13+0+0], v[vgprValuA_X0_I0+13+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+21+0+0], v[vgprValuA_X0_I0+21+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+29+0+0], v[vgprValuA_X0_I0+29+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+6+0+0], v[vgprValuA_X0_I0+6+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+14+0+0], v[vgprValuA_X0_I0+14+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+22+0+0], v[vgprValuA_X0_I0+22+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+30+0+0], v[vgprValuA_X0_I0+30+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+7+0+0], v[vgprValuA_X0_I0+7+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+15+0+0], v[vgprValuA_X0_I0+15+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+23+0+0], v[vgprValuA_X0_I0+23+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+31+0+0], v[vgprValuA_X0_I0+31+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_and_b32 v109, 63, v[vgprSerial]                  // v109 = v[vgprSerial] % 64
v_lshrrev_b32 v109, 4, v109                        // 109 = 109 / 16
v_lshlrev_b32 v109, 2, v109                        // v109 = v109 * 4
v_add_u32 v110, v109, 0
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+0+0+0], v[vgprValuB_T0_I0+0+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+4+0+0], v[vgprValuB_T0_I0+4+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+8+0+0], v[vgprValuB_T0_I0+8+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+12+0+0], v[vgprValuB_T0_I0+12+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+1+0+0], v[vgprValuB_T0_I0+1+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+5+0+0], v[vgprValuB_T0_I0+5+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+9+0+0], v[vgprValuB_T0_I0+9+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+13+0+0], v[vgprValuB_T0_I0+13+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+2+0+0], v[vgprValuB_T0_I0+2+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+6+0+0], v[vgprValuB_T0_I0+6+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+10+0+0], v[vgprValuB_T0_I0+10+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+14+0+0], v[vgprValuB_T0_I0+14+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+3+0+0], v[vgprValuB_T0_I0+3+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+7+0+0], v[vgprValuB_T0_I0+7+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+11+0+0], v[vgprValuB_T0_I0+11+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_T0_I0+15+0+0], v[vgprValuB_T0_I0+15+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0], v[vgprValuB_X0_I0+4+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0], v[vgprValuB_X0_I0+12+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+20+0+0], v[vgprValuB_X0_I0+20+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+28+0+0], v[vgprValuB_X0_I0+28+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+5+0+0], v[vgprValuB_X0_I0+5+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+13+0+0], v[vgprValuB_X0_I0+13+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+21+0+0], v[vgprValuB_X0_I0+21+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+29+0+0], v[vgprValuB_X0_I0+29+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+6+0+0], v[vgprValuB_X0_I0+6+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+14+0+0], v[vgprValuB_X0_I0+14+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+22+0+0], v[vgprValuB_X0_I0+22+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+30+0+0], v[vgprValuB_X0_I0+30+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+7+0+0], v[vgprValuB_X0_I0+7+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+15+0+0], v[vgprValuB_X0_I0+15+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+23+0+0], v[vgprValuB_X0_I0+23+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+31+0+0], v[vgprValuB_X0_I0+31+0+0], 0, s[68:69] // set 0 if K_idx >= sizeL
s_and_b32 s70, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s70, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_8S4L1KCK9VFC7AQU // skip mask
s_and_b32 s70, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s70, 8, s70                              // use shift to fill 0 for outside element
s_lshl_b32 s70, s70, 5                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[112:113], s70, v[vgprValuA_T0_I0+0+0+0:vgprValuA_T0_I0+0+0+0+1]
v_lshlrev_b64 v[114:115], s70, v[vgprValuA_T0_I0+2+0+0:vgprValuA_T0_I0+2+0+0+1]
v_lshlrev_b64 v[116:117], s70, v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1]
v_lshlrev_b64 v[118:119], s70, v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1]
v_add_u32 v110, v109, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+0+0+0], v[vgprValuA_T0_I0+0+0+0], v112, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+1+0+0], v[vgprValuA_T0_I0+1+0+0], v113, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+2+0+0], v[vgprValuA_T0_I0+2+0+0], v114, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+3+0+0], v[vgprValuA_T0_I0+3+0+0], v115, s[68:69]
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0], v[vgprValuA_X0_I0+4+0+0], v116, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+5+0+0], v[vgprValuA_X0_I0+5+0+0], v117, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+6+0+0], v[vgprValuA_X0_I0+6+0+0], v118, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+7+0+0], v[vgprValuA_X0_I0+7+0+0], v119, s[68:69]
v_lshlrev_b64 v[112:113], s70, v[vgprValuA_T0_I0+4+0+0:vgprValuA_T0_I0+4+0+0+1]
v_lshlrev_b64 v[114:115], s70, v[vgprValuA_T0_I0+6+0+0:vgprValuA_T0_I0+6+0+0+1]
v_lshlrev_b64 v[116:117], s70, v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1]
v_lshlrev_b64 v[118:119], s70, v[vgprValuA_X0_I0+14+0+0:vgprValuA_X0_I0+14+0+0+1]
v_add_u32 v110, v109, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+4+0+0], v[vgprValuA_T0_I0+4+0+0], v112, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+5+0+0], v[vgprValuA_T0_I0+5+0+0], v113, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+6+0+0], v[vgprValuA_T0_I0+6+0+0], v114, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+7+0+0], v[vgprValuA_T0_I0+7+0+0], v115, s[68:69]
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0], v[vgprValuA_X0_I0+12+0+0], v116, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+13+0+0], v[vgprValuA_X0_I0+13+0+0], v117, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+14+0+0], v[vgprValuA_X0_I0+14+0+0], v118, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+15+0+0], v[vgprValuA_X0_I0+15+0+0], v119, s[68:69]
v_lshlrev_b64 v[112:113], s70, v[vgprValuA_T0_I0+8+0+0:vgprValuA_T0_I0+8+0+0+1]
v_lshlrev_b64 v[114:115], s70, v[vgprValuA_T0_I0+10+0+0:vgprValuA_T0_I0+10+0+0+1]
v_lshlrev_b64 v[116:117], s70, v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+1]
v_lshlrev_b64 v[118:119], s70, v[vgprValuA_X0_I0+22+0+0:vgprValuA_X0_I0+22+0+0+1]
v_add_u32 v110, v109, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+8+0+0], v[vgprValuA_T0_I0+8+0+0], v112, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+9+0+0], v[vgprValuA_T0_I0+9+0+0], v113, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+10+0+0], v[vgprValuA_T0_I0+10+0+0], v114, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+11+0+0], v[vgprValuA_T0_I0+11+0+0], v115, s[68:69]
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0], v[vgprValuA_X0_I0+20+0+0], v116, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+21+0+0], v[vgprValuA_X0_I0+21+0+0], v117, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+22+0+0], v[vgprValuA_X0_I0+22+0+0], v118, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+23+0+0], v[vgprValuA_X0_I0+23+0+0], v119, s[68:69]
v_lshlrev_b64 v[112:113], s70, v[vgprValuA_T0_I0+12+0+0:vgprValuA_T0_I0+12+0+0+1]
v_lshlrev_b64 v[114:115], s70, v[vgprValuA_T0_I0+14+0+0:vgprValuA_T0_I0+14+0+0+1]
v_lshlrev_b64 v[116:117], s70, v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+1]
v_lshlrev_b64 v[118:119], s70, v[vgprValuA_X0_I0+30+0+0:vgprValuA_X0_I0+30+0+0+1]
v_add_u32 v110, v109, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+12+0+0], v[vgprValuA_T0_I0+12+0+0], v112, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+13+0+0], v[vgprValuA_T0_I0+13+0+0], v113, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+14+0+0], v[vgprValuA_T0_I0+14+0+0], v114, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+15+0+0], v[vgprValuA_T0_I0+15+0+0], v115, s[68:69]
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0], v[vgprValuA_X0_I0+28+0+0], v116, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+29+0+0], v[vgprValuA_X0_I0+29+0+0], v117, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+30+0+0], v[vgprValuA_X0_I0+30+0+0], v118, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+31+0+0], v[vgprValuA_X0_I0+31+0+0], v119, s[68:69]
v_lshlrev_b64 v[112:113], s70, v[vgprValuB_T0_I0+0+0+0:vgprValuB_T0_I0+0+0+0+1]
v_lshlrev_b64 v[114:115], s70, v[vgprValuB_T0_I0+2+0+0:vgprValuB_T0_I0+2+0+0+1]
v_lshlrev_b64 v[116:117], s70, v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1]
v_lshlrev_b64 v[118:119], s70, v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1]
v_add_u32 v110, v109, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+0+0+0], v[vgprValuB_T0_I0+0+0+0], v112, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+1+0+0], v[vgprValuB_T0_I0+1+0+0], v113, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+2+0+0], v[vgprValuB_T0_I0+2+0+0], v114, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+3+0+0], v[vgprValuB_T0_I0+3+0+0], v115, s[68:69]
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0], v[vgprValuB_X0_I0+4+0+0], v116, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+5+0+0], v[vgprValuB_X0_I0+5+0+0], v117, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+6+0+0], v[vgprValuB_X0_I0+6+0+0], v118, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+7+0+0], v[vgprValuB_X0_I0+7+0+0], v119, s[68:69]
v_lshlrev_b64 v[112:113], s70, v[vgprValuB_T0_I0+4+0+0:vgprValuB_T0_I0+4+0+0+1]
v_lshlrev_b64 v[114:115], s70, v[vgprValuB_T0_I0+6+0+0:vgprValuB_T0_I0+6+0+0+1]
v_lshlrev_b64 v[116:117], s70, v[vgprValuB_X0_I0+12+0+0:vgprValuB_X0_I0+12+0+0+1]
v_lshlrev_b64 v[118:119], s70, v[vgprValuB_X0_I0+14+0+0:vgprValuB_X0_I0+14+0+0+1]
v_add_u32 v110, v109, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+4+0+0], v[vgprValuB_T0_I0+4+0+0], v112, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+5+0+0], v[vgprValuB_T0_I0+5+0+0], v113, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+6+0+0], v[vgprValuB_T0_I0+6+0+0], v114, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+7+0+0], v[vgprValuB_T0_I0+7+0+0], v115, s[68:69]
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+12+0+0], v[vgprValuB_X0_I0+12+0+0], v116, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+13+0+0], v[vgprValuB_X0_I0+13+0+0], v117, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+14+0+0], v[vgprValuB_X0_I0+14+0+0], v118, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+15+0+0], v[vgprValuB_X0_I0+15+0+0], v119, s[68:69]
v_lshlrev_b64 v[112:113], s70, v[vgprValuB_T0_I0+8+0+0:vgprValuB_T0_I0+8+0+0+1]
v_lshlrev_b64 v[114:115], s70, v[vgprValuB_T0_I0+10+0+0:vgprValuB_T0_I0+10+0+0+1]
v_lshlrev_b64 v[116:117], s70, v[vgprValuB_X0_I0+20+0+0:vgprValuB_X0_I0+20+0+0+1]
v_lshlrev_b64 v[118:119], s70, v[vgprValuB_X0_I0+22+0+0:vgprValuB_X0_I0+22+0+0+1]
v_add_u32 v110, v109, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+8+0+0], v[vgprValuB_T0_I0+8+0+0], v112, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+9+0+0], v[vgprValuB_T0_I0+9+0+0], v113, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+10+0+0], v[vgprValuB_T0_I0+10+0+0], v114, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+11+0+0], v[vgprValuB_T0_I0+11+0+0], v115, s[68:69]
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+20+0+0], v[vgprValuB_X0_I0+20+0+0], v116, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+21+0+0], v[vgprValuB_X0_I0+21+0+0], v117, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+22+0+0], v[vgprValuB_X0_I0+22+0+0], v118, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+23+0+0], v[vgprValuB_X0_I0+23+0+0], v119, s[68:69]
v_lshlrev_b64 v[112:113], s70, v[vgprValuB_T0_I0+12+0+0:vgprValuB_T0_I0+12+0+0+1]
v_lshlrev_b64 v[114:115], s70, v[vgprValuB_T0_I0+14+0+0:vgprValuB_T0_I0+14+0+0+1]
v_lshlrev_b64 v[116:117], s70, v[vgprValuB_X0_I0+28+0+0:vgprValuB_X0_I0+28+0+0+1]
v_lshlrev_b64 v[118:119], s70, v[vgprValuB_X0_I0+30+0+0:vgprValuB_X0_I0+30+0+0+1]
v_add_u32 v110, v109, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+12+0+0], v[vgprValuB_T0_I0+12+0+0], v112, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+13+0+0], v[vgprValuB_T0_I0+13+0+0], v113, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+14+0+0], v[vgprValuB_T0_I0+14+0+0], v114, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_T0_I0+15+0+0], v[vgprValuB_T0_I0+15+0+0], v115, s[68:69]
v_add_u32 v110, v110, 14                           // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+28+0+0], v[vgprValuB_X0_I0+28+0+0], v116, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+29+0+0], v[vgprValuB_X0_I0+29+0+0], v117, s[68:69]
v_add_u32 v110, v110, 2                            // add part of K
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+30+0+0], v[vgprValuB_X0_I0+30+0+0], v118, s[68:69]
v_cmp_ge_i32 s[68:69], v110, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+31+0+0], v[vgprValuB_X0_I0+31+0+0], v119, s[68:69]
label_TailLoop_SkipZeroOutMask_8S4L1KCK9VFC7AQU:
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+1], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[74:75], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[74:75], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], v[vgprValuB_T0_I0+0], v[vgprValuB_T0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], v[vgprValuB_T0_I0+2], v[vgprValuB_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+0:vgprValuB_T0_I0+0+3], v[74:75], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprValuB_T0_I0+0:vgprValuB_T0_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[74:75], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+5], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+4], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+5], v[vgprValuB_T0_I0+2], v[vgprValuB_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+4], v[vgprValuB_T0_I0+0], v[vgprValuB_T0_I0+1] // __TF32_2_B_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+8], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+9], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+10], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+11], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[74:75], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[74:75], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+8], v[vgprValuB_T0_I0+4], v[vgprValuB_T0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+9], v[vgprValuB_T0_I0+6], v[vgprValuB_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+11], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+4:vgprValuB_T0_I0+4+3], v[74:75], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1], v[vgprValuB_T0_I0+4:vgprValuB_T0_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[74:75], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1], v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+13], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+12], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+15], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+13], v[vgprValuB_T0_I0+6], v[vgprValuB_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+12], v[vgprValuB_T0_I0+4], v[vgprValuB_T0_I0+5] // __TF32_2_B_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+16], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+17], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+18], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+19], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[74:75], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+1], v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[74:75], v[vgprValuA_X0_I0+18:vgprValuA_X0_I0+18+1], v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+16], v[vgprValuB_T0_I0+8], v[vgprValuB_T0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+17], v[vgprValuB_T0_I0+10], v[vgprValuB_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+19], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+8:vgprValuB_T0_I0+8+3], v[74:75], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1], v[vgprValuB_T0_I0+8:vgprValuB_T0_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[74:75], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1], v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+23], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+21], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+20], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+23], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+21], v[vgprValuB_T0_I0+10], v[vgprValuB_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+20], v[vgprValuB_T0_I0+8], v[vgprValuB_T0_I0+9] // __TF32_2_B_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+24], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+25], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+26], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+27], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[74:75], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+1], v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[74:75], v[vgprValuA_X0_I0+26:vgprValuA_X0_I0+26+1], v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+24], v[vgprValuB_T0_I0+12], v[vgprValuB_T0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+25], v[vgprValuB_T0_I0+14], v[vgprValuB_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+27], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+12:vgprValuB_T0_I0+12+3], v[74:75], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1], v[vgprValuB_T0_I0+12:vgprValuB_T0_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[74:75], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1], v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+31], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+29], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+28], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+31], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+29], v[vgprValuB_T0_I0+14], v[vgprValuB_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+28], v[vgprValuB_T0_I0+12], v[vgprValuB_T0_I0+13] // __TF32_2_B_3 pack final end
s_nop 1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_T0_I0, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_T0_I0, UNDEF
.set IdentityMatrix, UNDEF

/* Tail: add MISC Vgpr [0...10) to pool */
label_Summation_End_ZU0B7F2XE71N7LVL:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // 4 = Serial / 64
v_lshrrev_b32 v5, 1, v4                            // 5 = 4 / 2
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 1
v_and_b32 v1, 63, v[vgprSerial]                    // v1 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v1                            // 1 = 1 / 16
v_lshlrev_b32 v1, 2, v1                            // thread0 * continuous_output
v_add_lshl_u32 v1, v5, v1, 2                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v0, 1, v4                                // v0 = v4 % 2
v_mul_lo_u32 v0, 0x10, v0                          // wave coordination offset 0
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v0, v5, v0, 2                       // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v0, s8, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 128, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v1, s8, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:
s_and_b32 s30, 127, s[sgprSizeI]                   // s30 = s[sgprSizeI] % 128
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MB_Else         // jump if edges required
s_and_b32 s30, 127, s[sgprSizeJ]                   // s30 = s[sgprSizeJ] % 128
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MB_Then         // jump if edges required
label_GW_B0_FD0_VW4_MB_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=51 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_lshl_u32 v7, v3, v0, 2                       // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+16], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+17], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+18], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+19], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+20], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+21], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+22], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+23], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+24], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+25], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+26], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+27], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+28], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+29], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+30], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+31], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+32], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+33], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+34], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+35], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+36], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+37], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+38], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+39], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+40], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+41], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+42], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+43], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+44], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+45], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+46], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+47], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+48], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+49], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+50], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+51], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+52], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+53], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+54], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+55], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+56], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+57], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+58], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+59], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+60], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+61], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+62], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+63], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+64], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+65], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+66], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+67], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+68], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+69], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+70], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+71], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+112], acc51         // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+113], acc55         // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+114], acc59         // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+115], acc63         // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[16:19], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[20:23], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[28:31], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[36:39], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[44:47], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[52:55], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[60:63], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[68:71], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[112:115], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_MB_NonEdgeEnd:
label_GW_B0_FD0_VW4_MB_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=41 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v7, v3, v0, 2                       // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v7, v6, v7, s[34:35]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v72, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v73, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v6, v73, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v109, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v110, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v111, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v112, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v6, v112, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v113, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v6, v113, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v114, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v6, v114, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v115, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v6, v115, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v116, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v117, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v118, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v6, v118, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v119, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v6, v119, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v120, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v6, v120, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v121, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v6, v121, s[34:35]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+24], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+25], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+26], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+27], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+28], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+29], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+30], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+31], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+32], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+33], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+34], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+35], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+36], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+37], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+38], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+39], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+44], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+45], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+46], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+47], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+48], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+49], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+50], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+51], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+52], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+53], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+54], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+55], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+56], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+57], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+58], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+59], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+60], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+61], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+62], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+63], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+64], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+65], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+66], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+67], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+68], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+69], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+70], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+71], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[8:11], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[12:15], v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[16:19], v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[20:23], v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[24:27], v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[28:31], v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[32:35], v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[36:39], v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[40:43], v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[44:47], v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[48:51], v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[52:55], v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[56:59], v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[60:63], v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[64:67], v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[68:71], v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_MB_Else:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=106 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,8,0:vw1); (0,0,8,1:vw1); (0,0,8,2:vw1); (0,0,8,3:vw1); (0,0,9,0:vw1); (0,0,9,1:vw1); (0,0,9,2:vw1); (0,0,9,3:vw1); (0,0,10,0:vw1); (0,0,10,1:vw1); (0,0,10,2:vw1); (0,0,10,3:vw1); (0,0,11,0:vw1); (0,0,11,1:vw1); (0,0,11,2:vw1); (0,0,11,3:vw1); (0,0,12,0:vw1); (0,0,12,1:vw1); (0,0,12,2:vw1); (0,0,12,3:vw1); (0,0,13,0:vw1); (0,0,13,1:vw1); (0,0,13,2:vw1); (0,0,13,3:vw1); (0,0,14,0:vw1); (0,0,14,1:vw1); (0,0,14,2:vw1); (0,0,14,3:vw1); (0,0,15,0:vw1); (0,0,15,1:vw1); (0,0,15,2:vw1); (0,0,15,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v71, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v6, v71, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v72, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v73, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v6, v73, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v109, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v110, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v111, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v112, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v6, v112, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v113, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v6, v113, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v114, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v6, v114, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v115, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v6, v115, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v116, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v117, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v118, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v6, v118, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v119, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v6, v119, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v120, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v6, v120, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v121, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v6, v121, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v122, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v6, v122, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v123, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v6, v123, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v124, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v6, v124, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v125, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v6, v125, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v126, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v6, v126, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v127, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v6, v127, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v128, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v6, v128, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v129, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v6, v129, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v130, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v6, v130, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v131, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v6, v131, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v132, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v6, v132, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v133, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v6, v133, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v134, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v6, v134, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v135, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v6, v135, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v136, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v6, v136, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v137, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v6, v137, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v138, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v6, v138, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v139, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v6, v139, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v140, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v6, v140, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v141, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v6, v141, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v142, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v6, v142, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v143, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v6, v143, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v144, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v6, v144, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v145, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v6, v145, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v146, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v6, v146, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v147, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v6, v147, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v148, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v6, v148, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v149, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v6, v149, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v150, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v6, v150, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v151, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v6, v151, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v152, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v6, v152, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v153, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v6, v153, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v154, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v6, v154, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v155, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v6, v155, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v156, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v6, v156, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v157, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v6, v157, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v158, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v6, v158, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v159, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v6, v159, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v160, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v6, v160, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v161, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v6, v161, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v162, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v6, v162, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v163, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v6, v163, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v164, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v6, v164, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v165, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v6, v165, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v166, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v6, v166, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v167, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v6, v167, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v168, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v6, v168, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v169, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v6, v169, s[34:35]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+7], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+8], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+9], acc8            // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+10], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+11], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+12], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+13], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+14], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+15], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+16], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+17], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+18], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+19], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+20], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+21], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+22], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+23], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+24], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+25], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+26], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+27], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+28], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+29], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+30], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+31], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+32], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+33], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+34], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+35], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+36], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+37], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+38], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+39], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+40], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+41], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+42], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+43], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+44], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+45], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+46], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+47], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+48], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+49], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+50], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+51], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+52], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+53], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+54], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+55], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+56], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+57], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+58], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+59], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+60], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+61], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+62], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+63], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+64], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+65], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+66], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+67], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+68], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+69], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+70], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 8, 0), (0, 0, 8, 1), (0, 0, 8, 2), (0, 0, 8, 3), (0, 0, 9, 0), (0, 0, 9, 1), (0, 0, 9, 2), (0, 0, 9, 3), (0, 0, 10, 0), (0, 0, 10, 1), (0, 0, 10, 2), (0, 0, 10, 3), (0, 0, 11, 0), (0, 0, 11, 1), (0, 0, 11, 2), (0, 0, 11, 3), (0, 0, 12, 0), (0, 0, 12, 1), (0, 0, 12, 2), (0, 0, 12, 3), (0, 0, 13, 0), (0, 0, 13, 1), (0, 0, 13, 2), (0, 0, 13, 3), (0, 0, 14, 0), (0, 0, 14, 1), (0, 0, 14, 2), (0, 0, 14, 3), (0, 0, 15, 0), (0, 0, 15, 1), (0, 0, 15, 2), (0, 0, 15, 3)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v7, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v8, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v9, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v10, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v11, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v12, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v13, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v14, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v15, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v16, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v17, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v18, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v19, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v20, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v21, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v25, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v26, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v27, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v29, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v30, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v31, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v33, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v34, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v35, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v37, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v38, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v39, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v40, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v41, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v42, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v43, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v44, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v45, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v46, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v47, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v48, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v49, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v50, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v51, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v52, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v53, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v54, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v55, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v56, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v57, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v58, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v59, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v60, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v61, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v62, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v63, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v64, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v65, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v66, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v67, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v68, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v69, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v70, v169, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
s_getpc_b64 s[30:31]                               // addr of next instr
s_add_i32 s32, label_KernelEnd, 4                  // target branch offset
s_add_u32 s30, s30, s32                            // add target branch offset
s_addc_u32 s31, s31, 0                             // add high and carry
s_setpc_b64 s[30:31]                               // branch to label_KernelEnd
label_GSU_4:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:
s_and_b32 s30, 127, s[sgprSizeI]                   // s30 = s[sgprSizeI] % 128
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Else       // jump if edges required
s_and_b32 s30, 127, s[sgprSizeJ]                   // s30 = s[sgprSizeJ] % 128
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=51 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_lshl_u32 v7, v3, v0, 2                       // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+16], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+17], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+18], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+19], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+20], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+21], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+22], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+23], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+24], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+25], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+26], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+27], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+28], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+29], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+30], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+31], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+32], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+33], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+34], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+35], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+36], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+37], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+38], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+39], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+40], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+41], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+42], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+43], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+44], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+45], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+46], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+47], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+48], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+49], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+50], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+51], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+52], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+53], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+54], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+55], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+56], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+57], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+58], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+59], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+60], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+61], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+62], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+63], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+64], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+65], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+66], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+67], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+68], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+69], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+70], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+71], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+112], acc51         // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+113], acc55         // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+114], acc59         // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+115], acc63         // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+112:vgprValuC+112+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+112:vgprValuC+112+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+114:vgprValuC+114+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+114:vgprValuC+114+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[16:19], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[20:23], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[28:31], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[36:39], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[44:47], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[52:55], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[60:63], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[68:71], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[112:115], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=41 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v7, v3, v0, 2                       // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v7, v6, v7, s[34:35]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v72, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v73, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v6, v73, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v109, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v110, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v111, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v112, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v6, v112, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v113, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v6, v113, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v114, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v6, v114, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v115, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v6, v115, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v116, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v117, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v118, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v6, v118, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v119, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v6, v119, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v120, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v6, v120, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v121, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v6, v121, s[34:35]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+24], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+25], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+26], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+27], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+28], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+29], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+30], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+31], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+32], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+33], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+34], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+35], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+36], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+37], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+38], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+39], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+44], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+45], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+46], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+47], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+48], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+49], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+50], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+51], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+52], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+53], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+54], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+55], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+56], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+57], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+58], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+59], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+60], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+61], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+62], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+63], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+64], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+65], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+66], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+67], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+68], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+69], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+70], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+71], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[8:11], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[12:15], v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[16:19], v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[20:23], v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[24:27], v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[28:31], v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[32:35], v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[36:39], v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[40:43], v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[44:47], v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[48:51], v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[52:55], v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[56:59], v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[60:63], v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[64:67], v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[68:71], v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW4_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=106 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,8,0:vw1); (0,0,8,1:vw1); (0,0,8,2:vw1); (0,0,8,3:vw1); (0,0,9,0:vw1); (0,0,9,1:vw1); (0,0,9,2:vw1); (0,0,9,3:vw1); (0,0,10,0:vw1); (0,0,10,1:vw1); (0,0,10,2:vw1); (0,0,10,3:vw1); (0,0,11,0:vw1); (0,0,11,1:vw1); (0,0,11,2:vw1); (0,0,11,3:vw1); (0,0,12,0:vw1); (0,0,12,1:vw1); (0,0,12,2:vw1); (0,0,12,3:vw1); (0,0,13,0:vw1); (0,0,13,1:vw1); (0,0,13,2:vw1); (0,0,13,3:vw1); (0,0,14,0:vw1); (0,0,14,1:vw1); (0,0,14,2:vw1); (0,0,14,3:vw1); (0,0,15,0:vw1); (0,0,15,1:vw1); (0,0,15,2:vw1); (0,0,15,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v71, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v6, v71, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v72, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v73, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v6, v73, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v109, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v110, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v111, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v112, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v6, v112, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v113, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v6, v113, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v114, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v6, v114, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v115, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v6, v115, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v116, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v117, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v118, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v6, v118, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v119, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v6, v119, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v120, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v6, v120, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v121, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v6, v121, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v122, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v6, v122, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v123, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v6, v123, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v124, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v6, v124, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v125, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v6, v125, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v126, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v6, v126, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v127, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v6, v127, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v128, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v6, v128, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v129, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v6, v129, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v130, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v6, v130, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v131, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v6, v131, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v132, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v6, v132, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v133, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v6, v133, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v134, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v6, v134, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v135, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v6, v135, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v136, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v6, v136, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v137, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v6, v137, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v138, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v6, v138, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v139, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v6, v139, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v140, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v6, v140, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v141, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v6, v141, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v142, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v6, v142, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v143, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v6, v143, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v144, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v6, v144, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v145, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v6, v145, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v146, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v6, v146, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v147, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v6, v147, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v148, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v6, v148, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v149, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v6, v149, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v150, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v6, v150, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v151, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v6, v151, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v152, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v6, v152, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v153, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v6, v153, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v154, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v6, v154, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v155, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v6, v155, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v156, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v6, v156, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v157, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v6, v157, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v158, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v6, v158, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v159, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v6, v159, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v160, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v6, v160, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v161, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v6, v161, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v162, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v6, v162, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v163, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v6, v163, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v164, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v164, v6, v164, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v165, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v6, v165, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v166, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v6, v166, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v167, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v6, v167, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v168, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v168, v6, v168, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v169, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v6, v169, s[34:35]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+7], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+8], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+9], acc8            // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+10], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+11], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+12], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+13], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+14], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+15], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+16], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+17], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+18], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+19], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+20], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+21], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+22], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+23], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+24], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+25], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+26], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+27], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+28], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+29], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+30], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+31], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+32], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+33], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+34], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+35], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+36], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+37], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+38], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+39], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+40], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+41], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+42], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+43], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+44], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+45], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+46], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+47], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+48], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+49], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+50], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+51], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+52], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+53], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+54], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+55], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+56], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+57], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+58], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+59], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+60], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+61], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+62], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+63], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+64], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+65], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+66], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+67], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+68], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+69], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+70], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 8, 0), (0, 0, 8, 1), (0, 0, 8, 2), (0, 0, 8, 3), (0, 0, 9, 0), (0, 0, 9, 1), (0, 0, 9, 2), (0, 0, 9, 3), (0, 0, 10, 0), (0, 0, 10, 1), (0, 0, 10, 2), (0, 0, 10, 3), (0, 0, 11, 0), (0, 0, 11, 1), (0, 0, 11, 2), (0, 0, 11, 3), (0, 0, 12, 0), (0, 0, 12, 1), (0, 0, 12, 2), (0, 0, 12, 3), (0, 0, 13, 0), (0, 0, 13, 1), (0, 0, 13, 2), (0, 0, 13, 3), (0, 0, 14, 0), (0, 0, 14, 1), (0, 0, 14, 2), (0, 0, 14, 3), (0, 0, 15, 0), (0, 0, 15, 1), (0, 0, 15, 2), (0, 0, 15, 3)] */
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha

/* apply mask, calc new C and issue writes */
buffer_store_dword v7, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v8, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v9, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v10, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v11, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v12, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v13, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v14, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v15, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v16, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v17, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v18, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v19, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v20, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v21, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v25, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v26, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v27, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v29, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v30, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v31, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v33, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v34, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v35, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v37, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v38, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v39, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v40, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v41, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v42, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v43, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v44, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v45, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v46, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v47, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v48, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v49, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v50, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v51, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v52, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v53, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v54, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v55, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v56, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v57, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v58, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v59, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v60, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v61, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v62, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v63, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v64, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v65, v164, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v66, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v67, v166, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v68, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v69, v168, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v70, v169, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:
s_and_b32 s30, 127, s[sgprSizeI]                   // s30 = s[sgprSizeI] % 128
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Else       // jump if edges required
s_and_b32 s30, 127, s[sgprSizeJ]                   // s30 = s[sgprSizeJ] % 128
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=25 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v8, v2, v0, 2                       // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
buffer_load_dwordx4 v[116:119], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[120:123], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[124:127], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[128:131], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[132:135], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[140:143], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[144:147], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[148:151], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[152:155], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[156:159], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[160:163], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[164:167], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[168:171], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[172:175], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
s_lshl_b32 s12, s[sgprStrideC1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[176:179], v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v7, v3, v0, 2                       // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+16], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+17], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+18], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+19], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+20], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+21], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+22], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+23], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+24], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+25], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+26], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+27], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+28], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+29], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+30], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+31], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+32], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+33], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+34], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+35], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+36], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+37], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+38], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+39], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+40], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+41], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+42], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+43], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+44], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+45], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+46], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+47], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+48], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+49], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+50], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+51], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+52], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+53], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+54], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+55], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+56], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+57], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+58], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+59], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+60], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+61], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+62], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+63], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+64], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+65], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+66], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+67], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+68], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+69], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+70], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+71], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+112], acc51         // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+113], acc55         // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+114], acc59         // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+115], acc63         // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+112:vgprValuC+112+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+112:vgprValuC+112+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+114:vgprValuC+114+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+114:vgprValuC+114+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // vlcnt(15) = 16 - 1 (beta) vscnt(0) (interleaved)
v_fmac_f32 v[vgprValuC+12], v116, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+13], v117, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+14], v118, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+15], v119, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[12:15], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(14) = 16 - 2 (beta) vscnt(1) (interleaved)
v_fmac_f32 v[vgprValuC+16], v120, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+17], v121, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+18], v122, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+19], v123, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[16:19], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(13) = 16 - 3 (beta) vscnt(2) (interleaved)
v_fmac_f32 v[vgprValuC+20], v124, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+21], v125, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+22], v126, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+23], v127, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[20:23], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(12) = 16 - 4 (beta) vscnt(3) (interleaved)
v_fmac_f32 v[vgprValuC+24], v128, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+25], v129, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+26], v130, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+27], v131, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(11) = 16 - 5 (beta) vscnt(4) (interleaved)
v_fmac_f32 v[vgprValuC+28], v132, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+29], v133, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+30], v134, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+31], v135, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[28:31], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(10) = 16 - 6 (beta) vscnt(5) (interleaved)
v_fmac_f32 v[vgprValuC+32], v136, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+33], v137, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+34], v138, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+35], v139, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(9) = 16 - 7 (beta) vscnt(6) (interleaved)
v_fmac_f32 v[vgprValuC+36], v140, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+37], v141, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+38], v142, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+39], v143, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[36:39], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(8) = 16 - 8 (beta) vscnt(7) (interleaved)
v_fmac_f32 v[vgprValuC+40], v144, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+41], v145, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+42], v146, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+43], v147, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(7) = 16 - 9 (beta) vscnt(8) (interleaved)
v_fmac_f32 v[vgprValuC+44], v148, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+45], v149, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+46], v150, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+47], v151, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[44:47], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(6) = 16 - 10 (beta) vscnt(9) (interleaved)
v_fmac_f32 v[vgprValuC+48], v152, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+49], v153, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+50], v154, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+51], v155, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(5) = 16 - 11 (beta) vscnt(10) (interleaved)
v_fmac_f32 v[vgprValuC+52], v156, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+53], v157, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+54], v158, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+55], v159, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[52:55], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(4) = 16 - 12 (beta) vscnt(11) (interleaved)
v_fmac_f32 v[vgprValuC+56], v160, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+57], v161, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+58], v162, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+59], v163, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(3) = 16 - 13 (beta) vscnt(12) (interleaved)
v_fmac_f32 v[vgprValuC+60], v164, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+61], v165, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+62], v166, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+63], v167, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[60:63], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(2) = 16 - 14 (beta) vscnt(13) (interleaved)
v_fmac_f32 v[vgprValuC+64], v168, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+65], v169, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+66], v170, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+67], v171, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(1) = 16 - 15 (beta) vscnt(14) (interleaved)
v_fmac_f32 v[vgprValuC+68], v172, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+69], v173, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+70], v174, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+71], v175, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[68:71], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(0) = 16 - 16 (beta) vscnt(15) (interleaved)
v_fmac_f32 v[vgprValuC+112], v176, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+113], v177, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+114], v178, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+115], v179, s[sgprBeta]     // finalSum = sum*alpha + C*beta
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[112:115], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=23 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v7, v2, v0, 2                       // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v7, v6, v7, s[34:35]                 // LDC clip if OOB. offset
buffer_load_dwordx4 v[112:115], v7, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v7, v3, v0, 2                       // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v7, v6, v7, s[34:35]                 // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v72, v2, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[34:35]               // LDC clip if OOB. offset
buffer_load_dwordx4 v[116:119], v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v72, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v73, v2, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v6, v73, s[34:35]               // LDC clip if OOB. offset
buffer_load_dwordx4 v[120:123], v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v73, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v6, v73, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v109, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[124:127], v109, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v109, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v110, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[128:131], v110, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v110, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v111, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[132:135], v111, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v111, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v140, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v6, v140, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[136:139], v140, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v140, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v6, v140, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v141, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v6, v141, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[144:147], v141, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v141, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v6, v141, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v142, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v6, v142, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[148:151], v142, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v142, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v6, v142, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v143, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v6, v143, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[152:155], v143, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v143, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v6, v143, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v160, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v6, v160, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[156:159], v160, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v160, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v6, v160, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v161, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v6, v161, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[164:167], v161, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v161, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v6, v161, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v162, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v6, v162, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[168:171], v162, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v162, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v6, v162, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v163, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v6, v163, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[172:175], v163, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v163, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v6, v163, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v180, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v6, v180, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[176:179], v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v6, v180, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v181, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v6, v181, s[34:35]             // LDC clip if OOB. offset
buffer_load_dwordx4 v[184:187], v181, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v181, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v6, v181, s[34:35]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+16], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+17], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+18], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+19], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+20], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+21], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+22], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+23], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+24], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+25], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+26], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+27], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+28], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+29], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+30], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+31], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+32], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+33], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+34], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+35], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+36], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+37], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+38], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+39], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+44], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+45], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+46], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+47], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+48], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+49], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+50], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+51], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+52], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+53], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+54], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+55], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+56], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+57], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+58], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+59], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+60], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+61], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+62], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+63], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+64], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+65], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+66], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+67], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+68], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+69], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+70], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+71], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+70:vgprValuC+70+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+70:vgprValuC+70+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+8], v112, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+9], v113, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+10], v114, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+11], v115, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[8:11], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+12], v116, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+13], v117, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+14], v118, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+15], v119, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[12:15], v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+16], v120, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+17], v121, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+18], v122, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+19], v123, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[16:19], v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+20], v124, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+21], v125, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+22], v126, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+23], v127, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[20:23], v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+24], v128, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+25], v129, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+26], v130, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+27], v131, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[24:27], v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+28], v132, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+29], v133, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+30], v134, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+31], v135, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[28:31], v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+32], v136, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+33], v137, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+34], v138, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+35], v139, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[32:35], v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+36], v144, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+37], v145, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+38], v146, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+39], v147, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[36:39], v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+40], v148, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+41], v149, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+42], v150, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+43], v151, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[40:43], v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+44], v152, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+45], v153, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+46], v154, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+47], v155, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[44:47], v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+48], v156, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+49], v157, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+50], v158, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+51], v159, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[48:51], v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+52], v164, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+53], v165, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+54], v166, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+55], v167, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[52:55], v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+56], v168, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+57], v169, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+58], v170, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+59], v171, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[56:59], v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+60], v172, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+61], v173, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+62], v174, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+63], v175, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[60:63], v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+64], v176, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+65], v177, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+66], v178, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+67], v179, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[64:67], v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+68], v184, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+69], v185, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+70], v186, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+71], v187, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[68:71], v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW4_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=71 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,8,0:vw1); (0,0,8,1:vw1); (0,0,8,2:vw1); (0,0,8,3:vw1); (0,0,9,0:vw1); (0,0,9,1:vw1); (0,0,9,2:vw1); (0,0,9,3:vw1); (0,0,10,0:vw1); (0,0,10,1:vw1); (0,0,10,2:vw1); (0,0,10,3:vw1); (0,0,11,0:vw1); (0,0,11,1:vw1); (0,0,11,2:vw1); (0,0,11,3:vw1); (0,0,12,0:vw1); (0,0,12,1:vw1); (0,0,12,2:vw1); (0,0,12,3:vw1); (0,0,13,0:vw1); (0,0,13,1:vw1); (0,0,13,2:vw1); (0,0,13,3:vw1); (0,0,14,0:vw1); (0,0,14,1:vw1); (0,0,14,2:vw1); (0,0,14,3:vw1); (0,0,15,0:vw1); (0,0,15,1:vw1); (0,0,15,2:vw1); (0,0,15,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v72, v2, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[34:35]               // LDC clip if OOB. offset
buffer_load_dword v71, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v72, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[34:35]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v109, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v73, v109, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v109, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v111, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v110, v111, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v111, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v113, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v6, v113, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v112, v113, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v113, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v6, v113, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v115, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v6, v115, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v114, v115, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v115, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v6, v115, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v117, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v116, v117, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v117, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v119, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v6, v119, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v118, v119, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v119, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v6, v119, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v121, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v6, v121, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v120, v121, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v121, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v6, v121, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v123, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v6, v123, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v122, v123, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v123, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v6, v123, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v125, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v6, v125, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v124, v125, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v125, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v6, v125, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v127, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v6, v127, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v126, v127, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v127, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v6, v127, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v129, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v6, v129, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v128, v129, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v129, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v6, v129, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v131, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v6, v131, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v130, v131, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v131, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v6, v131, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v133, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v6, v133, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v132, v133, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v133, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v6, v133, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v135, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v6, v135, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v134, v135, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v135, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v6, v135, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v137, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v6, v137, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v136, v137, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v137, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v6, v137, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v139, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v6, v139, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v138, v139, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v139, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v6, v139, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v141, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v6, v141, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v140, v141, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v141, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v6, v141, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v143, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v6, v143, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v142, v143, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v143, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v6, v143, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v145, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v6, v145, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v144, v145, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v145, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v6, v145, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v147, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v6, v147, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v146, v147, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v147, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v6, v147, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v149, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v6, v149, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v148, v149, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v149, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v6, v149, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v151, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v6, v151, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v150, v151, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v151, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v6, v151, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v153, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v6, v153, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v152, v153, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v153, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v6, v153, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v155, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v6, v155, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v154, v155, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v155, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v6, v155, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v157, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v6, v157, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v156, v157, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v157, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v6, v157, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v159, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v6, v159, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v158, v159, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v159, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v6, v159, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v161, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v6, v161, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v160, v161, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v161, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v6, v161, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v163, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v6, v163, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v162, v163, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v163, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v6, v163, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v165, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v6, v165, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v164, v165, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v165, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v6, v165, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v167, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v6, v167, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v166, v167, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v167, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v6, v167, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v169, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v6, v169, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v168, v169, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v169, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v6, v169, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v171, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v171, v6, v171, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v170, v171, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v171, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v171, v6, v171, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v173, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v6, v173, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v172, v173, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v173, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v6, v173, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v175, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v6, v175, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v174, v175, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v175, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v6, v175, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v177, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v6, v177, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v176, v177, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v177, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v6, v177, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v179, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v6, v179, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v178, v179, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v179, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v6, v179, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v181, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v6, v181, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v180, v181, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v181, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v6, v181, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v183, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v6, v183, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v182, v183, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v183, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v6, v183, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v185, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v6, v185, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v184, v185, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v185, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v6, v185, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v187, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v6, v187, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v186, v187, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v187, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v187, v6, v187, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v189, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v6, v189, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v188, v189, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v189, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v189, v6, v189, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v191, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v6, v191, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v190, v191, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v191, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v6, v191, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v193, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v6, v193, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v192, v193, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v193, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v193, v6, v193, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v195, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v6, v195, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v194, v195, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v195, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v195, v6, v195, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v197, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v6, v197, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v196, v197, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v197, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v197, v6, v197, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v199, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v6, v199, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v198, v199, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v199, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v199, v6, v199, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v201, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v6, v201, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v200, v201, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v201, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v6, v201, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v203, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v6, v203, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v202, v203, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v203, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v6, v203, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v205, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v6, v205, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v204, v205, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v205, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v6, v205, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v207, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v6, v207, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v206, v207, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v207, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v6, v207, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v209, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v6, v209, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v208, v209, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v209, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v6, v209, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v211, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v6, v211, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v210, v211, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v211, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v6, v211, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v213, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v6, v213, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v212, v213, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v213, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v6, v213, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v215, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v6, v215, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v214, v215, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v215, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v6, v215, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v217, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v6, v217, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v216, v217, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v217, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v6, v217, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v219, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v6, v219, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v218, v219, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v219, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v6, v219, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v221, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v6, v221, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v220, v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v6, v221, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v223, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v6, v223, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v222, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v6, v223, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v225, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v6, v225, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v225, v6, v225, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[30:31], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v227, v2, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v6, v227, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v226, v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v227, v6, v227, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v229, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v6, v229, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v228, v229, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v229, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v6, v229, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v231, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v6, v231, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v230, v231, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v231, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v6, v231, s[34:35]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[30:31], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[34:35], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[34:35], s[30:31], s[34:35]             // in0 && in1
v_add_lshl_u32 v233, v2, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v6, v233, s[34:35]             // LDC clip if OOB. offset
buffer_load_dword v232, v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v233, v6, v233, s[34:35]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+7], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+8], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+9], acc8            // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+10], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+11], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+12], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+13], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+14], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+15], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+16], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+17], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+18], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+19], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+20], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+21], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+22], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+23], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+24], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+25], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+26], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+27], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+28], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+29], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+30], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+31], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+32], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+33], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+34], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+35], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+36], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+37], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+38], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+39], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+40], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+41], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+42], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+43], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+44], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+45], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+46], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+47], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+48], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+49], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+50], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+51], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+52], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+53], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+54], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+55], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+56], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+57], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+58], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+59], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+60], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+61], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+62], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+63], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+64], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+65], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+66], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+67], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+68], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+69], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+70], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 8, 0), (0, 0, 8, 1), (0, 0, 8, 2), (0, 0, 8, 3), (0, 0, 9, 0), (0, 0, 9, 1), (0, 0, 9, 2), (0, 0, 9, 3), (0, 0, 10, 0), (0, 0, 10, 1), (0, 0, 10, 2), (0, 0, 10, 3), (0, 0, 11, 0), (0, 0, 11, 1), (0, 0, 11, 2), (0, 0, 11, 3), (0, 0, 12, 0), (0, 0, 12, 1), (0, 0, 12, 2), (0, 0, 12, 3), (0, 0, 13, 0), (0, 0, 13, 1), (0, 0, 13, 2), (0, 0, 13, 3), (0, 0, 14, 0), (0, 0, 14, 1), (0, 0, 14, 2), (0, 0, 14, 3), (0, 0, 15, 0), (0, 0, 15, 1), (0, 0, 15, 2), (0, 0, 15, 3)] */
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_pk_mul_f32 v[vgprValuC+8:vgprValuC+8+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+8:vgprValuC+8+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+10:vgprValuC+10+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+10:vgprValuC+10+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+20:vgprValuC+20+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+22:vgprValuC+22+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+24:vgprValuC+24+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+26:vgprValuC+26+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+44:vgprValuC+44+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+44:vgprValuC+44+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+46:vgprValuC+46+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+46:vgprValuC+46+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+48:vgprValuC+48+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+48:vgprValuC+48+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+50:vgprValuC+50+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+50:vgprValuC+50+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+52:vgprValuC+52+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+52:vgprValuC+52+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+54:vgprValuC+54+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+54:vgprValuC+54+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+56:vgprValuC+56+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+56:vgprValuC+56+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+58:vgprValuC+58+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+58:vgprValuC+58+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+60:vgprValuC+60+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+60:vgprValuC+60+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+62:vgprValuC+62+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+62:vgprValuC+62+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+64:vgprValuC+64+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+64:vgprValuC+64+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+66:vgprValuC+66+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+66:vgprValuC+66+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+68:vgprValuC+68+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+68:vgprValuC+68+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+7], v71, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v7, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+8], v73, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_dword v8, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+9], v110, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v9, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+10], v112, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v10, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+11], v114, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v11, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+12], v116, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v12, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+13], v118, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v13, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+14], v120, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v14, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+15], v122, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v15, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+16], v124, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v16, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+17], v126, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v17, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+18], v128, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v18, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+19], v130, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v19, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+20], v132, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v20, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+21], v134, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v21, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+22], v136, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v22, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+23], v138, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v23, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+24], v140, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v24, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+25], v142, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v25, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+26], v144, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v26, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+27], v146, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v27, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+28], v148, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v28, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+29], v150, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v29, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+30], v152, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v30, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+31], v154, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v31, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+32], v156, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v32, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+33], v158, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v33, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+34], v160, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v34, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+35], v162, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v35, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+36], v164, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v36, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+37], v166, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v37, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+38], v168, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v38, v169, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+39], v170, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v39, v171, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+40], v172, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v40, v173, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+41], v174, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v41, v175, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+42], v176, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v42, v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+43], v178, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v43, v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+44], v180, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v44, v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+45], v182, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v45, v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+46], v184, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v46, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+47], v186, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v47, v187, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+48], v188, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v48, v189, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+49], v190, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v49, v191, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+50], v192, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v50, v193, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+51], v194, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v51, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+52], v196, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v52, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+53], v198, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v53, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+54], v200, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v54, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+55], v202, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v55, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+56], v204, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v56, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+57], v206, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v57, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+58], v208, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v58, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+59], v210, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v59, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+60], v212, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v60, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+61], v214, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v61, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+62], v216, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v62, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+63], v218, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v63, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+64], v220, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v64, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+65], v222, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v65, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+66], v224, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v66, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+67], v226, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v67, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+68], v228, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v68, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+69], v230, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v69, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+70], v232, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v70, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_End_2:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
