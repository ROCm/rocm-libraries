
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x160x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_5_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
.globl Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x160x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_5_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
.p2align 8
.type Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x160x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_5_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x160x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_5_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 256 // accvgpr offset
  .amdhsa_next_free_vgpr 336 // vgprs
  .amdhsa_next_free_sgpr 88 // sgprs
  .amdhsa_group_segment_fixed_size 152064 // lds bytes
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
/* Num VGPR   =256 */
/* Num AccVGPR=80 */
/* Num SGPR   =88 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 5 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=4 */
/* VectorWidthB=1 */
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
  - 1
amdhsa.kernels:
  - .name: Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x160x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_5_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1
    .symbol: 'Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x160x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_5_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1.kd'
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
      - .name:            AddressWS
        .size:            8
        .offset:          64
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressFlags
        .size:            8
        .offset:          72
        .value_kind:      global_buffer
        .value_type:      f32
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
      - .name:            strideB0
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      f32
      - .name:            ItersPerTile
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberItersPerTile
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicShiftItersPerTile
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SKItersPerWG
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skGrid
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skTiles
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   152064
    .kernarg_segment_align:      8
    .kernarg_segment_size:       144
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 88
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_S_MX_B_UserArgs_MT128x160x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA8_LPB8_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_5_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_SGRO0_TDMI0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1:
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
.set vgprBase, 20
.set vgprGlobalReadOffsetA, 0
.set vgprGlobalReadOffsetB, 8
.set vgprLocalReadAddrA, 18
.set vgprLocalReadAddrB, 19
.set vgprLocalReadSwapAddrA, 164
.set vgprLocalReadSwapAddrB, 165
.set vgprSerial, 200

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+64
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+32
.set vgprValuA_T0_I0, 168
.set vgprValuA_T1_I0, 184
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+40
.set IdentityMatrix, 166

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprArgType, 5
.set sgprStaggerU, 6
.set sgprWGM, 7
.set sgprLoopCounterL, 8
.set sgprOrigLoopCounter, 9
.set sgprSrdD, 12
.set sgprSrdC, 16
.set sgprNumWorkGroups0, 10
.set sgprNumWorkGroups1, 11
.set sgprSizesFree, 20
.set sgprSizesSum, 23
.set sgprAddressD, 24
.set sgprAddressC, 26
.set sgprAddressA, 28
.set sgprAddressB, 30
.set sgprAddressWS, 32
.set sgprAddressFlags, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprItersPerTile, 46
.set sgprMagicNumberItersPerTile, 47
.set sgprMagicShiftItersPerTile, 48
.set sgprSKItersPerWG, 49
.set sgprskGrid, 50
.set sgprskTiles, 51
.set sgprLocalWriteAddrA, 52
.set sgprLocalWriteAddrB, 53
.set sgprSwapA, 54
.set sgprSwapB, 55
.set sgprStreamKIdx, 56
.set sgprStreamKIter, 57
.set sgprStreamKIterEnd, 58
.set sgprStreamKLocalStart, 59
.set sgprStreamKLocalEnd, 60
.set sgprSrdWS, 64

/* StreamK Parallel Reduction Assignments */
.set sgprSkSplit, sgprskTiles+0
.set sgprSkPartialIdx, sgprBeta+0

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
.set MT1, 160
.set DepthU, 64
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
s_load_dword s16, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_dword s18, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_dword s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_dword s19, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s17, s16, 0x1e                          // Get arg type
s_and_b32 s16, 0x3fffffff, s16                     // Get nums of gemm
s_cmp_eq_u32 s17, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s17, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_dwordx16 s[36:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_waitcnt lgkmcnt(0)                               // preload
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_branch label_common_kernel_entry

/* pad 37 snops to satisfy 0x100 code size for Preload Backward Compatibility Prologue */
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
s_nop 0
s_nop 0
label_Preload_Offset_Start:
s_and_b32 s16, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s17, s2, 0x1e                           // Get arg type
s_mov_b32 s18, s3                                  // Preload internal args
s_cmp_eq_u32 s17, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance2
s_cmp_eq_u32 s17, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance2:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dword s27, s[sgprKernArgAddress:sgprKernArgAddress+1], 28 // 28
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 32 // 32
s_load_dwordx8 s[44:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_mov_b64 s[20:21], s[6:7]                         // move preload data to correct sgpr
s_mov_b64 s[22:23], s[8:9]                         // move preload data to correct sgpr
s_mov_b64 s[24:25], s[10:11]                       // move preload data to correct sgpr
s_mov_b32 s26, s12                                 // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
s_mov_b32 s19, s5                                  // Load num of WGs
label_common_kernel_entry:  /// for both preload/non-preload common code
s_mov_b32 s[sgprWorkGroup0+0], s13                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+1], s14                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+2], s15                 // restore workgroup id
s_and_b32 s[sgprStaggerU], s18, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_mov_b32 s[sgprArgType], s17
s_mov_b32 m0, 0x25200                              // LDS clamp at 152064 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* remap workgroup to XCCs */
s_lshr_b32 s66, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s66, s66                             // Get log(WGMXCC)
s_lshr_b32 s67, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s66, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s63, s19, s66
s_lshl_b32 s63, s63, s66
s_cmp_ge_u32 s[sgprWorkGroup0], s63
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s67, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s63, s[sgprWorkGroup0], s66
s_bfm_b32 s64, s66, 0
s_and_b32 s64, s[sgprWorkGroup0], s64
s_lshr_b32 s65, s19, s66
s_mul_i32 s64, s64, s65
s_add_u32 s[sgprWorkGroup0], s63, s64
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[20:21], s67                        // s63 = s[sgprWorkGroup0] / s67
v_rcp_f64 v[20:21], v[20:21]                       // s63 = s[sgprWorkGroup0] / s67
v_cvt_f64_u32 v[22:23], s[sgprWorkGroup0]          // s63 = s[sgprWorkGroup0] / s67
v_mul_f64 v[20:21], v[20:21], v[22:23]             // s63 = s[sgprWorkGroup0] / s67
v_cvt_u32_f64 v20, v[20:21]                        // s63 = s[sgprWorkGroup0] / s67
v_mul_lo_u32 v21, v20, s67                         // s63 = s[sgprWorkGroup0] / s67
v_sub_u32 v22, s[sgprWorkGroup0], v21              // s63 = s[sgprWorkGroup0] / s67
v_cmpx_ge_u32 exec, v22, s67                       // s63 = s[sgprWorkGroup0] / s67
v_add_u32 v20, v20, 1                              // s63 = s[sgprWorkGroup0] / s67
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v21, v20, s67                         // s63 = s[sgprWorkGroup0] / s67
v_sub_u32 v22, s[sgprWorkGroup0], v21              // s63 = s[sgprWorkGroup0] / s67
v_readfirstlane_b32 s63, v20                       // quotient
v_readfirstlane_b32 s64, v22                       // remainder
s_mul_i32 s63, s63, s67
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s64, s64, s66
/* temp0 = temp0 + temp1 */
s_add_u32 s63, s63, s64
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[20:21], s67                        // s64 = s19 / s67
v_rcp_f64 v[20:21], v[20:21]                       // s64 = s19 / s67
v_cvt_f64_u32 v[22:23], s19                        // s64 = s19 / s67
v_mul_f64 v[20:21], v[20:21], v[22:23]             // s64 = s19 / s67
v_cvt_u32_f64 v20, v[20:21]                        // s64 = s19 / s67
v_mul_lo_u32 v21, v20, s67                         // s64 = s19 / s67
v_sub_u32 v22, s19, v21                            // s64 = s19 / s67
v_cmpx_ge_u32 exec, v22, s67                       // s64 = s19 / s67
v_add_u32 v20, v20, 1                              // s64 = s19 / s67
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s64, v20                       // quotient
s_mul_i32 s64, s64, s67
s_sub_u32 s65, s19, s64
s_cmp_gt_u32 s[sgprWorkGroup0], s64
s_cselect_b32 s64, s65, s67
s_lshr_b32 s64, s64, s66
s_bfm_b32 s65, s66, 0
s_and_b32 s65, s[sgprWorkGroup0], s65
s_mul_i32 s64, s64, s65
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s63, s64
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s17, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s17, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [20...184) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...80) to pool */
v_mov_b32 v22, MT0                                 // set MT0 into sgpr
v_mov_b32 v21, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v20, v22                             // v20 = ceil(v21 / v22)
v_rcp_iflag_f32 v20, v20                           // v20 = ceil(v21 / v22)
v_cvt_f32_u32 v23, v21                             // v20 = ceil(v21 / v22)
v_mul_f32 v20, v20, v23                            // v20 = ceil(v21 / v22)
v_cvt_u32_f32 v20, v20                             // v20 = ceil(v21 / v22)
v_mul_u32_u24 v23, v20, v22                        // v20 = ceil(v21 / v22)
v_sub_u32 v23, v21, v23                            // v20 = ceil(v21 / v22)
v_cmp_ne_u32 vcc, v23, 0                           // v20 = ceil(v21 / v22)
v_addc_co_u32 v20, vcc, v20, 0, vcc                // ceil
v_mov_b32 v22, MT1                                 // set MT1 into sgpr
v_mov_b32 v21, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v20     // set back to numWorkGroup0
v_cvt_f32_u32 v20, v22                             // v20 = ceil(v21 / v22)
v_rcp_iflag_f32 v20, v20                           // v20 = ceil(v21 / v22)
v_cvt_f32_u32 v23, v21                             // v20 = ceil(v21 / v22)
v_mul_f32 v20, v20, v23                            // v20 = ceil(v21 / v22)
v_cvt_u32_f32 v20, v20                             // v20 = ceil(v21 / v22)
v_mul_u32_u24 v23, v20, v22                        // v20 = ceil(v21 / v22)
v_sub_u32 v23, v21, v23                            // v20 = ceil(v21 / v22)
v_cmp_ne_u32 vcc, v23, 0                           // v20 = ceil(v21 / v22)
v_addc_co_u32 v20, vcc, v20, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v20     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 84/0 bytes of kern args
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s11, 128                                 // KernArgAddressOffset
s_mul_i32 s68, s16, 4
s_mov_b64 s[62:63], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s11, 220
s_mov_b32 s68, 0
s_mov_b64 s[62:63], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s10, 1
s_mov_b32 s69, 0
s_load_dwordx4 s[20:23], s[62:63], s68
s_cmpk_eq_u32 s16, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s66, s20, 7                             // s66 = s20 / 128
s_and_b32 s64, 127, s20                            // s64 = s20 % 128
s_addc_u32 s66, s66, 0
s_mov_b32 s65, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s64, 819, s21                            // tmp1 = dividend * magic hi
s_lshl_b64 s[64:65], s[64:65], 16                  // left shift 16 bits
s_mul_i32 s67, s21, 13108                          // tmp0 = dividend * magic lo
s_add_u32 s64, s67, s64                            // add lo
s_addc_u32 s65, s65, 0                             // add hi
s_lshr_b64 s[64:65], s[64:65], 33                  // tmp0 = quotient
s_mul_i32 s65, s64, 160                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s65, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s67, s64, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s66, s66, s67
s_mul_i32 s66, s66, s22
s_add_u32 s69, s69, s66
s_cmp_lt_u32 s[sgprWorkGroup0], s69
s_cbranch_scc1 label_FOUND
s_add_u32 s68, s68, s11
s_load_dwordx4 s[20:23], s[62:63], s68
s_add_u32 s10, s10, 1
s_cmp_lt_u32 s10, s16
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s66, s20, 7                             // s66 = s20 / 128
s_and_b32 s64, 127, s20                            // s64 = s20 % 128
s_addc_u32 s66, s66, 0
s_mov_b32 s65, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s64, 819, s21                            // tmp1 = dividend * magic hi
s_lshl_b64 s[64:65], s[64:65], 16                  // left shift 16 bits
s_mul_i32 s67, s21, 13108                          // tmp0 = dividend * magic lo
s_add_u32 s64, s67, s64                            // add lo
s_addc_u32 s65, s65, 0                             // add hi
s_lshr_b64 s[64:65], s[64:65], 33                  // tmp0 = quotient
s_mul_i32 s65, s64, 160                            // tmp1 = quotient * divisor
s_cmp_lg_u32 s65, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s67, s64, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s66, s66, s67
s_mul_i32 s66, s66, s22
s_add_u32 s69, s69, s66

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s63, s10, 1
s_sub_u32 s62, s69, s66
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s62
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s16, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s63, s63, 128                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s63
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx8 s[40:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_dwordx4 s[48:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s63, s63, 220
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s63
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx8 s[40:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_dwordx2 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1], 112 // 112
s_load_dword s50, s[sgprKernArgAddress:sgprKernArgAddress+1], 120 // 120
// Read Beta
s_load_dword s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 136 // 136
label_LoadExternalStructEnd:
/* init: add vgpr [20...184) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...80) to pool */
v_mov_b32 v22, MT0                                 // set MT0 into sgpr
v_mov_b32 v21, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v20, v22                             // v20 = ceil(v21 / v22)
v_rcp_iflag_f32 v20, v20                           // v20 = ceil(v21 / v22)
v_cvt_f32_u32 v23, v21                             // v20 = ceil(v21 / v22)
v_mul_f32 v20, v20, v23                            // v20 = ceil(v21 / v22)
v_cvt_u32_f32 v20, v20                             // v20 = ceil(v21 / v22)
v_mul_u32_u24 v23, v20, v22                        // v20 = ceil(v21 / v22)
v_sub_u32 v23, v21, v23                            // v20 = ceil(v21 / v22)
v_cmp_ne_u32 vcc, v23, 0                           // v20 = ceil(v21 / v22)
v_addc_co_u32 v20, vcc, v20, 0, vcc                // ceil
v_mov_b32 v22, MT1                                 // set MT1 into sgpr
v_mov_b32 v21, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v20     // set back to numWorkGroup0
v_cvt_f32_u32 v20, v22                             // v20 = ceil(v21 / v22)
v_rcp_iflag_f32 v20, v20                           // v20 = ceil(v21 / v22)
v_cvt_f32_u32 v23, v21                             // v20 = ceil(v21 / v22)
v_mul_f32 v20, v20, v23                            // v20 = ceil(v21 / v22)
v_cvt_u32_f32 v20, v20                             // v20 = ceil(v21 / v22)
v_mul_u32_u24 v23, v20, v22                        // v20 = ceil(v21 / v22)
v_sub_u32 v23, v21, v23                            // v20 = ceil(v21 / v22)
v_cmp_ne_u32 vcc, v23, 0                           // v20 = ceil(v21 / v22)
v_addc_co_u32 v20, vcc, v20, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v20     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 84/0 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

label_MultiGemmEnd:
.set sgprSrdA, 68
.set sgprSrdB, 72
.set sgprShadowLimitA, 62
.set sgprShadowLimitB, 76
.set sgprStaggerUIter, 61
.set sgprWrapUA, 78
.set sgprWrapUB, 80
.set sgprGlobalReadIncsA, 82
.set sgprGlobalReadIncsB, 83
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
s_mov_b32 s[sgprStreamKIdx], s[sgprWorkGroup0]     // Save original StreamK index
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitInit                  // Jump to single kernel init
v_cvt_f32_u32 v20, s[sgprSkSplit]                  // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_rcp_iflag_f32 v20, v20                           // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cvt_f32_u32 v21, s[sgprStreamKIdx]               // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_f32 v20, v20, v21                            // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cvt_u32_f32 v20, v20                             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_u32_u24 v21, v20, s[sgprSkSplit]             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_sub_u32 v21, s[sgprStreamKIdx], v21              // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cmpx_eq_u32 exec, v21, s[sgprSkSplit]            // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_add_u32 v20, 1, v20                              // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mov_b32 v21, 0                                   // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v21, s[sgprSkSplit]            // overflow happened in remainder
v_sub_u32 v20, v20, 1                              // quotient - 1
v_mul_u32_u24 v21, v20, s[sgprSkSplit]             // re-calculate remainder
v_sub_u32 v21, s[sgprStreamKIdx], v21              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s12, v20                       // quotient
v_readfirstlane_b32 s13, v21                       // remainder
s_mul_i32 s14, s[sgprSkSplit], s[sgprSKItersPerWG]
s_sub_u32 s14, s[sgprItersPerTile], s14            // extraIters = itersPerTile - SkSplit * skItersPerWG
s_mul_i32 s[sgprStreamKIter], s13, s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_cmp_lt_u32 s13, s14                              // Check if WG gets an extra iteration
s_cbranch_scc1 label_SK_HasExtra                   // Has extra iter
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s14 // This WG does not have an extra iteration
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_branch label_SK_DoneExtra                        // Done init for parallel reduction
label_SK_HasExtra:
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s13 // This WG has an extra iteration
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], 1 // StreamK ending iteration (case: after extra iters)
label_SK_DoneExtra:
s_mul_i32 s12, s12, s[sgprItersPerTile]            // Tile offset = tilesIdx * itersPerTile
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s12 // Offset to correct tile
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s12 // Offset to correct tile
s_mov_b32 s[sgprSkPartialIdx], s13                 // Save partial idx for SrdD calculation
s_branch label_SK_InitDone                         // Done init for parallel reduction
label_SK_SplitInit:
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprItersPerTile] // DP starting iteration (case: DP work to do)
s_mul_i32 s12, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s12, s12, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s12, s12, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_mov_b32 s[sgprStreamKIterEnd], s12               // DP ending iteration (case: only DP work to do)
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_cmp_lt_u32 s12, s[sgprStreamKIterEnd]            // Check if there are DP tiles to do
s_cbranch_scc1 label_SK_InitDone                   // Done init
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s13, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s12, s12, s13                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s12 // Add extra iters
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s14, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s13, s[sgprStreamKIdx], s14              // StreamK starting iteration (case: before extra iters)
s_add_u32 s14, s13, s14                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprStreamKIdx], s12                // Check if lane gets an extra iteration
s_cselect_b32 s[sgprStreamKIter], s13, s[sgprStreamKIter] // Set start iter
s_cselect_b32 s[sgprStreamKIterEnd], s14, s[sgprStreamKIterEnd] // Set end iter
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_min_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s12 // Cap ending iter at total SK iters
label_SK_InitDone:
s_mul_i32 s12, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s12, s12, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s12, s12, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_cmp_lt_u32 s[sgprStreamKIter], s12               // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc0
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s14, label_KernelEnd, 4                  // target branch offset
s_add_u32 s12, s12, s14                            // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_setpc_b64 s[12:13]                               // branch to label_KernelEnd
label_NoBranch_T8JHFHKM7BO5OHXW:
/* Create a negative identity matrix used by TF32 MFMA emulation. */
v_and_b32 v21, 3, v[vgprSerial]                    // lane % 4
v_mov_b64 v[166:167], 0
v_mov_b32 v20, 0xbf80
v_cmp_eq_u32 vcc, 0, v21                           // Lane %4 == 0 ?
s_nop 1
v_cndmask_b32 v166, v166, v20, vcc
v_cmp_eq_u32 vcc, 2, v21                           // Lane %4 == 2 ?
s_nop 1
v_cndmask_b32 v167, v167, v20, vcc
v_mov_b32 v20, 0xbf800000
v_cmp_eq_u32 vcc, 1, v21                           // Lane %4 == 1 ?
s_nop 1
v_cndmask_b32 v166, v166, v20, vcc
v_cmp_eq_u32 vcc, 3, v21                           // Lane %4 == 3 ?
s_nop 1
v_cndmask_b32 v167, v167, v20, vcc

/******************************************/
/* Persistent Loop Start                  */
/******************************************/
label_PersistentLoopStart:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */

/* localReadResetOffsets */
/* handled internally */
v_xor_b32 v20, v[vgprLocalReadSwapAddrA], v[vgprLocalReadAddrA] // Get other lds buffer offset value
v_min_i32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA], v20 // Set LRA to first buffer offset

/* localReadResetOffsets */
/* handled internally */
v_xor_b32 v20, v[vgprLocalReadSwapAddrB], v[vgprLocalReadAddrB] // Get other lds buffer offset value
v_min_i32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB], v20 // Set LRA to first buffer offset
/* StreamK calculate tile idx and map to WG */
s_mul_hi_u32 s13, s[sgprStreamKIter], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s14, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s12, s[sgprStreamKIter], s14             // s_magic mul, div alg 2
s_add_u32 s12, s12, s13
s_and_b32 s14, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s12, s12, s14                           // sMagicDiv Alg 2
s_mul_i32 s13, s12, s[sgprItersPerTile]            // Tile start iteration
s_add_u32 s14, s13, s[sgprItersPerTile]            // Tile end iteration
s_sub_u32 s[sgprStreamKLocalStart], s[sgprStreamKIter], s13 // Local iteration start
s_min_u32 s[sgprStreamKLocalEnd], s[sgprStreamKIterEnd], s14 // 1. (Local) iteration end (SK tile)
s_sub_u32 s[sgprStreamKLocalEnd], s[sgprStreamKLocalEnd], s13 // 2. Local iteration end (SK tile)
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitUpdate                // Jump to single kernel update
s_mov_b32 s13, s[sgprStreamKIterEnd]               // Parallel reduction, work contained to single partial tile
s_branch label_SK_UpdateDone                       // Done update for parallel reduction
label_SK_SplitUpdate:
s_mul_i32 s15, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s15, s15, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_sub_u32 s15, s15, s[sgprskTiles]                 // dpTiles = totalTiles - skTiles
s_mul_i32 s15, s15, s[sgprItersPerTile]            // dpSectionSize = dpTiles * ItersPerTile
s_mul_i32 s13, s[sgprskGrid], s[sgprItersPerTile]  // DP iterations shift
s_add_u32 s13, s13, s[sgprStreamKIter]             // Add DP shift
s_cmp_lt_u32 s13, s15                              // Check if still in DP section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s13, s14                                 // SK iterations shift
s_cmp_le_u32 s15, s[sgprStreamKIter]               // Check if continuing in SK section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mul_i32 s16, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s17, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s16, s16, s17                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_mul_i32 s[sgprStreamKIter], s[sgprStreamKIdx], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprStreamKIter], s[sgprStreamKIter], s16 // Add extra iters
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIter], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s18, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s17, s[sgprStreamKIdx], s18              // StreamK starting iteration (case: before extra iters)
s_add_u32 s18, s17, s18                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprStreamKIdx], s16                // Check if lane gets an extra iteration
s_cselect_b32 s[sgprStreamKIter], s17, s[sgprStreamKIter] // Set start iter
s_cselect_b32 s[sgprStreamKIterEnd], s18, s[sgprStreamKIterEnd] // Set end iter
s_add_u32 s13, s[sgprStreamKIter], s15             // Offset to start of SK section
s_add_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s15 // Offset to start of SK section
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s16, s16, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s16, s16, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_min_u32 s[sgprStreamKIterEnd], s[sgprStreamKIterEnd], s16 // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprStreamKIter], s16               // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_S4FDBQ587JJL6NOU     // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_KernelEnd, 4                  // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_KernelEnd
label_NoBranch_S4FDBQ587JJL6NOU:
label_SK_UpdateDone:
s_mov_b32 s[sgprStreamKIter], s13                  // Store current iteration
/* Map StreamK tile index to wg0/1/2 */
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // Total tiles
v_cvt_f32_u32 v20, s13                             // TileID // nWG0*nWG1
v_rcp_iflag_f32 v20, v20                           // TileID // nWG0*nWG1
v_cvt_f32_u32 v21, s12                             // TileID // nWG0*nWG1
v_mul_f32 v20, v20, v21                            // TileID // nWG0*nWG1
v_cvt_u32_f32 v20, v20                             // TileID // nWG0*nWG1
v_mul_u32_u24 v21, v20, s13                        // TileID // nWG0*nWG1
v_sub_u32 v21, s12, v21                            // TileID // nWG0*nWG1
v_cmpx_eq_u32 exec, v21, s13                       // TileID // nWG0*nWG1
v_add_u32 v20, 1, v20                              // TileID // nWG0*nWG1
v_mov_b32 v21, 0                                   // TileID // nWG0*nWG1
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v21, s13                       // overflow happened in remainder
v_sub_u32 v20, v20, 1                              // quotient - 1
v_mul_u32_u24 v21, v20, s13                        // re-calculate remainder
v_sub_u32 v21, s12, v21                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup2], v20         // quotient
v_readfirstlane_b32 s14, v21                       // remainder
v_cvt_f32_u32 v20, s[sgprNumWorkGroups0]           // TileID // nWG0
v_rcp_iflag_f32 v20, v20                           // TileID // nWG0
v_cvt_f32_u32 v21, s14                             // TileID // nWG0
v_mul_f32 v20, v20, v21                            // TileID // nWG0
v_cvt_u32_f32 v20, v20                             // TileID // nWG0
v_mul_u32_u24 v21, v20, s[sgprNumWorkGroups0]      // TileID // nWG0
v_sub_u32 v21, s14, v21                            // TileID // nWG0
v_cmpx_eq_u32 exec, v21, s[sgprNumWorkGroups0]     // TileID // nWG0
v_add_u32 v20, 1, v20                              // TileID // nWG0
v_mov_b32 v21, 0                                   // TileID // nWG0
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v21, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v20, v20, 1                              // quotient - 1
v_mul_u32_u24 v21, v20, s[sgprNumWorkGroups0]      // re-calculate remainder
v_sub_u32 v21, s14, v21                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v20         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v21         // remainder

v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck                  // branch if s[Alpha] != 0
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_UR8VN3A1SJCPC6PO     // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_SK_CloseLoop, 4               // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_SK_CloseLoop
label_NoBranch_UR8VN3A1SJCPC6PO:
s_mov_b32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Skip iterations
label_SKAlphaCheck:
/* WGM Calculation */
s_mov_b32 s12, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s12, s12                            // Restore WGM
s_cmp_gt_i32 s12, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s12, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s12, s12                                 // abs(WGM)
v_cvt_f64_u32 v[20:21], s12                        // s13 = s[sgprWorkGroup0] / s12
v_rcp_f64 v[20:21], v[20:21]                       // s13 = s[sgprWorkGroup0] / s12
v_cvt_f64_u32 v[22:23], s[sgprWorkGroup0]          // s13 = s[sgprWorkGroup0] / s12
v_mul_f64 v[20:21], v[20:21], v[22:23]             // s13 = s[sgprWorkGroup0] / s12
v_cvt_u32_f64 v20, v[20:21]                        // s13 = s[sgprWorkGroup0] / s12
v_mul_lo_u32 v21, v20, s12                         // s13 = s[sgprWorkGroup0] / s12
v_sub_u32 v22, s[sgprWorkGroup0], v21              // s13 = s[sgprWorkGroup0] / s12
v_cmpx_ge_u32 exec, v22, s12                       // s13 = s[sgprWorkGroup0] / s12
v_add_u32 v20, v20, 1                              // s13 = s[sgprWorkGroup0] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v20                       // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup0], s16              // WorkGroup0=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s16, s16, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[20:21], s12                        // s14 = s[sgprNumWorkGroups0] / s12
v_rcp_f64 v[20:21], v[20:21]                       // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_f64_u32 v[22:23], s[sgprNumWorkGroups0]      // s14 = s[sgprNumWorkGroups0] / s12
v_mul_f64 v[20:21], v[20:21], v[22:23]             // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_u32_f64 v20, v[20:21]                        // s14 = s[sgprNumWorkGroups0] / s12
v_mul_lo_u32 v21, v20, s12                         // s14 = s[sgprNumWorkGroups0] / s12
v_sub_u32 v22, s[sgprNumWorkGroups0], v21          // s14 = s[sgprNumWorkGroups0] / s12
v_cmpx_ge_u32 exec, v22, s12                       // s14 = s[sgprNumWorkGroups0] / s12
v_add_u32 v20, v20, 1                              // s14 = s[sgprNumWorkGroups0] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s14, v20                       // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups0], s15          // NumWorkGroups0=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[20:21], s14                        // s[sgprWorkGroup1] = s16 / s14
v_rcp_f64 v[20:21], v[20:21]                       // s[sgprWorkGroup1] = s16 / s14
v_cvt_f64_u32 v[22:23], s16                        // s[sgprWorkGroup1] = s16 / s14
v_mul_f64 v[20:21], v[20:21], v[22:23]             // s[sgprWorkGroup1] = s16 / s14
v_cvt_u32_f64 v20, v[20:21]                        // s[sgprWorkGroup1] = s16 / s14
v_mul_lo_u32 v21, v20, s14                         // s[sgprWorkGroup1] = s16 / s14
v_sub_u32 v22, s16, v21                            // s[sgprWorkGroup1] = s16 / s14
v_cmpx_ge_u32 exec, v22, s14                       // s[sgprWorkGroup1] = s16 / s14
v_add_u32 v20, v20, 1                              // s[sgprWorkGroup1] = s16 / s14
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v21, v20, s14                         // s[sgprWorkGroup1] = s16 / s14
v_sub_u32 v22, s16, v21                            // s[sgprWorkGroup1] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup1], v20         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v22         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s14 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s16, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s13, s13, s12                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s13 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s12, s12                                 // WGM
v_cvt_f64_u32 v[20:21], s12                        // s13 = s[sgprWorkGroup1] / s12
v_rcp_f64 v[20:21], v[20:21]                       // s13 = s[sgprWorkGroup1] / s12
v_cvt_f64_u32 v[22:23], s[sgprWorkGroup1]          // s13 = s[sgprWorkGroup1] / s12
v_mul_f64 v[20:21], v[20:21], v[22:23]             // s13 = s[sgprWorkGroup1] / s12
v_cvt_u32_f64 v20, v[20:21]                        // s13 = s[sgprWorkGroup1] / s12
v_mul_lo_u32 v21, v20, s12                         // s13 = s[sgprWorkGroup1] / s12
v_sub_u32 v22, s[sgprWorkGroup1], v21              // s13 = s[sgprWorkGroup1] / s12
v_cmpx_ge_u32 exec, v22, s12                       // s13 = s[sgprWorkGroup1] / s12
v_add_u32 v20, v20, 1                              // s13 = s[sgprWorkGroup1] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v20                       // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup1], s16              // WorkGroup1=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s16, s16, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[20:21], s12                        // s14 = s[sgprNumWorkGroups1] / s12
v_rcp_f64 v[20:21], v[20:21]                       // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_f64_u32 v[22:23], s[sgprNumWorkGroups1]      // s14 = s[sgprNumWorkGroups1] / s12
v_mul_f64 v[20:21], v[20:21], v[22:23]             // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_u32_f64 v20, v[20:21]                        // s14 = s[sgprNumWorkGroups1] / s12
v_mul_lo_u32 v21, v20, s12                         // s14 = s[sgprNumWorkGroups1] / s12
v_sub_u32 v22, s[sgprNumWorkGroups1], v21          // s14 = s[sgprNumWorkGroups1] / s12
v_cmpx_ge_u32 exec, v22, s12                       // s14 = s[sgprNumWorkGroups1] / s12
v_add_u32 v20, v20, 1                              // s14 = s[sgprNumWorkGroups1] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s14, v20                       // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups1], s15          // NumWorkGroups1=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[20:21], s14                        // s[sgprWorkGroup0] = s16 / s14
v_rcp_f64 v[20:21], v[20:21]                       // s[sgprWorkGroup0] = s16 / s14
v_cvt_f64_u32 v[22:23], s16                        // s[sgprWorkGroup0] = s16 / s14
v_mul_f64 v[20:21], v[20:21], v[22:23]             // s[sgprWorkGroup0] = s16 / s14
v_cvt_u32_f64 v20, v[20:21]                        // s[sgprWorkGroup0] = s16 / s14
v_mul_lo_u32 v21, v20, s14                         // s[sgprWorkGroup0] = s16 / s14
v_sub_u32 v22, s16, v21                            // s[sgprWorkGroup0] = s16 / s14
v_cmpx_ge_u32 exec, v22, s14                       // s[sgprWorkGroup0] = s16 / s14
v_add_u32 v20, v20, 1                              // s[sgprWorkGroup0] = s16 / s14
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v21, v20, s14                         // s[sgprWorkGroup0] = s16 / s14
v_sub_u32 v22, s16, v21                            // s[sgprWorkGroup0] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup0], v20         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v22         // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s14 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s16, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s13, s13, s12                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s13 // wg1 += blockId * WGM
label_WGM:

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v21, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v20, 15, v21                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v20, 6, v20                          // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v20, 2, v20                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v21, 4, v21                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v20, v21, 2, v20                    // 5. K offset: lrKOffset = kIdx * mStride(4); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v24, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v24, 1, v24                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v20, v24, 12, v20                   // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v22, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v21, 15, v22                             // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
/* Computing strided(2) perp indicies */
v_and_b32 v27, 7, v21                              // r0 = I % (16 // 2)
v_lshlrev_b32 v27, 1, v27                          // r0 = 2 * r0
/* Computing r1 = (I % 16) // (16 // 2) */
v_and_b32 v28, 15, v21                             // r1 = I % (16)
v_lshrrev_b32 v28, 3, v28                          // r1 = (r1) // (16 // 2)
v_add_u32 v27, v27, v28                            // r0 = r0 + r1
v_lshrrev_b32 v28, 4, v21                          // r1 = I // 16
v_lshl_add_u32 v21, v28, 4, v27                    // v21 = v28 * 16
/* Done computing strided(2) perp indices */
v_lshlrev_b32 v21, 6, v21                          // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshrrev_b32 v22, 4, v22                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v21, v22, 2, v21                    // 5. K offset: lrKOffset = kIdx * mStride(4); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v23, 7, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v23, 1, v23                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v21, v23, 10, v21                   // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v22, 6, v[vgprSerial]                // 22 = Serial / 64
v_lshrrev_b32 v22, 2, v22                          // LSU offset: Get LSU wave_id
s_mov_b32 s12, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v22, s12, v22                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v22, v20          // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 2, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v23, 10, v[vgprLocalReadAddrA]       // Final Offset: padding 32 per block 1024
v_lshl_add_u32 v[vgprLocalReadAddrA], v23, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 1024

/* local read addresses: final offsets b */
v_lshrrev_b32 v20, 6, v[vgprSerial]                // 20 = Serial / 64
v_lshrrev_b32 v20, 2, v20                          // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v20, s12, v20                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v20, v21          // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 2, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v22, 10, v[vgprLocalReadAddrB]       // Final Offset: padding 32 per block 1024
v_lshl_add_u32 v[vgprLocalReadAddrB], v22, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 1024

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x8400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)
v_add_u32 v[vgprLocalReadSwapAddrA], 76032, v[vgprLocalReadAddrA] // Calculate starting lds addr of second buffer
v_xor_b32 v[vgprLocalReadSwapAddrA], v[vgprLocalReadSwapAddrA], v[vgprLocalReadAddrA] // xor both lds buffer offsets to enable swapping
v_add_u32 v[vgprLocalReadSwapAddrB], 76032, v[vgprLocalReadAddrB] // Calculate starting lds addr of second buffer
v_xor_b32 v[vgprLocalReadSwapAddrB], v[vgprLocalReadSwapAddrB], v[vgprLocalReadAddrB] // xor both lds buffer offsets to enable swapping

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v21 = A-unroll = serial%LVCA */
v_lshrrev_b32 v20, 4, v[vgprSerial]                // 20 = Serial / 16
v_and_b32 v21, 15, v[vgprSerial]                   // 21 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v21, 2, v21                          // v21 = v21 * 4
v_mov_b32 v24, v21                                 // copy for GlobalSplitU
/* LVCB = 16 */
/* v23 = B-unroll = serial%LVCB */
v_lshrrev_b32 v22, 4, v[vgprSerial]                // 22 = Serial / 16
v_and_b32 v23, 15, v[vgprSerial]                   // 23 = Serial % 16
/* unroll *= glvw */
v_lshlrev_b32 v23, 2, v23                          // v23 = v23 * 4
v_mov_b32 v25, v23                                 // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v24 */
/* lwaUnrollAssignmentB = v25 */

/* local write addresses: first offset a */
v_mul_u32_u24 v26, 0x40, v20                       // lwAL**(DepthU_Compute + PAD)
v_add_u32 v26, v24, v26                            // lwFOA = (lwAA + lwAL*(DepthU+PAD))
v_lshlrev_b32 v26, 2, v26                          //  (multiple bpe)
v_lshrrev_b32 v28, 10, v26                         // padding 32 per block 1024
v_lshl_add_u32 v26, v28, 5, v26                    // padding 32 per block 1024
v_lshrrev_b32 v27, 6, v[vgprSerial]                // Compute waveID
s_nop 0                                            // 1 wait states required before reading vgpr by lane
v_readfirstlane_b32 s[sgprLocalWriteAddrA], v27    // Copy lds write address VGPR to SGPR
s_mul_i32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], 1056
s_nop 0                                            // 1 wait states
s_add_u32 s[sgprSwapA], s[sgprLocalWriteAddrA], 76032 // Calculate starting lds addr of second buffer
s_xor_b32 s[sgprSwapA], s[sgprSwapA], s[sgprLocalWriteAddrA] // xor both lds buffer offsets to enable swapping

/* local write addresses: first offset b */
v_mul_u32_u24 v26, 0x40, v22                       // lwBL**(DepthU_Compute + PAD)
v_add_u32 v26, v25, v26                            // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v26, 2, v26                          //  (multiple bpe)
v_lshrrev_b32 v28, 10, v26                         // padding 32 per block 1024
v_lshl_add_u32 v26, v28, 5, v26                    // padding 32 per block 1024
v_add_co_u32 v26, vcc, 0x8400, v26                 // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=33792
v_lshrrev_b32 v27, 6, v[vgprSerial]                // Compute waveID
s_nop 0                                            // 1 wait states required before reading vgpr by lane
v_readfirstlane_b32 s[sgprLocalWriteAddrB], v27    // Copy lds write address VGPR to SGPR
s_mul_i32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], 1056
s_add_u32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], 33792
s_nop 0                                            // 1 wait states
s_add_u32 s[sgprSwapB], s[sgprLocalWriteAddrB], 76032 // Calculate starting lds addr of second buffer
s_xor_b32 s[sgprSwapB], s[sgprSwapB], s[sgprLocalWriteAddrB] // xor both lds buffer offsets to enable swapping

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v20 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v22 */

/* global read addresses: unroll assignment a */
/* v21 */

/* global read addresses: unroll assignment b */
/* v23 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */
v_mov_b32 v26, v20                                 // groA0I_0
v_add_co_u32 v27, vcc, 16, v26                     // groA0I_1 += LSPA
v_add_co_u32 v28, vcc, 16, v27                     // groA0I_2 += LSPA
v_add_co_u32 v29, vcc, 16, v28                     // groA0I_3 += LSPA
v_add_co_u32 v30, vcc, 16, v29                     // groA0I_4 += LSPA
v_add_co_u32 v31, vcc, 16, v30                     // groA0I_5 += LSPA
v_add_co_u32 v32, vcc, 16, v31                     // groA0I_6 += LSPA
v_add_co_u32 v33, vcc, 16, v32                     // groA0I_7 += LSPA

/* global read addresses: tile offsets b */
v_mov_b32 v34, v22                                 // groB1J_0
v_add_co_u32 v35, vcc, 16, v34                     // groB1J_1 += LSPB
v_add_co_u32 v36, vcc, 16, v35                     // groB1J_2 += LSPB
v_add_co_u32 v37, vcc, 16, v36                     // groB1J_3 += LSPB
v_add_co_u32 v38, vcc, 16, v37                     // groB1J_4 += LSPB
v_add_co_u32 v39, vcc, 16, v38                     // groB1J_5 += LSPB
v_add_co_u32 v40, vcc, 16, v39                     // groB1J_6 += LSPB
v_add_co_u32 v41, vcc, 16, v40                     // groB1J_7 += LSPB
v_add_co_u32 v42, vcc, 16, v41                     // groB1J_8 += LSPB
v_add_co_u32 v43, vcc, 16, v42                     // groB1J_9 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v44, v21                                 // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v45, v23                                 // groBL_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s15, s14, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], 64        // StreamK tile start offset
s_mul_hi_u32 s13, s12, constStrideAL               // StreamK tile start offset
s_mul_i32 s12, s12, constStrideAL                  // StreamK tile start offset
s_add_u32 s14, s14, s12                            // accum GsuOffset term to tilestart
s_addc_u32 s15, s15, s13                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s12, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s13, constStrideAL, s12               // stride x (size-1)
s_mul_i32 s12, constStrideAL, s12                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s12, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s13, s[sgprStrideA0I], s12            // stride x (size-1)
s_mul_i32 s12, s[sgprStrideA0I], s12               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s14 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s15 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 2 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s12, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_add_u32 s12, s12, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s13, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdA:sgprSrdA+1], s[12:13], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[14:15], s[14:15], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s14, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s15, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s13, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s15   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup1], 160           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup1], 160              // WorkGroup[01] * MT
s_mul_hi_u32 s15, s14, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s14, s14, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_i32 s12, s[sgprStreamKLocalStart], 64        // StreamK tile start offset
s_mul_hi_u32 s13, s12, constStrideBL               // StreamK tile start offset
s_mul_i32 s12, s12, constStrideBL                  // StreamK tile start offset
s_add_u32 s14, s14, s12                            // accum GsuOffset term to tilestart
s_addc_u32 s15, s15, s13                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s12, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s13, constStrideBL, s12               // stride x (size-1)
s_mul_i32 s12, constStrideBL, s12                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // sum tensor size
s_sub_u32 s12, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s13, s[sgprStrideB1J], s12            // stride x (size-1)
s_mul_i32 s12, s[sgprStrideB1J], s12               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s14 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s15 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 2 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s12, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_add_u32 s12, s12, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s13, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdB:sgprSrdB+1], s[12:13], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[14:15], s[14:15], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s14, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s15, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s13, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 2                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s15   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
// Using GLNC for A
/* NumThreadsCoalescedA = 16, 256 total threads, 2 thread groups */
v_mov_b32 v[vgprGlobalReadOffsetA+0], v[vgprSerial]
v_add_u32 v[vgprGlobalReadOffsetA+1], 256, v[vgprGlobalReadOffsetA+0] //  = vgprSerial + 1 * 256
v_add_u32 v[vgprGlobalReadOffsetA+2], 256, v[vgprGlobalReadOffsetA+1] //  = vgprSerial + 2 * 256
v_add_u32 v[vgprGlobalReadOffsetA+3], 256, v[vgprGlobalReadOffsetA+2] //  = vgprSerial + 3 * 256
v_add_u32 v[vgprGlobalReadOffsetA+4], 256, v[vgprGlobalReadOffsetA+3] //  = vgprSerial + 4 * 256
v_add_u32 v[vgprGlobalReadOffsetA+5], 256, v[vgprGlobalReadOffsetA+4] //  = vgprSerial + 5 * 256
v_add_u32 v[vgprGlobalReadOffsetA+6], 256, v[vgprGlobalReadOffsetA+5] //  = vgprSerial + 6 * 256
v_add_u32 v[vgprGlobalReadOffsetA+7], 256, v[vgprGlobalReadOffsetA+6] //  = vgprSerial + 7 * 256
v_lshrrev_b32 v50, 4, v[vgprGlobalReadOffsetA+0]   // division
v_and_b32 v49, 0xf, v[vgprGlobalReadOffsetA+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 2, v49
v_mul_lo_u32 v50, s[sgprStridesA], v50
v_add_u32 v[vgprGlobalReadOffsetA+0], v50, v[vgprGlobalReadOffsetA+0] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 2, v[vgprGlobalReadOffsetA+0]
v_add_u32 v[vgprGlobalReadOffsetA+0], 16, v[vgprGlobalReadOffsetA+0] // ptr-shift
v_lshrrev_b32 v50, 4, v[vgprGlobalReadOffsetA+1]   // division
v_and_b32 v49, 0xf, v[vgprGlobalReadOffsetA+1]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+1], 2, v49
v_mul_lo_u32 v50, s[sgprStridesA], v50
v_add_u32 v[vgprGlobalReadOffsetA+1], v50, v[vgprGlobalReadOffsetA+1] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+1], 2, v[vgprGlobalReadOffsetA+1]
v_add_u32 v[vgprGlobalReadOffsetA+1], 16, v[vgprGlobalReadOffsetA+1] // ptr-shift
v_lshrrev_b32 v50, 4, v[vgprGlobalReadOffsetA+2]   // division
v_and_b32 v49, 0xf, v[vgprGlobalReadOffsetA+2]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+2], 2, v49
v_mul_lo_u32 v50, s[sgprStridesA], v50
v_add_u32 v[vgprGlobalReadOffsetA+2], v50, v[vgprGlobalReadOffsetA+2] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+2], 2, v[vgprGlobalReadOffsetA+2]
v_add_u32 v[vgprGlobalReadOffsetA+2], 16, v[vgprGlobalReadOffsetA+2] // ptr-shift
v_lshrrev_b32 v50, 4, v[vgprGlobalReadOffsetA+3]   // division
v_and_b32 v49, 0xf, v[vgprGlobalReadOffsetA+3]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+3], 2, v49
v_mul_lo_u32 v50, s[sgprStridesA], v50
v_add_u32 v[vgprGlobalReadOffsetA+3], v50, v[vgprGlobalReadOffsetA+3] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+3], 2, v[vgprGlobalReadOffsetA+3]
v_add_u32 v[vgprGlobalReadOffsetA+3], 16, v[vgprGlobalReadOffsetA+3] // ptr-shift
v_lshrrev_b32 v50, 4, v[vgprGlobalReadOffsetA+4]   // division
v_and_b32 v49, 0xf, v[vgprGlobalReadOffsetA+4]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+4], 2, v49
v_mul_lo_u32 v50, s[sgprStridesA], v50
v_add_u32 v[vgprGlobalReadOffsetA+4], v50, v[vgprGlobalReadOffsetA+4] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+4], 2, v[vgprGlobalReadOffsetA+4]
v_add_u32 v[vgprGlobalReadOffsetA+4], 16, v[vgprGlobalReadOffsetA+4] // ptr-shift
v_lshrrev_b32 v50, 4, v[vgprGlobalReadOffsetA+5]   // division
v_and_b32 v49, 0xf, v[vgprGlobalReadOffsetA+5]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+5], 2, v49
v_mul_lo_u32 v50, s[sgprStridesA], v50
v_add_u32 v[vgprGlobalReadOffsetA+5], v50, v[vgprGlobalReadOffsetA+5] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+5], 2, v[vgprGlobalReadOffsetA+5]
v_add_u32 v[vgprGlobalReadOffsetA+5], 16, v[vgprGlobalReadOffsetA+5] // ptr-shift
v_lshrrev_b32 v50, 4, v[vgprGlobalReadOffsetA+6]   // division
v_and_b32 v49, 0xf, v[vgprGlobalReadOffsetA+6]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+6], 2, v49
v_mul_lo_u32 v50, s[sgprStridesA], v50
v_add_u32 v[vgprGlobalReadOffsetA+6], v50, v[vgprGlobalReadOffsetA+6] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+6], 2, v[vgprGlobalReadOffsetA+6]
v_add_u32 v[vgprGlobalReadOffsetA+6], 16, v[vgprGlobalReadOffsetA+6] // ptr-shift
v_lshrrev_b32 v50, 4, v[vgprGlobalReadOffsetA+7]   // division
v_and_b32 v49, 0xf, v[vgprGlobalReadOffsetA+7]
v_lshlrev_b32 v[vgprGlobalReadOffsetA+7], 2, v49
v_mul_lo_u32 v50, s[sgprStridesA], v50
v_add_u32 v[vgprGlobalReadOffsetA+7], v50, v[vgprGlobalReadOffsetA+7] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetA+7], 2, v[vgprGlobalReadOffsetA+7]
v_add_u32 v[vgprGlobalReadOffsetA+7], 16, v[vgprGlobalReadOffsetA+7] // ptr-shift

/* global read addresses: final offsets b */
// Using GLNC for B
/* NumThreadsCoalescedB = 16, 256 total threads, 2 thread groups */
v_mov_b32 v[vgprGlobalReadOffsetB+0], v[vgprSerial]
v_add_u32 v[vgprGlobalReadOffsetB+1], 256, v[vgprGlobalReadOffsetB+0] //  = vgprSerial + 1 * 256
v_add_u32 v[vgprGlobalReadOffsetB+2], 256, v[vgprGlobalReadOffsetB+1] //  = vgprSerial + 2 * 256
v_add_u32 v[vgprGlobalReadOffsetB+3], 256, v[vgprGlobalReadOffsetB+2] //  = vgprSerial + 3 * 256
v_add_u32 v[vgprGlobalReadOffsetB+4], 256, v[vgprGlobalReadOffsetB+3] //  = vgprSerial + 4 * 256
v_add_u32 v[vgprGlobalReadOffsetB+5], 256, v[vgprGlobalReadOffsetB+4] //  = vgprSerial + 5 * 256
v_add_u32 v[vgprGlobalReadOffsetB+6], 256, v[vgprGlobalReadOffsetB+5] //  = vgprSerial + 6 * 256
v_add_u32 v[vgprGlobalReadOffsetB+7], 256, v[vgprGlobalReadOffsetB+6] //  = vgprSerial + 7 * 256
v_add_u32 v[vgprGlobalReadOffsetB+8], 256, v[vgprGlobalReadOffsetB+7] //  = vgprSerial + 8 * 256
v_add_u32 v[vgprGlobalReadOffsetB+9], 256, v[vgprGlobalReadOffsetB+8] //  = vgprSerial + 9 * 256
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+0]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+0]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+0], v20, v[vgprGlobalReadOffsetB+0] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 2, v[vgprGlobalReadOffsetB+0]
v_add_u32 v[vgprGlobalReadOffsetB+0], 16, v[vgprGlobalReadOffsetB+0] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+1]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+1]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+1], v20, v[vgprGlobalReadOffsetB+1] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 2, v[vgprGlobalReadOffsetB+1]
v_add_u32 v[vgprGlobalReadOffsetB+1], 16, v[vgprGlobalReadOffsetB+1] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+2]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+2]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+2], v20, v[vgprGlobalReadOffsetB+2] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 2, v[vgprGlobalReadOffsetB+2]
v_add_u32 v[vgprGlobalReadOffsetB+2], 16, v[vgprGlobalReadOffsetB+2] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+3]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+3]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+3], v20, v[vgprGlobalReadOffsetB+3] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 2, v[vgprGlobalReadOffsetB+3]
v_add_u32 v[vgprGlobalReadOffsetB+3], 16, v[vgprGlobalReadOffsetB+3] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+4]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+4]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+4], v20, v[vgprGlobalReadOffsetB+4] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 2, v[vgprGlobalReadOffsetB+4]
v_add_u32 v[vgprGlobalReadOffsetB+4], 16, v[vgprGlobalReadOffsetB+4] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+5]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+5]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+5], v20, v[vgprGlobalReadOffsetB+5] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 2, v[vgprGlobalReadOffsetB+5]
v_add_u32 v[vgprGlobalReadOffsetB+5], 16, v[vgprGlobalReadOffsetB+5] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+6]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+6]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+6], v20, v[vgprGlobalReadOffsetB+6] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 2, v[vgprGlobalReadOffsetB+6]
v_add_u32 v[vgprGlobalReadOffsetB+6], 16, v[vgprGlobalReadOffsetB+6] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+7]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+7]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+7], v20, v[vgprGlobalReadOffsetB+7] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 2, v[vgprGlobalReadOffsetB+7]
v_add_u32 v[vgprGlobalReadOffsetB+7], 16, v[vgprGlobalReadOffsetB+7] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+8]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+8]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+8], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+8], v20, v[vgprGlobalReadOffsetB+8] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+8], 2, v[vgprGlobalReadOffsetB+8]
v_add_u32 v[vgprGlobalReadOffsetB+8], 16, v[vgprGlobalReadOffsetB+8] // ptr-shift
v_lshrrev_b32 v20, 4, v[vgprGlobalReadOffsetB+9]   // division
v_and_b32 v24, 0xf, v[vgprGlobalReadOffsetB+9]
/* Computing strided(8) perp indicies */
v_and_b32 v29, 1, v20                              // r0 = I % (16 // 8)
v_lshlrev_b32 v29, 3, v29                          // r0 = 8 * r0
/* Computing r1 = (I % 16) // (16 // 8) */
v_and_b32 v30, 15, v20                             // r1 = I % (16)
v_lshrrev_b32 v30, 1, v30                          // r1 = (r1) // (16 // 8)
v_add_u32 v29, v29, v30                            // r0 = r0 + r1
v_lshrrev_b32 v30, 4, v20                          // r1 = I // 16
v_lshl_add_u32 v20, v30, 4, v29                    // v20 = v30 * 16
/* Done computing strided(8) perp indices */
v_lshlrev_b32 v[vgprGlobalReadOffsetB+9], 2, v24
v_mul_lo_u32 v20, s[sgprStridesB], v20
v_add_u32 v[vgprGlobalReadOffsetB+9], v20, v[vgprGlobalReadOffsetB+9] // final
v_lshlrev_b32 v[vgprGlobalReadOffsetB+9], 2, v[vgprGlobalReadOffsetB+9]
v_add_u32 v[vgprGlobalReadOffsetB+9], 16, v[vgprGlobalReadOffsetB+9] // ptr-shift

/* global read addresses: increments a */
s_mov_b32 s[sgprGlobalReadIncsA+0], 256            // incrA (unrollIdx)

/* global read addresses: increments b */
s_mov_b32 s[sgprGlobalReadIncsB+0], 256            // incrB (unrollIdx)
/* declare loop num iterations */
s_sub_u32 s[sgprLoopCounterL], s[sgprStreamKLocalEnd], s[sgprStreamKLocalStart] // StreamK loop counter = localEnd - localStart
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck2                 // branch if s[Alpha] != 0
s_mov_b32 s[sgprLoopCounterL], 0                   // Skip iterations
label_SKAlphaCheck2:
s_and_b32 s13, 63, s[sgprSizesSum+0]               // s13 = s[sgprSizesSum+0] % 64
s_cmp_eq_u32 s13, 0                                // numIterL == 0
s_cselect_b32 s12, 0, 1                            // check if size uses tail loop
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Check if WG processes final iteration of tile
s_cselect_b32 s12, s12, 0                          // this WG runs tail loop
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], s12 // Adjust loop counter for tail loop
s_max_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0 // Avoid setting negative value to loopCounter
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s14, s[sgprStaggerU], 0x1f00
s_lshr_b32 s14, s14, 0x8
s_and_b32 s15, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s12, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s13, s12, s14                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s13           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s12, s12, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s13, s12, 1                              // staggerU mask
s_cmp_ge_u32 s12, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s13, 0          // set Mask
s_cmp_eq_u32 s15, 0x0
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s12, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s15, 0x2000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s12, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s15, 0x4000
s_cbranch_scc0 label_StaggerUMapping_3
s_mov_b32 s12, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s15, 0x6000
s_cbranch_scc0 label_StaggerUMapping_4
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s12, s12, s13
s_add_u32 s12, s12, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s15, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s12, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s12 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s14 // shift by StaggerUStride
s_cmp_gt_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cmov_b32 s[sgprStaggerUIter], 0                  // set stagger=0 for partial tiles
s_cmp_lt_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cmov_b32 s[sgprStaggerUIter], 0                  // set stagger=0 for partial tiles

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s13, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s12, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s13, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s12, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // limit -= inc)
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
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_4_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_5_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_6_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_7_0
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_4_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_5_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_6_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_7_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_8_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_9_0

/* global read inc A loopL */
s_add_u32 s14, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s14              // Is this wrapIter? (pf)
s_cselect_b32 s12, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s13, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s14, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s14              // Is this wrapIter? (pf)
s_cselect_b32 s12, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s13, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmSrdInitiationD
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_GeneralBatchedGemmSrdInitiationD:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationD_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmSrdInitiationC
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_GeneralBatchedGemmSrdInitiationC:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationC_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s64, 2
s_mov_b32 s65, 2
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_BPEDone                       // If synchronizer, use regular output BPE
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_BPEDone                       // If split == 1, use reguler output BPE
s_mov_b32 s64, 2
s_mov_b32 s65, 2
label_BPEDone:

s_mul_i32 s86, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s85, s86, s[sgprStrideC1J]            // ScaleC s86 by Stride
s_mul_i32 s84, s86, s[sgprStrideC1J]               // ScaleC s86 by Stride
s_lshl_b64 s[84:85], s[84:85], s64                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s84        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s85       // add hi to SRD
s_mul_hi_u32 s85, s86, s[sgprStrideD1J]            // ScaleD s86 by Stride
s_mul_i32 s84, s86, s[sgprStrideD1J]               // ScaleD s86 by Stride
s_lshl_b64 s[84:85], s[84:85], s65                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s84        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s85       // add hi to SRD

s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s85, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s84, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[84:85], s[84:85], s64                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s84        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s85       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s84, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s84, s84, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s85, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[84:85], s[84:85], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s84        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s85       // Offsetting within the Batch Matrix [Higher half of address]
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s85, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s84, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[84:85], s[84:85], s65                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s84        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s85       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s84, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s84, s84, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s85, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[84:85], s[84:85], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s84        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s85       // Offsetting within the Batch Matrix [Higher half of address]
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitSrd                   // Skip this block if using single-kernel stream-k fixup
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_SK_SplitSrd                   // branch if split == 1
// Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s
s_mul_hi_u32 s85, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_mul_i32 s84, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_sub_u32 s86, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s86, s86, s[sgprSkPartialIdx]            // Free1
s_mul_hi_u32 s87, s86, s[sgprStrideC1J]            // Free1
s_mul_i32 s86, s86, s[sgprStrideC1J]               // Free1
s_add_u32 s84, s84, s86                            // Free1
s_addc_u32 s85, s85, s87                           // Free1
s_sub_u32 s86, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s86, s86, s[sgprSkPartialIdx]            // Free2
s_mul_hi_u32 s87, s86, s[sgprStrideCK]             // Free2
s_mul_i32 s86, s86, s[sgprStrideCK]                // Free2
s_add_u32 s84, s84, s86                            // Free2
s_addc_u32 s85, s85, s87                           // Free2
s_lshl_b64 s[84:85], s[84:85], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s84        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s85       // add hi GSU offset to SRD
label_SK_SplitSrd:

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...80) from pool */

/* initC: remove ValuA/B vgpr buffer [20...164) from pool */
v_mov_b64 v[202:203], 0                            // A/B=0
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
v_mfma_i32_32x32x16_i8 acc[16:31], v[202:203], v[202:203], acc[0:15] // initC: [16, 31]
v_mfma_i32_32x32x16_i8 acc[32:47], v[202:203], v[202:203], acc[0:15] // initC: [32, 47]
v_mfma_i32_32x32x16_i8 acc[48:63], v[202:203], v[202:203], acc[0:15] // initC: [48, 63]
v_mfma_i32_32x32x16_i8 acc[64:79], v[202:203], v[202:203], acc[0:15] // initC: [64, 79]
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_8S4L1KCK9VFC7AQU     // Only branch on scc1
s_getpc_b64 s[64:65]                               // addr of next instr
s_add_i32 s66, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s64, s64, s66                            // add target branch offset
s_addc_u32 s65, s65, 0                             // add high and carry
s_setpc_b64 s[64:65]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_8S4L1KCK9VFC7AQU:
s_barrier                                          // For stream-k / persistent loop

/* local write swap a */
s_xor_b32 s[sgprLocalWriteAddrA], s[sgprSwapA], s[sgprLocalWriteAddrA] // swap Red Blk SGPR

/* local write swap b */
s_xor_b32 s[sgprLocalWriteAddrB], s[sgprSwapB], s[sgprLocalWriteAddrB] // swap Red Blk SGPR
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
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_4_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_5_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_6_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_7_0
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_4_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_5_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_6_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_7_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_8_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_9_0

/* local write swap a */
s_xor_b32 s[sgprLocalWriteAddrA], s[sgprSwapA], s[sgprLocalWriteAddrA] // swap Red Blk SGPR

/* local write swap b */
s_xor_b32 s[sgprLocalWriteAddrB], s[sgprSwapB], s[sgprLocalWriteAddrB] // swap Red Blk SGPR
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
s_waitcnt vmcnt(0)                                 // wait for global reads with lds (for early exit)
label_skipPGR2_2:
s_waitcnt vmcnt(18)                                // wait for global reads with lds
// Skip force waitcnt0
s_barrier                                          // LW to PLR, sync LDS0

/* local read prefetch a */
ds_read_b128 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:16896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:16960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:25344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:25408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:33792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:33856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
/* N/A, lro->32 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read inc b */
/* N/A, lro->32 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

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

/* iter 0 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:26, lwEndMfmaIndex:97  */
/*  numMfmaForLR:22, syncPlrMfmaIndex:97 , sync1LdsMfmaIndex:25 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+1], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[166:167], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[166:167], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[166:167], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s64, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s65, s[sgprWrapUA+1], 0              // incUpper <- ?
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+5], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+4], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+5], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+4], v202, v203 // __TF32_2_B_0 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s64 // limit -= inc)
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+8], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+9], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+10], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+11], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[166:167], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[166:167], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+8], v[vgprValuB_X0_I0+8], v[vgprValuB_X0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+9], v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+11], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[166:167], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1], v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+13], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+12], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+15], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+13], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+12], v202, v203 // __TF32_2_B_1 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s64, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s65, s[sgprWrapUB+1], 0              // incUpper <- ?
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+16], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+17], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+18], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+19], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[166:167], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+1], v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[166:167], v[vgprValuA_X0_I0+18:vgprValuA_X0_I0+18+1], v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+16], v[vgprValuB_X0_I0+16], v[vgprValuB_X0_I0+17]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+17], v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+19]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+19], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[166:167], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1], v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s64 // limit -= inc)
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+23], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+21], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+20], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+23], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+21], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+20], v202, v203 // __TF32_2_B_2 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+24], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+25], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+26], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+27], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[166:167], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+1], v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[166:167], v[vgprValuA_X0_I0+26:vgprValuA_X0_I0+26+1], v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+24], v[vgprValuB_X0_I0+24], v[vgprValuB_X0_I0+25]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+25], v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+27]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+27], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[166:167], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1], v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3], v[vgprLocalReadAddrA+0] offset:640 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+31], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+29], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+28], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+31], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+29], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+28], v202, v203 // __TF32_2_B_3 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:704 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+34:vgprValuB_X0_I0+34+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+32], v[vgprValuB_X0_I0+32], v[vgprValuB_X0_I0+33]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+33], v[vgprValuB_X0_I0+34], v[vgprValuB_X0_I0+35]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+34], v[vgprValuB_X0_I0+36], v[vgprValuB_X0_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+35], v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+39]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[166:167], v[vgprValuB_X0_I0+34:vgprValuB_X0_I0+34+1], v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3] // Calculate low bits for TF32 emulation__TF32_1_B_4: 
s_nop 4                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3], v[vgprLocalReadAddrA+0] offset:896 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+39], v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+39] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+36], v[vgprValuB_X0_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+37], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+36], v202, v203 // __TF32_2_B_4 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:10  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:960 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
/*  mfmaIndex:11  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:8576 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:12  */
ds_read_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:8640 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:17024 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:17088 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprLocalReadAddrB+0] offset:25472 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:16  */
ds_read_b128 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[vgprLocalReadAddrB+0] offset:25536 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
/*  mfmaIndex:17  */
ds_read_b128 v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprLocalReadAddrB+0] offset:33920 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:18  */
ds_read_b128 v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3], v[vgprLocalReadAddrB+0] offset:33984 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:24  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:25  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
/*  mfmaIndex:26  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:27  */
s_mov_b32 m0, s[sgprLocalWriteAddrA]               // m0 <- LDS write address
/* before DirectToLds load, ensure prior ds_reads have finished */
s_waitcnt lgkmcnt(0)
s_barrier
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:28  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
/*  mfmaIndex:29  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:30  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:31  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
/*  mfmaIndex:32  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:33  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:34  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
/*  mfmaIndex:35  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:36  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:37  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
/*  mfmaIndex:38  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:39  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:40  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
/*  mfmaIndex:41  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:42  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:43  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_4_0
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
/*  mfmaIndex:44  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:45  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:46  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
/*  mfmaIndex:47  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_5_0
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:48  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[64:67] // src0_h*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:49  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[64:67] // src0_h*src1_l, left value = acc[64+0:67+0]
/*  mfmaIndex:50  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[64:67] // src0_l*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:51  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[68:71] // src0_h*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:52  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_6_0
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[68:71] // src0_h*src1_l, left value = acc[68+0:71+0]
/*  mfmaIndex:53  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[68:71] // src0_l*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:54  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[72:75] // src0_h*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:55  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[72:75] // src0_h*src1_l, left value = acc[72+0:75+0]
/*  mfmaIndex:56  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_7_0
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[72:75] // src0_l*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:57  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[76:79] // src0_h*src1_h, left value = acc[76+0:79+0]
/*  mfmaIndex:58  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[76:79] // src0_h*src1_l, left value = acc[76+0:79+0]
/*  mfmaIndex:59  */

/* local read swap offsets a */
v_xor_b32 v[vgprLocalReadAddrA], v[vgprLocalReadSwapAddrA], v[vgprLocalReadAddrA] // swap Red Blk

/* local read swap offsets b */
v_xor_b32 v[vgprLocalReadAddrB], v[vgprLocalReadSwapAddrB], v[vgprLocalReadAddrB] // swap Red Blk

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[76:79] // src0_l*src1_h, left value = acc[76+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=10 */

/* iter 1 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:26, lwEndMfmaIndex:97  */
/*  numMfmaForLR:22, syncPlrMfmaIndex:97 , sync1LdsMfmaIndex:25 */
/*  mfmaIndex:60  */
s_mov_b32 m0, s[sgprLocalWriteAddrB]               // m0 <- LDS write address
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_0_0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+0], v[vgprValuA_T1_I0+0], v[vgprValuA_T1_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+1], v[vgprValuA_T1_I0+2], v[vgprValuA_T1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3], v[166:167], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[166:167], v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+1], v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+3], v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[166:167], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:61  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+7], v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+5], v[vgprValuA_T1_I0+2], v[vgprValuA_T1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+4], v[vgprValuA_T1_I0+0], v[vgprValuA_T1_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+7], v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+5], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+4], v202, v203 // __TF32_2_B_0 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
/*  mfmaIndex:62  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+8], v[vgprValuA_T1_I0+4], v[vgprValuA_T1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+9], v[vgprValuA_T1_I0+6], v[vgprValuA_T1_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+10], v[vgprValuA_X1_I0+12], v[vgprValuA_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+11], v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3], v[166:167], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[166:167], v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+10:vgprValuB_X1_I0+10+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+8], v[vgprValuB_X1_I0+8], v[vgprValuB_X1_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+9], v[vgprValuB_X1_I0+10], v[vgprValuB_X1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+10], v[vgprValuB_X1_I0+12], v[vgprValuB_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+11], v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[166:167], v[vgprValuB_X1_I0+10:vgprValuB_X1_I0+10+1], v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:63  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+15], v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+12], v[vgprValuA_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+13], v[vgprValuA_T1_I0+6], v[vgprValuA_T1_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+12], v[vgprValuA_T1_I0+4], v[vgprValuA_T1_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+15], v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+12], v[vgprValuB_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+13], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+12], v202, v203 // __TF32_2_B_1 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:64  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_1_0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+16], v[vgprValuA_T1_I0+8], v[vgprValuA_T1_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+17], v[vgprValuA_T1_I0+10], v[vgprValuA_T1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+18], v[vgprValuA_X1_I0+20], v[vgprValuA_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+19], v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3], v[166:167], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+1], v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[166:167], v[vgprValuA_X1_I0+18:vgprValuA_X1_I0+18+1], v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+18:vgprValuB_X1_I0+18+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+16], v[vgprValuB_X1_I0+16], v[vgprValuB_X1_I0+17]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+17], v[vgprValuB_X1_I0+18], v[vgprValuB_X1_I0+19]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+18], v[vgprValuB_X1_I0+20], v[vgprValuB_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+19], v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[166:167], v[vgprValuB_X1_I0+18:vgprValuB_X1_I0+18+1], v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
/*  mfmaIndex:65  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+23], v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+20], v[vgprValuA_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+21], v[vgprValuA_T1_I0+10], v[vgprValuA_T1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+20], v[vgprValuA_T1_I0+8], v[vgprValuA_T1_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+23], v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+20], v[vgprValuB_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+21], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+20], v202, v203 // __TF32_2_B_2 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:66  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+24], v[vgprValuA_T1_I0+12], v[vgprValuA_T1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+25], v[vgprValuA_T1_I0+14], v[vgprValuA_T1_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+26], v[vgprValuA_X1_I0+28], v[vgprValuA_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+27], v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3], v[166:167], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+1], v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[166:167], v[vgprValuA_X1_I0+26:vgprValuA_X1_I0+26+1], v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+26:vgprValuB_X1_I0+26+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+24], v[vgprValuB_X1_I0+24], v[vgprValuB_X1_I0+25]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+25], v[vgprValuB_X1_I0+26], v[vgprValuB_X1_I0+27]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+26], v[vgprValuB_X1_I0+28], v[vgprValuB_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+27], v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[166:167], v[vgprValuB_X1_I0+26:vgprValuB_X1_I0+26+1], v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:67  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+31], v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+28], v[vgprValuA_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+29], v[vgprValuA_T1_I0+14], v[vgprValuA_T1_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+28], v[vgprValuA_T1_I0+12], v[vgprValuA_T1_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+31], v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+28], v[vgprValuB_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+29], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+28], v202, v203 // __TF32_2_B_3 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
/*  mfmaIndex:68  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_2_0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+34:vgprValuB_X1_I0+34+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+32], v[vgprValuB_X1_I0+32], v[vgprValuB_X1_I0+33]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+33], v[vgprValuB_X1_I0+34], v[vgprValuB_X1_I0+35]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+34], v[vgprValuB_X1_I0+36], v[vgprValuB_X1_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+35], v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+39]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3], v[166:167], v[vgprValuB_X1_I0+34:vgprValuB_X1_I0+34+1], v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3] // Calculate low bits for TF32 emulation__TF32_1_B_4: 
s_nop 4                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:69  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+39], v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+39] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+36], v[vgprValuB_X1_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+37], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+36], v202, v203 // __TF32_2_B_4 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:70  */
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
/*  mfmaIndex:71  */
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:72  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_3_0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:73  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
/*  mfmaIndex:74  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:75  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:76  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
/*  mfmaIndex:77  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_4_0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:78  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:79  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
/*  mfmaIndex:80  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:81  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_5_0
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:82  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
/*  mfmaIndex:83  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:84  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:85  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_6_0
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
/*  mfmaIndex:86  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:87  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:88  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
/*  mfmaIndex:89  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_7_0
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:90  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:91  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
/*  mfmaIndex:92  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:93  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_8_0
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:94  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
/*  mfmaIndex:95  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:96  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:97  */
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_9_0

/* local write swap offsets a */
s_xor_b32 s[sgprLocalWriteAddrA], s[sgprSwapA], s[sgprLocalWriteAddrA] // swap Red Blk SGPR

/* local write swap offsets b */
s_xor_b32 s[sgprLocalWriteAddrB], s[sgprSwapB], s[sgprLocalWriteAddrB] // swap Red Blk SGPR
s_waitcnt vmcnt(18)                                // wait for previous set of global reads
// Skip force waitcnt0
s_barrier                                          // PGR, and wait until LW done to sync LDS1
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
/*  mfmaIndex:98  */
ds_read_b128 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:99  */
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:100  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
/*  mfmaIndex:101  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:102  */
ds_read_b128 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:103  */
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
/*  mfmaIndex:104  */
ds_read_b128 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:105  */
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:106  */
ds_read_b128 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
/*  mfmaIndex:107  */
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:108  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[64:67] // src0_h*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:109  */
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[64:67] // src0_h*src1_l, left value = acc[64+0:67+0]
/*  mfmaIndex:110  */
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:16896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[64:67] // src0_l*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:111  */
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:16960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[68:71] // src0_h*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:112  */
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:25344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[68:71] // src0_h*src1_l, left value = acc[68+0:71+0]
/*  mfmaIndex:113  */
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:25408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[68:71] // src0_l*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:114  */
ds_read_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:33792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[72:75] // src0_h*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:115  */
ds_read_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:33856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[72:75] // src0_h*src1_l, left value = acc[72+0:75+0]
/*  mfmaIndex:116  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[72:75] // src0_l*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:117  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[76:79] // src0_h*src1_h, left value = acc[76+0:79+0]
/*  mfmaIndex:118  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[76:79] // src0_h*src1_l, left value = acc[76+0:79+0]
/*  mfmaIndex:119  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[76:79] // src0_l*src1_h, left value = acc[76+0:79+0]

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop_1 - Begin        */
/******************************************/

/* iter 0 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:26, lwEndMfmaIndex:97  */
/*  numMfmaForLR:22, syncPlrMfmaIndex:97 , sync1LdsMfmaIndex:25 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+1], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[166:167], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[166:167], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[166:167], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s64, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s65, s[sgprWrapUA+1], 0              // incUpper <- ?
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+5], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+4], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+5], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+4], v202, v203 // __TF32_2_B_0 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s64 // limit -= inc)
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+8], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+9], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+10], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+11], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[166:167], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[166:167], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+8], v[vgprValuB_X0_I0+8], v[vgprValuB_X0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+9], v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+11], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[166:167], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1], v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+13], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+12], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+15], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+13], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+12], v202, v203 // __TF32_2_B_1 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s64, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s65, s[sgprWrapUB+1], 0              // incUpper <- ?
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+16], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+17], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+18], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+19], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[166:167], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+1], v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[166:167], v[vgprValuA_X0_I0+18:vgprValuA_X0_I0+18+1], v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+16], v[vgprValuB_X0_I0+16], v[vgprValuB_X0_I0+17]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+17], v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+19]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+19], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[166:167], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1], v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s64 // limit -= inc)
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+23], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+21], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+20], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+23], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+21], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+20], v202, v203 // __TF32_2_B_2 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+24], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+25], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+26], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+27], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[166:167], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+1], v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[166:167], v[vgprValuA_X0_I0+26:vgprValuA_X0_I0+26+1], v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+24], v[vgprValuB_X0_I0+24], v[vgprValuB_X0_I0+25]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+25], v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+27]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+27], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[166:167], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1], v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3], v[vgprLocalReadAddrA+0] offset:640 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+31], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+29], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+28], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+31], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+29], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+28], v202, v203 // __TF32_2_B_3 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:704 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+34:vgprValuB_X0_I0+34+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+32], v[vgprValuB_X0_I0+32], v[vgprValuB_X0_I0+33]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+33], v[vgprValuB_X0_I0+34], v[vgprValuB_X0_I0+35]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+34], v[vgprValuB_X0_I0+36], v[vgprValuB_X0_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+35], v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+39]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[166:167], v[vgprValuB_X0_I0+34:vgprValuB_X0_I0+34+1], v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3] // Calculate low bits for TF32 emulation__TF32_1_B_4: 
s_nop 4                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3], v[vgprLocalReadAddrA+0] offset:896 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+39], v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+39] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+36], v[vgprValuB_X0_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+37], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+36], v202, v203 // __TF32_2_B_4 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:10  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:960 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
/*  mfmaIndex:11  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:8576 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:12  */
ds_read_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:8640 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:17024 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:17088 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprLocalReadAddrB+0] offset:25472 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:16  */
ds_read_b128 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[vgprLocalReadAddrB+0] offset:25536 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
/*  mfmaIndex:17  */
ds_read_b128 v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprLocalReadAddrB+0] offset:33920 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:18  */
ds_read_b128 v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3], v[vgprLocalReadAddrB+0] offset:33984 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:24  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:25  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
/*  mfmaIndex:26  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:27  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:28  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
/*  mfmaIndex:29  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:30  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:31  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
/*  mfmaIndex:32  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:33  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:34  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
/*  mfmaIndex:35  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:36  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:37  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
/*  mfmaIndex:38  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:39  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:40  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
/*  mfmaIndex:41  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:42  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:43  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
/*  mfmaIndex:44  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:45  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:46  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
/*  mfmaIndex:47  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:48  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[64:67] // src0_h*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:49  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[64:67] // src0_h*src1_l, left value = acc[64+0:67+0]
/*  mfmaIndex:50  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[64:67] // src0_l*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:51  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[68:71] // src0_h*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:52  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[68:71] // src0_h*src1_l, left value = acc[68+0:71+0]
/*  mfmaIndex:53  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[68:71] // src0_l*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:54  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[72:75] // src0_h*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:55  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[72:75] // src0_h*src1_l, left value = acc[72+0:75+0]
/*  mfmaIndex:56  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[72:75] // src0_l*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:57  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[76:79] // src0_h*src1_h, left value = acc[76+0:79+0]
/*  mfmaIndex:58  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[76:79] // src0_h*src1_l, left value = acc[76+0:79+0]
/*  mfmaIndex:59  */

/* local read swap offsets a */
v_xor_b32 v[vgprLocalReadAddrA], v[vgprLocalReadSwapAddrA], v[vgprLocalReadAddrA] // swap Red Blk

/* local read swap offsets b */
v_xor_b32 v[vgprLocalReadAddrB], v[vgprLocalReadSwapAddrB], v[vgprLocalReadAddrB] // swap Red Blk

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[76:79] // src0_l*src1_h, left value = acc[76+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=10 */

/* iter 1 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:26, lwEndMfmaIndex:97  */
/*  numMfmaForLR:22, syncPlrMfmaIndex:97 , sync1LdsMfmaIndex:25 */
/*  mfmaIndex:60  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+0], v[vgprValuA_T1_I0+0], v[vgprValuA_T1_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+1], v[vgprValuA_T1_I0+2], v[vgprValuA_T1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3], v[166:167], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[166:167], v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+1], v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+3], v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[166:167], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:61  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+7], v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+5], v[vgprValuA_T1_I0+2], v[vgprValuA_T1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+4], v[vgprValuA_T1_I0+0], v[vgprValuA_T1_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+7], v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+5], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+4], v202, v203 // __TF32_2_B_0 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
/*  mfmaIndex:62  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+8], v[vgprValuA_T1_I0+4], v[vgprValuA_T1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+9], v[vgprValuA_T1_I0+6], v[vgprValuA_T1_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+10], v[vgprValuA_X1_I0+12], v[vgprValuA_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+11], v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3], v[166:167], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[166:167], v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+10:vgprValuB_X1_I0+10+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+8], v[vgprValuB_X1_I0+8], v[vgprValuB_X1_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+9], v[vgprValuB_X1_I0+10], v[vgprValuB_X1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+10], v[vgprValuB_X1_I0+12], v[vgprValuB_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+11], v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[166:167], v[vgprValuB_X1_I0+10:vgprValuB_X1_I0+10+1], v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:63  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+15], v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+12], v[vgprValuA_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+13], v[vgprValuA_T1_I0+6], v[vgprValuA_T1_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+12], v[vgprValuA_T1_I0+4], v[vgprValuA_T1_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+15], v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+12], v[vgprValuB_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+13], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+12], v202, v203 // __TF32_2_B_1 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:64  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+16], v[vgprValuA_T1_I0+8], v[vgprValuA_T1_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+17], v[vgprValuA_T1_I0+10], v[vgprValuA_T1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+18], v[vgprValuA_X1_I0+20], v[vgprValuA_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+19], v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3], v[166:167], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+1], v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[166:167], v[vgprValuA_X1_I0+18:vgprValuA_X1_I0+18+1], v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+18:vgprValuB_X1_I0+18+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+16], v[vgprValuB_X1_I0+16], v[vgprValuB_X1_I0+17]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+17], v[vgprValuB_X1_I0+18], v[vgprValuB_X1_I0+19]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+18], v[vgprValuB_X1_I0+20], v[vgprValuB_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+19], v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[166:167], v[vgprValuB_X1_I0+18:vgprValuB_X1_I0+18+1], v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
/*  mfmaIndex:65  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+23], v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+20], v[vgprValuA_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+21], v[vgprValuA_T1_I0+10], v[vgprValuA_T1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+20], v[vgprValuA_T1_I0+8], v[vgprValuA_T1_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+23], v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+20], v[vgprValuB_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+21], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+20], v202, v203 // __TF32_2_B_2 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:66  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+24], v[vgprValuA_T1_I0+12], v[vgprValuA_T1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+25], v[vgprValuA_T1_I0+14], v[vgprValuA_T1_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+26], v[vgprValuA_X1_I0+28], v[vgprValuA_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+27], v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3], v[166:167], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+1], v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[166:167], v[vgprValuA_X1_I0+26:vgprValuA_X1_I0+26+1], v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+26:vgprValuB_X1_I0+26+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+24], v[vgprValuB_X1_I0+24], v[vgprValuB_X1_I0+25]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+25], v[vgprValuB_X1_I0+26], v[vgprValuB_X1_I0+27]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+26], v[vgprValuB_X1_I0+28], v[vgprValuB_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+27], v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[166:167], v[vgprValuB_X1_I0+26:vgprValuB_X1_I0+26+1], v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:67  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+31], v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+28], v[vgprValuA_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+29], v[vgprValuA_T1_I0+14], v[vgprValuA_T1_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+28], v[vgprValuA_T1_I0+12], v[vgprValuA_T1_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+31], v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+28], v[vgprValuB_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+29], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+28], v202, v203 // __TF32_2_B_3 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
/*  mfmaIndex:68  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+34:vgprValuB_X1_I0+34+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+32], v[vgprValuB_X1_I0+32], v[vgprValuB_X1_I0+33]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+33], v[vgprValuB_X1_I0+34], v[vgprValuB_X1_I0+35]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+34], v[vgprValuB_X1_I0+36], v[vgprValuB_X1_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+35], v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+39]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3], v[166:167], v[vgprValuB_X1_I0+34:vgprValuB_X1_I0+34+1], v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3] // Calculate low bits for TF32 emulation__TF32_1_B_4: 
s_nop 4                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:69  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+39], v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+39] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+36], v[vgprValuB_X1_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+37], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+36], v202, v203 // __TF32_2_B_4 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:70  */
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
/*  mfmaIndex:71  */
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:72  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:73  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
/*  mfmaIndex:74  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:75  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:76  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
/*  mfmaIndex:77  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:78  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:79  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
/*  mfmaIndex:80  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:81  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:82  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
/*  mfmaIndex:83  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:84  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:85  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
/*  mfmaIndex:86  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:87  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:88  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
/*  mfmaIndex:89  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:90  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:91  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
/*  mfmaIndex:92  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:93  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:94  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
/*  mfmaIndex:95  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:96  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:97  */
s_waitcnt vmcnt(0)                                 // wait for global reads with lds
// Skip force waitcnt0
s_barrier                                          // noLoadLoop sync LDS0
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
/*  mfmaIndex:98  */
ds_read_b128 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:99  */
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:100  */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
/*  mfmaIndex:101  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:102  */
ds_read_b128 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:103  */
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
/*  mfmaIndex:104  */
ds_read_b128 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:105  */
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:106  */
ds_read_b128 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
/*  mfmaIndex:107  */
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:108  */
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[64:67] // src0_h*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:109  */
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[64:67] // src0_h*src1_l, left value = acc[64+0:67+0]
/*  mfmaIndex:110  */
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:16896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[64:67] // src0_l*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:111  */
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:16960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[68:71] // src0_h*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:112  */
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:25344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[68:71] // src0_h*src1_l, left value = acc[68+0:71+0]
/*  mfmaIndex:113  */
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:25408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[68:71] // src0_l*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:114  */
ds_read_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:33792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[72:75] // src0_h*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:115  */
ds_read_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:33856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[72:75] // src0_h*src1_l, left value = acc[72+0:75+0]
/*  mfmaIndex:116  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[72:75] // src0_l*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:117  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[76:79] // src0_h*src1_h, left value = acc[76+0:79+0]
/*  mfmaIndex:118  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[76:79] // src0_h*src1_l, left value = acc[76+0:79+0]
/*  mfmaIndex:119  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[76:79] // src0_l*src1_h, left value = acc[76+0:79+0]
/* numPrefetchIter=1 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=10 */
label_toPGR1:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:97, lwEndMfmaIndex:97  */
/*  numMfmaForLR:22, syncPlrMfmaIndex:97 , sync1LdsMfmaIndex:96 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+1], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[166:167], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[166:167], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[166:167], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+5], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+4], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+5], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+4], v202, v203 // __TF32_2_B_0 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+8], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+9], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+10], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+11], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[166:167], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[166:167], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+8], v[vgprValuB_X0_I0+8], v[vgprValuB_X0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+9], v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+11], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[166:167], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1], v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:128 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+13], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+12], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+15], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+13], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+12], v202, v203 // __TF32_2_B_1 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:192 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+16], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+17], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+18], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+19], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[166:167], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+1], v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[166:167], v[vgprValuA_X0_I0+18:vgprValuA_X0_I0+18+1], v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+16], v[vgprValuB_X0_I0+16], v[vgprValuB_X0_I0+17]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+17], v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+19]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+19], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[166:167], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1], v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+23], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+21], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+20], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+23], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+21], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+20], v202, v203 // __TF32_2_B_2 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+24], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+25], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+26], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+27], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[166:167], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+1], v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[166:167], v[vgprValuA_X0_I0+26:vgprValuA_X0_I0+26+1], v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+24], v[vgprValuB_X0_I0+24], v[vgprValuB_X0_I0+25]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+25], v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+27]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+27], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[166:167], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1], v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3], v[vgprLocalReadAddrA+0] offset:640 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+31], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+29], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+28], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+31], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+29], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+28], v202, v203 // __TF32_2_B_3 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:704 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+34:vgprValuB_X0_I0+34+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+32], v[vgprValuB_X0_I0+32], v[vgprValuB_X0_I0+33]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+33], v[vgprValuB_X0_I0+34], v[vgprValuB_X0_I0+35]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+34], v[vgprValuB_X0_I0+36], v[vgprValuB_X0_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+35], v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+39]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[166:167], v[vgprValuB_X0_I0+34:vgprValuB_X0_I0+34+1], v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3] // Calculate low bits for TF32 emulation__TF32_1_B_4: 
s_nop 4                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:9  */
ds_read_b128 v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3], v[vgprLocalReadAddrA+0] offset:896 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+39], v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+39] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+36], v[vgprValuB_X0_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+37], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+36], v202, v203 // __TF32_2_B_4 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:10  */
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:960 // L -> Reg lro=32 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
/*  mfmaIndex:11  */
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:8576 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:12  */
ds_read_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:8640 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:13  */
ds_read_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:17024 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
/*  mfmaIndex:14  */
ds_read_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:17088 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:15  */
ds_read_b128 v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprLocalReadAddrB+0] offset:25472 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:16  */
ds_read_b128 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[vgprLocalReadAddrB+0] offset:25536 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
/*  mfmaIndex:17  */
ds_read_b128 v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprLocalReadAddrB+0] offset:33920 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:18  */
ds_read_b128 v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3], v[vgprLocalReadAddrB+0] offset:33984 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:24  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:25  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
/*  mfmaIndex:26  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:27  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:28  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
/*  mfmaIndex:29  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:30  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:31  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
/*  mfmaIndex:32  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:33  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:34  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:36  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:37  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
/*  mfmaIndex:38  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:39  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:40  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
/*  mfmaIndex:41  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:42  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:43  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
/*  mfmaIndex:44  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:46  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
/*  mfmaIndex:47  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:48  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[64:67] // src0_h*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:49  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[64:67] // src0_h*src1_l, left value = acc[64+0:67+0]
/*  mfmaIndex:50  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[64:67] // src0_l*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:51  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[68:71] // src0_h*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:52  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[68:71] // src0_h*src1_l, left value = acc[68+0:71+0]
/*  mfmaIndex:53  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[68:71] // src0_l*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:54  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[72:75] // src0_h*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:55  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[72:75] // src0_h*src1_l, left value = acc[72+0:75+0]
/*  mfmaIndex:56  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[72:75] // src0_l*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:57  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[76:79] // src0_h*src1_h, left value = acc[76+0:79+0]
/*  mfmaIndex:58  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[76:79] // src0_h*src1_l, left value = acc[76+0:79+0]
/*  mfmaIndex:59  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[76:79] // src0_l*src1_h, left value = acc[76+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=10 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:97, lwEndMfmaIndex:97  */
/*  numMfmaForLR:22, syncPlrMfmaIndex:97 , sync1LdsMfmaIndex:96 */
/*  mfmaIndex:60  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+0], v[vgprValuA_T1_I0+0], v[vgprValuA_T1_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+1], v[vgprValuA_T1_I0+2], v[vgprValuA_T1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3], v[166:167], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[166:167], v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+1], v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+3], v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[166:167], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:61  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+7], v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+5], v[vgprValuA_T1_I0+2], v[vgprValuA_T1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+4], v[vgprValuA_T1_I0+0], v[vgprValuA_T1_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+7], v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+5], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+4], v202, v203 // __TF32_2_B_0 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
/*  mfmaIndex:62  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+8], v[vgprValuA_T1_I0+4], v[vgprValuA_T1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+9], v[vgprValuA_T1_I0+6], v[vgprValuA_T1_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+10], v[vgprValuA_X1_I0+12], v[vgprValuA_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+11], v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3], v[166:167], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[166:167], v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+10:vgprValuB_X1_I0+10+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+8], v[vgprValuB_X1_I0+8], v[vgprValuB_X1_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+9], v[vgprValuB_X1_I0+10], v[vgprValuB_X1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+10], v[vgprValuB_X1_I0+12], v[vgprValuB_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+11], v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[166:167], v[vgprValuB_X1_I0+10:vgprValuB_X1_I0+10+1], v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
/*  mfmaIndex:63  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+15], v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+12], v[vgprValuA_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+13], v[vgprValuA_T1_I0+6], v[vgprValuA_T1_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+12], v[vgprValuA_T1_I0+4], v[vgprValuA_T1_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+15], v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+12], v[vgprValuB_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+13], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+12], v202, v203 // __TF32_2_B_1 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:64  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+16], v[vgprValuA_T1_I0+8], v[vgprValuA_T1_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+17], v[vgprValuA_T1_I0+10], v[vgprValuA_T1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+18], v[vgprValuA_X1_I0+20], v[vgprValuA_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+19], v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3], v[166:167], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+1], v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[166:167], v[vgprValuA_X1_I0+18:vgprValuA_X1_I0+18+1], v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+18:vgprValuB_X1_I0+18+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+16], v[vgprValuB_X1_I0+16], v[vgprValuB_X1_I0+17]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+17], v[vgprValuB_X1_I0+18], v[vgprValuB_X1_I0+19]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+18], v[vgprValuB_X1_I0+20], v[vgprValuB_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+19], v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[166:167], v[vgprValuB_X1_I0+18:vgprValuB_X1_I0+18+1], v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
/*  mfmaIndex:65  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+23], v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+20], v[vgprValuA_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+21], v[vgprValuA_T1_I0+10], v[vgprValuA_T1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+20], v[vgprValuA_T1_I0+8], v[vgprValuA_T1_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+23], v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+20], v[vgprValuB_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+21], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+20], v202, v203 // __TF32_2_B_2 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
/*  mfmaIndex:66  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+24], v[vgprValuA_T1_I0+12], v[vgprValuA_T1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+25], v[vgprValuA_T1_I0+14], v[vgprValuA_T1_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+26], v[vgprValuA_X1_I0+28], v[vgprValuA_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+27], v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3], v[166:167], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+1], v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[166:167], v[vgprValuA_X1_I0+26:vgprValuA_X1_I0+26+1], v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+26:vgprValuB_X1_I0+26+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+24], v[vgprValuB_X1_I0+24], v[vgprValuB_X1_I0+25]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+25], v[vgprValuB_X1_I0+26], v[vgprValuB_X1_I0+27]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+26], v[vgprValuB_X1_I0+28], v[vgprValuB_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+27], v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[166:167], v[vgprValuB_X1_I0+26:vgprValuB_X1_I0+26+1], v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:67  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+31], v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+28], v[vgprValuA_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+29], v[vgprValuA_T1_I0+14], v[vgprValuA_T1_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+28], v[vgprValuA_T1_I0+12], v[vgprValuA_T1_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+31], v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+28], v[vgprValuB_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+29], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+28], v202, v203 // __TF32_2_B_3 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
/*  mfmaIndex:68  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+34:vgprValuB_X1_I0+34+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+32], v[vgprValuB_X1_I0+32], v[vgprValuB_X1_I0+33]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+33], v[vgprValuB_X1_I0+34], v[vgprValuB_X1_I0+35]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+34], v[vgprValuB_X1_I0+36], v[vgprValuB_X1_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+35], v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+39]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3], v[166:167], v[vgprValuB_X1_I0+34:vgprValuB_X1_I0+34+1], v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3] // Calculate low bits for TF32 emulation__TF32_1_B_4: 
s_nop 4                                            // nop for x32f emulation
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
/*  mfmaIndex:69  */
/* pack scheduling: packAIdx:0, packBIdx:0 */
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+39], v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+39] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+36], v[vgprValuB_X1_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+37], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+36], v202, v203 // __TF32_2_B_4 pack final end
s_nop 0                                            // VALU packing writes to be consumed by matrix instruction (HACK)
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:70  */
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
/*  mfmaIndex:71  */
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
/*  mfmaIndex:72  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:73  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
/*  mfmaIndex:74  */
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
/*  mfmaIndex:75  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:76  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
/*  mfmaIndex:77  */
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
/*  mfmaIndex:78  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:79  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
/*  mfmaIndex:80  */
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
/*  mfmaIndex:81  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:82  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
/*  mfmaIndex:83  */
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
/*  mfmaIndex:84  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:85  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
/*  mfmaIndex:86  */
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
/*  mfmaIndex:87  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:88  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
/*  mfmaIndex:89  */
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
/*  mfmaIndex:90  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:91  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
/*  mfmaIndex:92  */
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
/*  mfmaIndex:93  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:94  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
/*  mfmaIndex:95  */
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
/*  mfmaIndex:96  */
/* schedule remaining localreads for one buffer scheduling */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:97  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
/*  mfmaIndex:98  */
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
/*  mfmaIndex:99  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:100  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
/*  mfmaIndex:101  */
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
/*  mfmaIndex:102  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:103  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
/*  mfmaIndex:104  */
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
/*  mfmaIndex:105  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:106  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
/*  mfmaIndex:107  */
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]
/*  mfmaIndex:108  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[64:67] // src0_h*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:109  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[64:67] // src0_h*src1_l, left value = acc[64+0:67+0]
/*  mfmaIndex:110  */
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[64:67] // src0_l*src1_h, left value = acc[64+0:67+0]
/*  mfmaIndex:111  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[68:71] // src0_h*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:112  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[68:71] // src0_h*src1_l, left value = acc[68+0:71+0]
/*  mfmaIndex:113  */
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[68:71] // src0_l*src1_h, left value = acc[68+0:71+0]
/*  mfmaIndex:114  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[72:75] // src0_h*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:115  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[72:75] // src0_h*src1_l, left value = acc[72+0:75+0]
/*  mfmaIndex:116  */
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[72:75] // src0_l*src1_h, left value = acc[72+0:75+0]
/*  mfmaIndex:117  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[76:79] // src0_h*src1_h, left value = acc[76+0:79+0]
/*  mfmaIndex:118  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[76:79] // src0_h*src1_l, left value = acc[76+0:79+0]
/*  mfmaIndex:119  */
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[76:79] // src0_l*src1_h, left value = acc[76+0:79+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=10 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [20...164) to pool */

/* Tail: add address/G2L vgpr [164...164) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */
s_xor_b32 s84, s[sgprSwapA], s[sgprLocalWriteAddrA] // Get other lds buffer offset value
s_min_u32 s[sgprLocalWriteAddrA], s[sgprLocalWriteAddrA], s84 // Set LWA to first buffer offset

/* local write reset offsets b */
s_xor_b32 s84, s[sgprSwapB], s[sgprLocalWriteAddrB] // Get other lds buffer offset value
s_min_u32 s[sgprLocalWriteAddrB], s[sgprLocalWriteAddrB], s84 // Set LWA to first buffer offset
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (32,40,0) */
.set vgprG2LA_BASE, 20
.set vgprG2LB_BASE, 52

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 63, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 64
s_cmp_lt_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Check if WG processes final iteration of tile
s_cmov_b32 s[sgprLoopCounterL], 0                  // This WG not completing tile
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
//  removeStagger A
s_sub_i32 s84, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s84, 0
s_cbranch_scc0 label_Negative_J5DQFVGFWLXU2DUR
s_mul_hi_u32 s85, s84, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s84, s84, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_DLSAQLEVYLOBCPNL
label_Negative_J5DQFVGFWLXU2DUR:
s_abs_i32 s84, s84
s_mul_hi_u32 s85, s84, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s84, s84, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s84, s84, 0xffffffff
s_xor_b32 s85, s85, 0xffffffff
s_add_u32 s84, s84, 0x1
s_addc_u32 s85, s85, 0
label_MultiplyDone_DLSAQLEVYLOBCPNL:
s_sub_u32 s84, s84, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s85, s85, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s84        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s85       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s84 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s85 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
//  removeStagger B
s_sub_i32 s84, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s84, 0
s_cbranch_scc0 label_Negative_LQI6BOBE0EY8XIP1
s_mul_hi_u32 s85, s84, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s84, s84, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_9N1QELR2XL4Z0HRB
label_Negative_LQI6BOBE0EY8XIP1:
s_abs_i32 s84, s84
s_mul_hi_u32 s85, s84, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s84, s84, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s84, s84, 0xffffffff
s_xor_b32 s85, s85, 0xffffffff
s_add_u32 s84, s84, 0x1
s_addc_u32 s85, s85, 0
label_MultiplyDone_9N1QELR2XL4Z0HRB:
s_sub_u32 s84, s84, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s85, s85, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s84        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s85       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s84 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s85 // limit -= inc)
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
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+4], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_4_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+5], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_5_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+6], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_6_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetA+7], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 lds // G -> Reg 0_0_7_0

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
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_4_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_5_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_6_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_7_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_8_0
s_add_u32 m0, m0, 4224                             // Move LDS write address to next line
buffer_load_dwordx4 v[vgprGlobalReadOffsetB+9], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 lds // G -> Reg 0_0_9_0
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip force waitcnt0
s_barrier

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip force waitcnt0
s_barrier                                          // Tail loop LW->LR, sync LDS0
.set vgprG2LA_BASE, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprValuA_X0_I0_BASE, 20
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+32
.set vgprValuA_T0_I0, 168
.set vgprValuA_T1_I0, 184
.set vgprValuB_X0_I0_BASE, 84
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+40
.set IdentityMatrix, 166

/* Tail: local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */
v_xor_b32 v201, v[vgprLocalReadSwapAddrA], v[vgprLocalReadAddrA] // Get other lds buffer offset value
v_min_i32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA], v201 // Set LRA to first buffer offset

/* Tail: local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_xor_b32 v201, v[vgprLocalReadSwapAddrB], v[vgprLocalReadAddrB] // Get other lds buffer offset value
v_min_i32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB], v201 // Set LRA to first buffer offset

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* tail loop unroll iter 0 */

/* local read a */
ds_read_b128 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA+0] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA+0] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:8448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:8512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:16896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:16960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:25344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:25408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprLocalReadAddrB+0] offset:33792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[vgprLocalReadAddrB+0] offset:33856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
s_mov_b32 s84, 128                                 // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s84, v[vgprLocalReadAddrA+0] // lrA += 128 (bpeDS)

/* local read inc b */
                                                   // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s84, v[vgprLocalReadAddrB+0] // lrB += 128 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_and_b32 v201, 63, v[vgprSerial]                  // v201 = v[vgprSerial] % 64
v_lshrrev_b32 v201, 4, v201                        // 201 = 201 / 16
v_lshlrev_b32 v201, 2, v201                        // v201 = v201 * 4
v_add_u32 v202, v201, 0
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+0+0+0], v[vgprValuA_T0_I0+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+4+0+0], v[vgprValuA_T0_I0+4+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+8+0+0], v[vgprValuA_T0_I0+8+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+12+0+0], v[vgprValuA_T0_I0+12+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+1+0+0], v[vgprValuA_T0_I0+1+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+5+0+0], v[vgprValuA_T0_I0+5+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+9+0+0], v[vgprValuA_T0_I0+9+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+13+0+0], v[vgprValuA_T0_I0+13+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+2+0+0], v[vgprValuA_T0_I0+2+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+6+0+0], v[vgprValuA_T0_I0+6+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+10+0+0], v[vgprValuA_T0_I0+10+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+14+0+0], v[vgprValuA_T0_I0+14+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+3+0+0], v[vgprValuA_T0_I0+3+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+7+0+0], v[vgprValuA_T0_I0+7+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+11+0+0], v[vgprValuA_T0_I0+11+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T0_I0+15+0+0], v[vgprValuA_T0_I0+15+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0], v[vgprValuA_X0_I0+4+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0], v[vgprValuA_X0_I0+12+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0], v[vgprValuA_X0_I0+20+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0], v[vgprValuA_X0_I0+28+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+5+0+0], v[vgprValuA_X0_I0+5+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+13+0+0], v[vgprValuA_X0_I0+13+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+21+0+0], v[vgprValuA_X0_I0+21+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+29+0+0], v[vgprValuA_X0_I0+29+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+6+0+0], v[vgprValuA_X0_I0+6+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+14+0+0], v[vgprValuA_X0_I0+14+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+22+0+0], v[vgprValuA_X0_I0+22+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+30+0+0], v[vgprValuA_X0_I0+30+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+7+0+0], v[vgprValuA_X0_I0+7+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+15+0+0], v[vgprValuA_X0_I0+15+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+23+0+0], v[vgprValuA_X0_I0+23+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+31+0+0], v[vgprValuA_X0_I0+31+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_and_b32 v201, 63, v[vgprSerial]                  // v201 = v[vgprSerial] % 64
v_lshrrev_b32 v201, 4, v201                        // 201 = 201 / 16
v_lshlrev_b32 v201, 2, v201                        // v201 = v201 * 4
v_add_u32 v202, v201, 0
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+0], v[vgprValuB_X0_I0+8+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+0], v[vgprValuB_X0_I0+16+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+0], v[vgprValuB_X0_I0+24+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+0], v[vgprValuB_X0_I0+32+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+1], v[vgprValuB_X0_I0+8+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+1], v[vgprValuB_X0_I0+16+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+1], v[vgprValuB_X0_I0+24+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+1], v[vgprValuB_X0_I0+32+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+2], v[vgprValuB_X0_I0+8+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+2], v[vgprValuB_X0_I0+16+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+2], v[vgprValuB_X0_I0+24+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+2], v[vgprValuB_X0_I0+32+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+3], v[vgprValuB_X0_I0+8+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+3], v[vgprValuB_X0_I0+16+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+3], v[vgprValuB_X0_I0+24+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+3], v[vgprValuB_X0_I0+32+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+4], v[vgprValuB_X0_I0+0+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+4], v[vgprValuB_X0_I0+8+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+4], v[vgprValuB_X0_I0+16+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+4], v[vgprValuB_X0_I0+24+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+4], v[vgprValuB_X0_I0+32+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+5], v[vgprValuB_X0_I0+0+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+5], v[vgprValuB_X0_I0+8+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+5], v[vgprValuB_X0_I0+16+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+5], v[vgprValuB_X0_I0+24+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+5], v[vgprValuB_X0_I0+32+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+6], v[vgprValuB_X0_I0+0+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+6], v[vgprValuB_X0_I0+8+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+6], v[vgprValuB_X0_I0+16+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+6], v[vgprValuB_X0_I0+24+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+6], v[vgprValuB_X0_I0+32+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+8+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+16+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+24+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+7], v[vgprValuB_X0_I0+32+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
s_and_b32 s86, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s86, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_DZOUDPYJU2HHRCOQ // skip mask
s_and_b32 s86, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s86, 8, s86                              // use shift to fill 0 for outside element
s_lshl_b32 s86, s86, 5                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[204:205], s86, v[vgprValuA_T0_I0+0+0+0:vgprValuA_T0_I0+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuA_T0_I0+2+0+0:vgprValuA_T0_I0+2+0+0+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+0+0+0], v[vgprValuA_T0_I0+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+1+0+0], v[vgprValuA_T0_I0+1+0+0], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+2+0+0], v[vgprValuA_T0_I0+2+0+0], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+3+0+0], v[vgprValuA_T0_I0+3+0+0], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0], v[vgprValuA_X0_I0+4+0+0], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+5+0+0], v[vgprValuA_X0_I0+5+0+0], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+6+0+0], v[vgprValuA_X0_I0+6+0+0], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+7+0+0], v[vgprValuA_X0_I0+7+0+0], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuA_T0_I0+4+0+0:vgprValuA_T0_I0+4+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuA_T0_I0+6+0+0:vgprValuA_T0_I0+6+0+0+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuA_X0_I0+12+0+0:vgprValuA_X0_I0+12+0+0+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuA_X0_I0+14+0+0:vgprValuA_X0_I0+14+0+0+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+4+0+0], v[vgprValuA_T0_I0+4+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+5+0+0], v[vgprValuA_T0_I0+5+0+0], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+6+0+0], v[vgprValuA_T0_I0+6+0+0], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+7+0+0], v[vgprValuA_T0_I0+7+0+0], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+12+0+0], v[vgprValuA_X0_I0+12+0+0], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+13+0+0], v[vgprValuA_X0_I0+13+0+0], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+14+0+0], v[vgprValuA_X0_I0+14+0+0], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+15+0+0], v[vgprValuA_X0_I0+15+0+0], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuA_T0_I0+8+0+0:vgprValuA_T0_I0+8+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuA_T0_I0+10+0+0:vgprValuA_T0_I0+10+0+0+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuA_X0_I0+20+0+0:vgprValuA_X0_I0+20+0+0+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuA_X0_I0+22+0+0:vgprValuA_X0_I0+22+0+0+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+8+0+0], v[vgprValuA_T0_I0+8+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+9+0+0], v[vgprValuA_T0_I0+9+0+0], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+10+0+0], v[vgprValuA_T0_I0+10+0+0], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+11+0+0], v[vgprValuA_T0_I0+11+0+0], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+20+0+0], v[vgprValuA_X0_I0+20+0+0], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+21+0+0], v[vgprValuA_X0_I0+21+0+0], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+22+0+0], v[vgprValuA_X0_I0+22+0+0], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+23+0+0], v[vgprValuA_X0_I0+23+0+0], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuA_T0_I0+12+0+0:vgprValuA_T0_I0+12+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuA_T0_I0+14+0+0:vgprValuA_T0_I0+14+0+0+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuA_X0_I0+28+0+0:vgprValuA_X0_I0+28+0+0+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuA_X0_I0+30+0+0:vgprValuA_X0_I0+30+0+0+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+12+0+0], v[vgprValuA_T0_I0+12+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+13+0+0], v[vgprValuA_T0_I0+13+0+0], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+14+0+0], v[vgprValuA_T0_I0+14+0+0], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T0_I0+15+0+0], v[vgprValuA_T0_I0+15+0+0], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+28+0+0], v[vgprValuA_X0_I0+28+0+0], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+29+0+0], v[vgprValuA_X0_I0+29+0+0], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+30+0+0], v[vgprValuA_X0_I0+30+0+0], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+31+0+0], v[vgprValuA_X0_I0+31+0+0], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X0_I0+0+0+0+0:vgprValuB_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X0_I0+0+0+0+4:vgprValuB_X0_I0+0+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X0_I0+0+0+0+6:vgprValuB_X0_I0+0+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+4], v[vgprValuB_X0_I0+0+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+5], v[vgprValuB_X0_I0+0+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+6], v[vgprValuB_X0_I0+0+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0+7], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X0_I0+8+0+0+0:vgprValuB_X0_I0+8+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X0_I0+8+0+0+2:vgprValuB_X0_I0+8+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X0_I0+8+0+0+4:vgprValuB_X0_I0+8+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X0_I0+8+0+0+6:vgprValuB_X0_I0+8+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+0], v[vgprValuB_X0_I0+8+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+1], v[vgprValuB_X0_I0+8+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+2], v[vgprValuB_X0_I0+8+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+3], v[vgprValuB_X0_I0+8+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+4], v[vgprValuB_X0_I0+8+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+5], v[vgprValuB_X0_I0+8+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+6], v[vgprValuB_X0_I0+8+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+8+0+0+7], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X0_I0+16+0+0+0:vgprValuB_X0_I0+16+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X0_I0+16+0+0+2:vgprValuB_X0_I0+16+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X0_I0+16+0+0+4:vgprValuB_X0_I0+16+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X0_I0+16+0+0+6:vgprValuB_X0_I0+16+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+0], v[vgprValuB_X0_I0+16+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+1], v[vgprValuB_X0_I0+16+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+2], v[vgprValuB_X0_I0+16+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+3], v[vgprValuB_X0_I0+16+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+4], v[vgprValuB_X0_I0+16+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+5], v[vgprValuB_X0_I0+16+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+6], v[vgprValuB_X0_I0+16+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+16+0+0+7], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X0_I0+24+0+0+0:vgprValuB_X0_I0+24+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X0_I0+24+0+0+2:vgprValuB_X0_I0+24+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X0_I0+24+0+0+4:vgprValuB_X0_I0+24+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X0_I0+24+0+0+6:vgprValuB_X0_I0+24+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+0], v[vgprValuB_X0_I0+24+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+1], v[vgprValuB_X0_I0+24+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+2], v[vgprValuB_X0_I0+24+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+3], v[vgprValuB_X0_I0+24+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+4], v[vgprValuB_X0_I0+24+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+5], v[vgprValuB_X0_I0+24+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+6], v[vgprValuB_X0_I0+24+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+24+0+0+7], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X0_I0+32+0+0+0:vgprValuB_X0_I0+32+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X0_I0+32+0+0+2:vgprValuB_X0_I0+32+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X0_I0+32+0+0+4:vgprValuB_X0_I0+32+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X0_I0+32+0+0+6:vgprValuB_X0_I0+32+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+0], v[vgprValuB_X0_I0+32+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+1], v[vgprValuB_X0_I0+32+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+2], v[vgprValuB_X0_I0+32+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+3], v[vgprValuB_X0_I0+32+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+4], v[vgprValuB_X0_I0+32+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+5], v[vgprValuB_X0_I0+32+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+6], v[vgprValuB_X0_I0+32+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+32+0+0+7], v[vgprValuB_X0_I0+32+0+0+7], v211, s[84:85]
label_TailLoop_SkipZeroOutMask_DZOUDPYJU2HHRCOQ:
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+1], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3], v[166:167], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[166:167], v[vgprValuA_X0_I0+2:vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[166:167], v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+5], v[vgprValuA_T0_I0+2], v[vgprValuA_T0_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+4], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+5], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+4], v202, v203 // __TF32_2_B_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+8], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+9], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+10], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+11], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3], v[166:167], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+1], v[vgprValuA_T0_I0+4:vgprValuA_T0_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[166:167], v[vgprValuA_X0_I0+10:vgprValuA_X0_I0+10+1], v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+8], v[vgprValuB_X0_I0+8], v[vgprValuB_X0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+9], v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+10], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+11], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[166:167], v[vgprValuB_X0_I0+10:vgprValuB_X0_I0+10+1], v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+12], v[vgprValuA_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+13], v[vgprValuA_T0_I0+6], v[vgprValuA_T0_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+12], v[vgprValuA_T0_I0+4], v[vgprValuA_T0_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+15], v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+14], v[vgprValuB_X0_I0+12], v[vgprValuB_X0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+13], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+12], v202, v203 // __TF32_2_B_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+16], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+17], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+18], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+19], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3], v[166:167], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+1], v[vgprValuA_T0_I0+8:vgprValuA_T0_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[166:167], v[vgprValuA_X0_I0+18:vgprValuA_X0_I0+18+1], v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+16], v[vgprValuB_X0_I0+16], v[vgprValuB_X0_I0+17]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+17], v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+19]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+18], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+19], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[166:167], v[vgprValuB_X0_I0+18:vgprValuB_X0_I0+18+1], v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+23], v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+22], v[vgprValuA_X0_I0+20], v[vgprValuA_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+21], v[vgprValuA_T0_I0+10], v[vgprValuA_T0_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+20], v[vgprValuA_T0_I0+8], v[vgprValuA_T0_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+23], v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+22], v[vgprValuB_X0_I0+20], v[vgprValuB_X0_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+21], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+20], v202, v203 // __TF32_2_B_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+24], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+25], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+26], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+27], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3], v[166:167], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+1], v[vgprValuA_T0_I0+12:vgprValuA_T0_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[166:167], v[vgprValuA_X0_I0+26:vgprValuA_X0_I0+26+1], v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+24], v[vgprValuB_X0_I0+24], v[vgprValuB_X0_I0+25]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+25], v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+27]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+26], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+27], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[166:167], v[vgprValuB_X0_I0+26:vgprValuB_X0_I0+26+1], v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+31], v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+30], v[vgprValuA_X0_I0+28], v[vgprValuA_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+29], v[vgprValuA_T0_I0+14], v[vgprValuA_T0_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+28], v[vgprValuA_T0_I0+12], v[vgprValuA_T0_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+31], v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+30], v[vgprValuB_X0_I0+28], v[vgprValuB_X0_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+29], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+28], v202, v203 // __TF32_2_B_3 pack final end
v_mov_b64 v[202:203], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+1]
v_mov_b64 v[204:205], v[vgprValuB_X0_I0+34:vgprValuB_X0_I0+34+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+32], v[vgprValuB_X0_I0+32], v[vgprValuB_X0_I0+33]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+33], v[vgprValuB_X0_I0+34], v[vgprValuB_X0_I0+35]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+34], v[vgprValuB_X0_I0+36], v[vgprValuB_X0_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+35], v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+39]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3], v[166:167], v[vgprValuB_X0_I0+34:vgprValuB_X0_I0+34+1], v[vgprValuB_X0_I0+36:vgprValuB_X0_I0+36+3] // Calculate low bits for TF32 emulation__TF32_1_B_4: 
s_nop 4                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+39], v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+39] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+38], v[vgprValuB_X0_I0+36], v[vgprValuB_X0_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+37], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X0_I0+36], v202, v203 // __TF32_2_B_4 pack final end
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
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[64:67] // src0_h*src1_h, left value = acc[64+0:67+0]
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+3], acc[64:67] // src0_h*src1_l, left value = acc[64+0:67+0]
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], acc[64:67] // src0_l*src1_h, left value = acc[64+0:67+0]
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[68:71] // src0_h*src1_h, left value = acc[68+0:71+0]
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+3], acc[68:71] // src0_h*src1_l, left value = acc[68+0:71+0]
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], acc[68:71] // src0_l*src1_h, left value = acc[68+0:71+0]
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[72:75] // src0_h*src1_h, left value = acc[72+0:75+0]
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+3], acc[72:75] // src0_h*src1_l, left value = acc[72+0:75+0]
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], acc[72:75] // src0_l*src1_h, left value = acc[72+0:75+0]
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[76:79] // src0_h*src1_h, left value = acc[76+0:79+0]
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32:vgprValuB_X0_I0+32+3], v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+3], acc[76:79] // src0_h*src1_l, left value = acc[76+0:79+0]
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X0_I0+32+4:vgprValuB_X0_I0+32+4+3], v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], acc[76:79] // src0_l*src1_h, left value = acc[76+0:79+0]

/* closeLoop loopL finalLoop=0 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc1 label_TailLoopEndL                  // exit LoopL

/* tail loop unroll iter 1 */

/* local read a */
ds_read_b128 v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[vgprLocalReadAddrA+0] offset:576 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[vgprLocalReadAddrA+0] offset:832 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read b */
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprLocalReadAddrB+0] offset:8448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[vgprLocalReadAddrB+0] offset:8512 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprLocalReadAddrB+0] offset:16896 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[vgprLocalReadAddrB+0] offset:16960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprLocalReadAddrB+0] offset:25344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[vgprLocalReadAddrB+0] offset:25408 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprLocalReadAddrB+0] offset:33792 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3], v[vgprLocalReadAddrB+0] offset:33856 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* local read inc a */
s_mov_b32 s84, 128                                 // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s84, v[vgprLocalReadAddrA+0] // lrA += 128 (bpeDS)

/* local read inc b */
                                                   // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s84, v[vgprLocalReadAddrB+0] // lrB += 128 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_and_b32 v201, 63, v[vgprSerial]                  // v201 = v[vgprSerial] % 64
v_lshrrev_b32 v201, 4, v201                        // 201 = 201 / 16
v_lshlrev_b32 v201, 2, v201                        // v201 = v201 * 4
v_add_u32 v202, v201, 0
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+0+0+0], v[vgprValuA_T1_I0+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+4+0+0], v[vgprValuA_T1_I0+4+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+8+0+0], v[vgprValuA_T1_I0+8+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+12+0+0], v[vgprValuA_T1_I0+12+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+1+0+0], v[vgprValuA_T1_I0+1+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+5+0+0], v[vgprValuA_T1_I0+5+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+9+0+0], v[vgprValuA_T1_I0+9+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+13+0+0], v[vgprValuA_T1_I0+13+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+2+0+0], v[vgprValuA_T1_I0+2+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+6+0+0], v[vgprValuA_T1_I0+6+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+10+0+0], v[vgprValuA_T1_I0+10+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+14+0+0], v[vgprValuA_T1_I0+14+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+3+0+0], v[vgprValuA_T1_I0+3+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+7+0+0], v[vgprValuA_T1_I0+7+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+11+0+0], v[vgprValuA_T1_I0+11+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_T1_I0+15+0+0], v[vgprValuA_T1_I0+15+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0], v[vgprValuA_X1_I0+4+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0], v[vgprValuA_X1_I0+12+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+20+0+0], v[vgprValuA_X1_I0+20+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+28+0+0], v[vgprValuA_X1_I0+28+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+5+0+0], v[vgprValuA_X1_I0+5+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+13+0+0], v[vgprValuA_X1_I0+13+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+21+0+0], v[vgprValuA_X1_I0+21+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+29+0+0], v[vgprValuA_X1_I0+29+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+6+0+0], v[vgprValuA_X1_I0+6+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+14+0+0], v[vgprValuA_X1_I0+14+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+22+0+0], v[vgprValuA_X1_I0+22+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+30+0+0], v[vgprValuA_X1_I0+30+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+7+0+0], v[vgprValuA_X1_I0+7+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+15+0+0], v[vgprValuA_X1_I0+15+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+23+0+0], v[vgprValuA_X1_I0+23+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X1_I0+31+0+0], v[vgprValuA_X1_I0+31+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_and_b32 v201, 63, v[vgprSerial]                  // v201 = v[vgprSerial] % 64
v_lshrrev_b32 v201, 4, v201                        // 201 = 201 / 16
v_lshlrev_b32 v201, 2, v201                        // v201 = v201 * 4
v_add_u32 v202, v201, 0
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+0], v[vgprValuB_X1_I0+0+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+0], v[vgprValuB_X1_I0+8+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+0], v[vgprValuB_X1_I0+16+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+0], v[vgprValuB_X1_I0+24+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+0], v[vgprValuB_X1_I0+32+0+0+0], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+1], v[vgprValuB_X1_I0+0+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+1], v[vgprValuB_X1_I0+8+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+1], v[vgprValuB_X1_I0+16+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+1], v[vgprValuB_X1_I0+24+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+1], v[vgprValuB_X1_I0+32+0+0+1], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+2], v[vgprValuB_X1_I0+0+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+2], v[vgprValuB_X1_I0+8+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+2], v[vgprValuB_X1_I0+16+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+2], v[vgprValuB_X1_I0+24+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+2], v[vgprValuB_X1_I0+32+0+0+2], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+3], v[vgprValuB_X1_I0+0+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+3], v[vgprValuB_X1_I0+8+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+3], v[vgprValuB_X1_I0+16+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+3], v[vgprValuB_X1_I0+24+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+3], v[vgprValuB_X1_I0+32+0+0+3], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+4], v[vgprValuB_X1_I0+0+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+4], v[vgprValuB_X1_I0+8+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+4], v[vgprValuB_X1_I0+16+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+4], v[vgprValuB_X1_I0+24+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+4], v[vgprValuB_X1_I0+32+0+0+4], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+5], v[vgprValuB_X1_I0+0+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+5], v[vgprValuB_X1_I0+8+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+5], v[vgprValuB_X1_I0+16+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+5], v[vgprValuB_X1_I0+24+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+5], v[vgprValuB_X1_I0+32+0+0+5], 0, s[84:85] // set 0 if K_idx >= sizeL
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+6], v[vgprValuB_X1_I0+0+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+6], v[vgprValuB_X1_I0+8+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+6], v[vgprValuB_X1_I0+16+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+6], v[vgprValuB_X1_I0+24+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+6], v[vgprValuB_X1_I0+32+0+0+6], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+7], v[vgprValuB_X1_I0+8+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+7], v[vgprValuB_X1_I0+16+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+7], v[vgprValuB_X1_I0+24+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+7], v[vgprValuB_X1_I0+32+0+0+7], 0, s[84:85] // set 0 if K_idx >= sizeL
s_and_b32 s86, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s86, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_QWMA7J3AUDGL0X23 // skip mask
s_and_b32 s86, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s86, 8, s86                              // use shift to fill 0 for outside element
s_lshl_b32 s86, s86, 5                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[204:205], s86, v[vgprValuA_T1_I0+0+0+0:vgprValuA_T1_I0+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuA_T1_I0+2+0+0:vgprValuA_T1_I0+2+0+0+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+0+0+0], v[vgprValuA_T1_I0+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+1+0+0], v[vgprValuA_T1_I0+1+0+0], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+2+0+0], v[vgprValuA_T1_I0+2+0+0], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+3+0+0], v[vgprValuA_T1_I0+3+0+0], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+4+0+0], v[vgprValuA_X1_I0+4+0+0], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+5+0+0], v[vgprValuA_X1_I0+5+0+0], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+6+0+0], v[vgprValuA_X1_I0+6+0+0], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+7+0+0], v[vgprValuA_X1_I0+7+0+0], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuA_T1_I0+4+0+0:vgprValuA_T1_I0+4+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuA_T1_I0+6+0+0:vgprValuA_T1_I0+6+0+0+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuA_X1_I0+12+0+0:vgprValuA_X1_I0+12+0+0+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuA_X1_I0+14+0+0:vgprValuA_X1_I0+14+0+0+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+4+0+0], v[vgprValuA_T1_I0+4+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+5+0+0], v[vgprValuA_T1_I0+5+0+0], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+6+0+0], v[vgprValuA_T1_I0+6+0+0], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+7+0+0], v[vgprValuA_T1_I0+7+0+0], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+12+0+0], v[vgprValuA_X1_I0+12+0+0], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+13+0+0], v[vgprValuA_X1_I0+13+0+0], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+14+0+0], v[vgprValuA_X1_I0+14+0+0], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+15+0+0], v[vgprValuA_X1_I0+15+0+0], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuA_T1_I0+8+0+0:vgprValuA_T1_I0+8+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuA_T1_I0+10+0+0:vgprValuA_T1_I0+10+0+0+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuA_X1_I0+20+0+0:vgprValuA_X1_I0+20+0+0+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuA_X1_I0+22+0+0:vgprValuA_X1_I0+22+0+0+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+8+0+0], v[vgprValuA_T1_I0+8+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+9+0+0], v[vgprValuA_T1_I0+9+0+0], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+10+0+0], v[vgprValuA_T1_I0+10+0+0], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+11+0+0], v[vgprValuA_T1_I0+11+0+0], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+20+0+0], v[vgprValuA_X1_I0+20+0+0], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+21+0+0], v[vgprValuA_X1_I0+21+0+0], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+22+0+0], v[vgprValuA_X1_I0+22+0+0], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+23+0+0], v[vgprValuA_X1_I0+23+0+0], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuA_T1_I0+12+0+0:vgprValuA_T1_I0+12+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuA_T1_I0+14+0+0:vgprValuA_T1_I0+14+0+0+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuA_X1_I0+28+0+0:vgprValuA_X1_I0+28+0+0+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuA_X1_I0+30+0+0:vgprValuA_X1_I0+30+0+0+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+12+0+0], v[vgprValuA_T1_I0+12+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+13+0+0], v[vgprValuA_T1_I0+13+0+0], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+14+0+0], v[vgprValuA_T1_I0+14+0+0], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_T1_I0+15+0+0], v[vgprValuA_T1_I0+15+0+0], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+28+0+0], v[vgprValuA_X1_I0+28+0+0], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+29+0+0], v[vgprValuA_X1_I0+29+0+0], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+30+0+0], v[vgprValuA_X1_I0+30+0+0], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X1_I0+31+0+0], v[vgprValuA_X1_I0+31+0+0], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X1_I0+0+0+0+0:vgprValuB_X1_I0+0+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X1_I0+0+0+0+2:vgprValuB_X1_I0+0+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X1_I0+0+0+0+4:vgprValuB_X1_I0+0+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X1_I0+0+0+0+6:vgprValuB_X1_I0+0+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+0], v[vgprValuB_X1_I0+0+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+1], v[vgprValuB_X1_I0+0+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+2], v[vgprValuB_X1_I0+0+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+3], v[vgprValuB_X1_I0+0+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+4], v[vgprValuB_X1_I0+0+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+5], v[vgprValuB_X1_I0+0+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+6], v[vgprValuB_X1_I0+0+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0+7], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X1_I0+8+0+0+0:vgprValuB_X1_I0+8+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X1_I0+8+0+0+2:vgprValuB_X1_I0+8+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X1_I0+8+0+0+4:vgprValuB_X1_I0+8+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X1_I0+8+0+0+6:vgprValuB_X1_I0+8+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+0], v[vgprValuB_X1_I0+8+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+1], v[vgprValuB_X1_I0+8+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+2], v[vgprValuB_X1_I0+8+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+3], v[vgprValuB_X1_I0+8+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+4], v[vgprValuB_X1_I0+8+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+5], v[vgprValuB_X1_I0+8+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+6], v[vgprValuB_X1_I0+8+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+8+0+0+7], v[vgprValuB_X1_I0+8+0+0+7], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X1_I0+16+0+0+0:vgprValuB_X1_I0+16+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X1_I0+16+0+0+2:vgprValuB_X1_I0+16+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X1_I0+16+0+0+4:vgprValuB_X1_I0+16+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X1_I0+16+0+0+6:vgprValuB_X1_I0+16+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+0], v[vgprValuB_X1_I0+16+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+1], v[vgprValuB_X1_I0+16+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+2], v[vgprValuB_X1_I0+16+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+3], v[vgprValuB_X1_I0+16+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+4], v[vgprValuB_X1_I0+16+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+5], v[vgprValuB_X1_I0+16+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+6], v[vgprValuB_X1_I0+16+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+16+0+0+7], v[vgprValuB_X1_I0+16+0+0+7], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X1_I0+24+0+0+0:vgprValuB_X1_I0+24+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X1_I0+24+0+0+2:vgprValuB_X1_I0+24+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X1_I0+24+0+0+4:vgprValuB_X1_I0+24+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X1_I0+24+0+0+6:vgprValuB_X1_I0+24+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+0], v[vgprValuB_X1_I0+24+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+1], v[vgprValuB_X1_I0+24+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+2], v[vgprValuB_X1_I0+24+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+3], v[vgprValuB_X1_I0+24+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+4], v[vgprValuB_X1_I0+24+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+5], v[vgprValuB_X1_I0+24+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+6], v[vgprValuB_X1_I0+24+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+24+0+0+7], v[vgprValuB_X1_I0+24+0+0+7], v211, s[84:85]
v_lshlrev_b64 v[204:205], s86, v[vgprValuB_X1_I0+32+0+0+0:vgprValuB_X1_I0+32+0+0+0+1]
v_lshlrev_b64 v[206:207], s86, v[vgprValuB_X1_I0+32+0+0+2:vgprValuB_X1_I0+32+0+0+2+1]
v_lshlrev_b64 v[208:209], s86, v[vgprValuB_X1_I0+32+0+0+4:vgprValuB_X1_I0+32+0+0+4+1]
v_lshlrev_b64 v[210:211], s86, v[vgprValuB_X1_I0+32+0+0+6:vgprValuB_X1_I0+32+0+0+6+1]
v_add_u32 v202, v201, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+0], v[vgprValuB_X1_I0+32+0+0+0], v204, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+1], v[vgprValuB_X1_I0+32+0+0+1], v205, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+2], v[vgprValuB_X1_I0+32+0+0+2], v206, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+3], v[vgprValuB_X1_I0+32+0+0+3], v207, s[84:85]
v_add_u32 v202, v202, 14                           // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+4], v[vgprValuB_X1_I0+32+0+0+4], v208, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+5], v[vgprValuB_X1_I0+32+0+0+5], v209, s[84:85]
v_add_u32 v202, v202, 2                            // add part of K
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+6], v[vgprValuB_X1_I0+32+0+0+6], v210, s[84:85]
v_cmp_ge_i32 s[84:85], v202, s[sgprLoopCounterL]   // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X1_I0+32+0+0+7], v[vgprValuB_X1_I0+32+0+0+7], v211, s[84:85]
label_TailLoop_SkipZeroOutMask_QWMA7J3AUDGL0X23:
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+0], v[vgprValuA_T1_I0+0], v[vgprValuA_T1_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+1], v[vgprValuA_T1_I0+2], v[vgprValuA_T1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3], v[166:167], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+1], v[vgprValuA_T1_I0+0:vgprValuA_T1_I0+0+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[166:167], v[vgprValuA_X1_I0+2:vgprValuA_X1_I0+2+1], v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_A_0: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+1], v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+3], v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+7]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[166:167], v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3] // Calculate low bits for TF32 emulation__TF32_1_B_0: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+7], v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+5], v[vgprValuA_T1_I0+2], v[vgprValuA_T1_I0+3]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+4], v[vgprValuA_T1_I0+0], v[vgprValuA_T1_I0+1] // __TF32_2_A_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+7], v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+7] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+5], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+4], v202, v203 // __TF32_2_B_0 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+8], v[vgprValuA_T1_I0+4], v[vgprValuA_T1_I0+5]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+9], v[vgprValuA_T1_I0+6], v[vgprValuA_T1_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+10], v[vgprValuA_X1_I0+12], v[vgprValuA_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+11], v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3], v[166:167], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+1], v[vgprValuA_T1_I0+4:vgprValuA_T1_I0+4+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3], v[166:167], v[vgprValuA_X1_I0+10:vgprValuA_X1_I0+10+1], v[vgprValuA_X1_I0+12:vgprValuA_X1_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_A_1: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+10:vgprValuB_X1_I0+10+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+8], v[vgprValuB_X1_I0+8], v[vgprValuB_X1_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+9], v[vgprValuB_X1_I0+10], v[vgprValuB_X1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+10], v[vgprValuB_X1_I0+12], v[vgprValuB_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+11], v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+15]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3], v[166:167], v[vgprValuB_X1_I0+10:vgprValuB_X1_I0+10+1], v[vgprValuB_X1_I0+12:vgprValuB_X1_I0+12+3] // Calculate low bits for TF32 emulation__TF32_1_B_1: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+15], v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+14], v[vgprValuA_X1_I0+12], v[vgprValuA_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+13], v[vgprValuA_T1_I0+6], v[vgprValuA_T1_I0+7]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+12], v[vgprValuA_T1_I0+4], v[vgprValuA_T1_I0+5] // __TF32_2_A_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+15], v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+15] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+14], v[vgprValuB_X1_I0+12], v[vgprValuB_X1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+13], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+12], v202, v203 // __TF32_2_B_1 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+16], v[vgprValuA_T1_I0+8], v[vgprValuA_T1_I0+9]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+17], v[vgprValuA_T1_I0+10], v[vgprValuA_T1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+18], v[vgprValuA_X1_I0+20], v[vgprValuA_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+19], v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3], v[166:167], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+1], v[vgprValuA_T1_I0+8:vgprValuA_T1_I0+8+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3], v[166:167], v[vgprValuA_X1_I0+18:vgprValuA_X1_I0+18+1], v[vgprValuA_X1_I0+20:vgprValuA_X1_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_A_2: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+18:vgprValuB_X1_I0+18+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+16], v[vgprValuB_X1_I0+16], v[vgprValuB_X1_I0+17]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+17], v[vgprValuB_X1_I0+18], v[vgprValuB_X1_I0+19]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+18], v[vgprValuB_X1_I0+20], v[vgprValuB_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+19], v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+23]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3], v[166:167], v[vgprValuB_X1_I0+18:vgprValuB_X1_I0+18+1], v[vgprValuB_X1_I0+20:vgprValuB_X1_I0+20+3] // Calculate low bits for TF32 emulation__TF32_1_B_2: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+23], v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+22], v[vgprValuA_X1_I0+20], v[vgprValuA_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+21], v[vgprValuA_T1_I0+10], v[vgprValuA_T1_I0+11]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+20], v[vgprValuA_T1_I0+8], v[vgprValuA_T1_I0+9] // __TF32_2_A_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+23], v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+23] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+22], v[vgprValuB_X1_I0+20], v[vgprValuB_X1_I0+21]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+21], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+20], v202, v203 // __TF32_2_B_2 pack final end
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+24], v[vgprValuA_T1_I0+12], v[vgprValuA_T1_I0+13]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+25], v[vgprValuA_T1_I0+14], v[vgprValuA_T1_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+26], v[vgprValuA_X1_I0+28], v[vgprValuA_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+27], v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3], v[166:167], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+1], v[vgprValuA_T1_I0+12:vgprValuA_T1_I0+12+3] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3], v[166:167], v[vgprValuA_X1_I0+26:vgprValuA_X1_I0+26+1], v[vgprValuA_X1_I0+28:vgprValuA_X1_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_A_3: 
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+26:vgprValuB_X1_I0+26+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+24], v[vgprValuB_X1_I0+24], v[vgprValuB_X1_I0+25]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+25], v[vgprValuB_X1_I0+26], v[vgprValuB_X1_I0+27]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+26], v[vgprValuB_X1_I0+28], v[vgprValuB_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+27], v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+31]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3], v[166:167], v[vgprValuB_X1_I0+26:vgprValuB_X1_I0+26+1], v[vgprValuB_X1_I0+28:vgprValuB_X1_I0+28+3] // Calculate low bits for TF32 emulation__TF32_1_B_3: 
s_nop 0                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+31], v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+30], v[vgprValuA_X1_I0+28], v[vgprValuA_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+29], v[vgprValuA_T1_I0+14], v[vgprValuA_T1_I0+15]
v_cvt_pk_bf16_f32 v[vgprValuA_X1_I0+28], v[vgprValuA_T1_I0+12], v[vgprValuA_T1_I0+13] // __TF32_2_A_3 pack final end
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+31], v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+31] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+30], v[vgprValuB_X1_I0+28], v[vgprValuB_X1_I0+29]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+29], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+28], v202, v203 // __TF32_2_B_3 pack final end
v_mov_b64 v[202:203], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+1]
v_mov_b64 v[204:205], v[vgprValuB_X1_I0+34:vgprValuB_X1_I0+34+1]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+32], v[vgprValuB_X1_I0+32], v[vgprValuB_X1_I0+33]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+33], v[vgprValuB_X1_I0+34], v[vgprValuB_X1_I0+35]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+34], v[vgprValuB_X1_I0+36], v[vgprValuB_X1_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+35], v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+39]
v_mfma_f32_4x4x4_16b_bf16 v[202:205], v[166:167], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+1], v[202:205] // Calculate low bits for TF32 emulation
v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3], v[166:167], v[vgprValuB_X1_I0+34:vgprValuB_X1_I0+34+1], v[vgprValuB_X1_I0+36:vgprValuB_X1_I0+36+3] // Calculate low bits for TF32 emulation__TF32_1_B_4: 
s_nop 4                                            // nop for x32f emulation
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+39], v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+39] // pack final begin
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+38], v[vgprValuB_X1_I0+36], v[vgprValuB_X1_I0+37]
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+37], v204, v205
v_cvt_pk_bf16_f32 v[vgprValuB_X1_I0+36], v202, v203 // __TF32_2_B_4 pack final end
s_nop 1
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[0:3] // src0_h*src1_h, left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[0:3] // src0_h*src1_l, left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[0:3], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[0:3] // src0_l*src1_h, left value = acc[0+0:3+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[4:7] // src0_h*src1_h, left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[4:7] // src0_h*src1_l, left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[4:7], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[4:7] // src0_l*src1_h, left value = acc[4+0:7+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[8:11] // src0_h*src1_h, left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[8:11] // src0_h*src1_l, left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[8:11], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[8:11] // src0_l*src1_h, left value = acc[8+0:11+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[12:15] // src0_h*src1_h, left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[12:15] // src0_h*src1_l, left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[12:15], v[vgprValuB_X1_I0+0+4:vgprValuB_X1_I0+0+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[12:15] // src0_l*src1_h, left value = acc[12+0:15+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[16:19] // src0_h*src1_h, left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[16:19] // src0_h*src1_l, left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[16:19], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[16:19] // src0_l*src1_h, left value = acc[16+0:19+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[20:23] // src0_h*src1_h, left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[20:23] // src0_h*src1_l, left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[20:23], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[20:23] // src0_l*src1_h, left value = acc[20+0:23+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[24:27] // src0_h*src1_h, left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[24:27] // src0_h*src1_l, left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[24:27], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[24:27] // src0_l*src1_h, left value = acc[24+0:27+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[28:31] // src0_h*src1_h, left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8:vgprValuB_X1_I0+8+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[28:31] // src0_h*src1_l, left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[28:31], v[vgprValuB_X1_I0+8+4:vgprValuB_X1_I0+8+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[28:31] // src0_l*src1_h, left value = acc[28+0:31+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[32:35] // src0_h*src1_h, left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[32:35] // src0_h*src1_l, left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[32:35], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[32:35] // src0_l*src1_h, left value = acc[32+0:35+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[36:39] // src0_h*src1_h, left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[36:39] // src0_h*src1_l, left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[36:39], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[36:39] // src0_l*src1_h, left value = acc[36+0:39+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[40:43] // src0_h*src1_h, left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[40:43] // src0_h*src1_l, left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[40:43], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[40:43] // src0_l*src1_h, left value = acc[40+0:43+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[44:47] // src0_h*src1_h, left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16:vgprValuB_X1_I0+16+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[44:47] // src0_h*src1_l, left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[44:47], v[vgprValuB_X1_I0+16+4:vgprValuB_X1_I0+16+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[44:47] // src0_l*src1_h, left value = acc[44+0:47+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[48:51] // src0_h*src1_h, left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[48:51] // src0_h*src1_l, left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[48:51], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[48:51] // src0_l*src1_h, left value = acc[48+0:51+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[52:55] // src0_h*src1_h, left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[52:55] // src0_h*src1_l, left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[52:55], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[52:55] // src0_l*src1_h, left value = acc[52+0:55+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[56:59] // src0_h*src1_h, left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[56:59] // src0_h*src1_l, left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[56:59], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[56:59] // src0_l*src1_h, left value = acc[56+0:59+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[60:63] // src0_h*src1_h, left value = acc[60+0:63+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24:vgprValuB_X1_I0+24+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[60:63] // src0_h*src1_l, left value = acc[60+0:63+0]
v_mfma_f32_16x16x32_bf16 acc[60:63], v[vgprValuB_X1_I0+24+4:vgprValuB_X1_I0+24+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[60:63] // src0_l*src1_h, left value = acc[60+0:63+0]
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[64:67] // src0_h*src1_h, left value = acc[64+0:67+0]
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+0+4:vgprValuA_X1_I0+0+4+3], acc[64:67] // src0_h*src1_l, left value = acc[64+0:67+0]
v_mfma_f32_16x16x32_bf16 acc[64:67], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], acc[64:67] // src0_l*src1_h, left value = acc[64+0:67+0]
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[68:71] // src0_h*src1_h, left value = acc[68+0:71+0]
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+8+4:vgprValuA_X1_I0+8+4+3], acc[68:71] // src0_h*src1_l, left value = acc[68+0:71+0]
v_mfma_f32_16x16x32_bf16 acc[68:71], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+8:vgprValuA_X1_I0+8+3], acc[68:71] // src0_l*src1_h, left value = acc[68+0:71+0]
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[72:75] // src0_h*src1_h, left value = acc[72+0:75+0]
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+16+4:vgprValuA_X1_I0+16+4+3], acc[72:75] // src0_h*src1_l, left value = acc[72+0:75+0]
v_mfma_f32_16x16x32_bf16 acc[72:75], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+16:vgprValuA_X1_I0+16+3], acc[72:75] // src0_l*src1_h, left value = acc[72+0:75+0]
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[76:79] // src0_h*src1_h, left value = acc[76+0:79+0]
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32:vgprValuB_X1_I0+32+3], v[vgprValuA_X1_I0+24+4:vgprValuA_X1_I0+24+4+3], acc[76:79] // src0_h*src1_l, left value = acc[76+0:79+0]
v_mfma_f32_16x16x32_bf16 acc[76:79], v[vgprValuB_X1_I0+32+4:vgprValuB_X1_I0+32+4+3], v[vgprValuA_X1_I0+24:vgprValuA_X1_I0+24+3], acc[76:79] // src0_l*src1_h, left value = acc[76+0:79+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
s_mov_b32 s84, 4                                   // tailloop lds offset
s_mul_i32 s84, s[sgprOrigLoopCounter], s84         // scale by mul
v_sub_u32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA], s84 // remove lro damage
s_mov_b32 s84, 4                                   // tailloop lds offset
s_mul_i32 s84, s[sgprOrigLoopCounter], s84         // scale by mul
v_sub_u32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB], s84 // remove lro damage
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_X1_I0, UNDEF
.set vgprValuA_T0_I0, UNDEF
.set vgprValuA_T1_I0, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_X1_I0, UNDEF
.set IdentityMatrix, UNDEF
label_Summation_End_2G3LC8VCGIZD1EUX:
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
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
v_lshrrev_b32 v24, 6, v[vgprSerial]                // 24 = Serial / 64
v_lshrrev_b32 v25, 1, v24                          // 25 = 24 / 2
v_mul_lo_u32 v25, 0x10, v25                        // wave coordination offset 1
v_and_b32 v21, 63, v[vgprSerial]                   // v21 = v[vgprSerial] % 64
v_lshrrev_b32 v21, 4, v21                          // 21 = 21 / 16
v_lshlrev_b32 v21, 2, v21                          // thread0 * continuous_output
v_add_lshl_u32 v21, v25, v21, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v22, v21, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v23, v21, s[sgprStrideD1J]            //  offset 1
v_and_b32 v20, 1, v24                              // v20 = v24 % 2
v_mul_lo_u32 v20, 0x10, v20                        // wave coordination offset 0
v_and_b32 v25, 15, v[vgprSerial]                   // v25 = v[vgprSerial] % 16
v_add_lshl_u32 v20, v25, v20, 2                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v20, s8, v20                             // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 160, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v21, s8, v21                             // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GSU                           // Branch to stream-k store code
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_GSU                           // branch if split == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:
s_and_b32 s68, 127, s[sgprSizeI]                   // s68 = s[sgprSizeI] % 128
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MB_Else         // jump if edges required
s_mov_b32 s71, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s70, 819, s[sgprSizeJ]                   // tmp1 = dividend * magic hi
s_lshl_b64 s[70:71], s[70:71], 16                  // left shift 16 bits
s_mul_i32 s69, s[sgprSizeJ], 13108                 // tmp0 = dividend * magic lo
s_add_u32 s70, s69, s70                            // add lo
s_addc_u32 s71, s71, 0                             // add hi
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp1 = (dividend * magic) << shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MB_Then         // jump if edges required
label_GW_B0_FD0_VW4_MB_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=46 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (1,0,0,0:vw4); (1,0,1,0:vw4); (1,0,2,0:vw4); (1,0,3,0:vw4); (2,0,0,0:vw4); (2,0,1,0:vw4); (2,0,2,0:vw4); (2,0,3,0:vw4); (3,0,0,0:vw4); (3,0,1,0:vw4); (3,0,2,0:vw4); (3,0,3,0:vw4); (4,0,0,0:vw4); (4,0,1,0:vw4); (4,0,2,0:vw4); (4,0,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
v_add_lshl_u32 v27, v23, v20, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=20, coord0Vgpr=20 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+32], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+33], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+34], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+35], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+36], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+37], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+38], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+39], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+44], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+45], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+46], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+47], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+48], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+49], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+50], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+51], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+52], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+53], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+54], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+55], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+56], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+57], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+58], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+59], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+60], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+61], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+62], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+63], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+64], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+65], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+66], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+67], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+68], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+69], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+70], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+71], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+72], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+73], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+74], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+75], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+76], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+77], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+78], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+79], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+80], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+81], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+82], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+83], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+84], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+85], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+86], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+87], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+88], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+89], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+90], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+91], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+92], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+93], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+94], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+95], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+96], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+97], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+98], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+99], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+100], acc65         // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+101], acc69         // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+102], acc73         // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+103], acc77         // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+104], acc66         // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+105], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+106], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+107], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+108], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+109], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+110], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+111], acc79         // copy acc to vreg[79]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0), (3, 0, 0, 0), (3, 0, 1, 0), (3, 0, 2, 0), (3, 0, 3, 0), (4, 0, 0, 0), (4, 0, 1, 0), (4, 0, 2, 0), (4, 0, 3, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[32:35], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[36:39], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[44:47], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[52:55], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[60:63], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[68:71], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[72:75], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[76:79], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[80:83], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[84:87], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[88:91], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[92:95], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[96:99], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[100:103], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[104:107], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[108:111], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_MB_NonEdgeEnd:
label_GW_B0_FD0_VW4_MB_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=37 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (1,0,0,0:vw4); (1,0,1,0:vw4); (1,0,2,0:vw4); (1,0,3,0:vw4); (2,0,0,0:vw4); (2,0,1,0:vw4); (2,0,2,0:vw4); (2,0,3,0:vw4); (3,0,0,0:vw4); (3,0,1,0:vw4); (3,0,2,0:vw4); (3,0,3,0:vw4); (4,0,0,0:vw4); (4,0,1,0:vw4); (4,0,2,0:vw4); (4,0,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v26, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v27, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v27, v26, v27, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v108, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v26, v108, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v109, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v26, v109, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v110, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v26, v110, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v111, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v26, v111, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v112, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v26, v112, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v113, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v26, v113, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v114, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v26, v114, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v115, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v26, v115, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v116, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v26, v116, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v117, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v26, v117, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v118, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v26, v118, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v119, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v26, v119, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v120, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v26, v120, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v121, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v26, v121, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v122, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v26, v122, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v123, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v26, v123, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v124, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v26, v124, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v125, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v26, v125, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v126, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v26, v126, s[72:73]            // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+28], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+29], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+30], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+31], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+36], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+37], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+38], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+39], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+40], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+41], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+42], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+43], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+44], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+45], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+46], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+47], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+48], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+49], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+50], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+51], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+52], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+53], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+54], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+55], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+56], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+57], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+58], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+59], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+60], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+61], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+62], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+63], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+64], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+65], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+66], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+67], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+68], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+69], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+70], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+71], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+72], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+73], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+74], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+75], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+76], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+77], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+78], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+79], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+80], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+81], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+82], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+83], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+84], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+85], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+86], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+87], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+88], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+89], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+90], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+91], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+92], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+93], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+94], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+95], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+96], acc65          // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+97], acc69          // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+98], acc73          // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+99], acc77          // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+100], acc66         // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+101], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+102], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+103], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+104], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+105], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+106], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+107], acc79         // copy acc to vreg[79]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0), (3, 0, 0, 0), (3, 0, 1, 0), (3, 0, 2, 0), (3, 0, 3, 0), (4, 0, 0, 0), (4, 0, 1, 0), (4, 0, 2, 0), (4, 0, 3, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[28:31], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[32:35], v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[36:39], v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[40:43], v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[44:47], v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[48:51], v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[52:55], v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[56:59], v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[60:63], v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[64:67], v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[68:71], v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[72:75], v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[76:79], v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[80:83], v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[84:87], v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[88:91], v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[92:95], v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[96:99], v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[100:103], v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[104:107], v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_MB_Else:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=95 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (1,0,0,0:vw1); (1,0,0,1:vw1); (1,0,0,2:vw1); (1,0,0,3:vw1); (1,0,1,0:vw1); (1,0,1,1:vw1); (1,0,1,2:vw1); (1,0,1,3:vw1); (1,0,2,0:vw1); (1,0,2,1:vw1); (1,0,2,2:vw1); (1,0,2,3:vw1); (1,0,3,0:vw1); (1,0,3,1:vw1); (1,0,3,2:vw1); (1,0,3,3:vw1); (2,0,0,0:vw1); (2,0,0,1:vw1); (2,0,0,2:vw1); (2,0,0,3:vw1); (2,0,1,0:vw1); (2,0,1,1:vw1); (2,0,1,2:vw1); (2,0,1,3:vw1); (2,0,2,0:vw1); (2,0,2,1:vw1); (2,0,2,2:vw1); (2,0,2,3:vw1); (2,0,3,0:vw1); (2,0,3,1:vw1); (2,0,3,2:vw1); (2,0,3,3:vw1); (3,0,0,0:vw1); (3,0,0,1:vw1); (3,0,0,2:vw1); (3,0,0,3:vw1); (3,0,1,0:vw1); (3,0,1,1:vw1); (3,0,1,2:vw1); (3,0,1,3:vw1); (3,0,2,0:vw1); (3,0,2,1:vw1); (3,0,2,2:vw1); (3,0,2,3:vw1); (3,0,3,0:vw1); (3,0,3,1:vw1); (3,0,3,2:vw1); (3,0,3,3:vw1); (4,0,0,0:vw1); (4,0,0,1:vw1); (4,0,0,2:vw1); (4,0,0,3:vw1); (4,0,1,0:vw1); (4,0,1,1:vw1); (4,0,1,2:vw1); (4,0,1,3:vw1); (4,0,2,0:vw1); (4,0,2,1:vw1); (4,0,2,2:vw1); (4,0,2,3:vw1); (4,0,3,0:vw1); (4,0,3,1:vw1); (4,0,3,2:vw1); (4,0,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v26, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v107, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v26, v107, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v108, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v26, v108, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v109, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v26, v109, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v110, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v26, v110, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v111, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v26, v111, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v112, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v26, v112, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v113, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v26, v113, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v114, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v26, v114, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v115, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v26, v115, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v116, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v26, v116, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v117, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v26, v117, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v118, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v26, v118, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v119, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v26, v119, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v120, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v26, v120, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v121, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v26, v121, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v122, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v26, v122, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v123, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v26, v123, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v124, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v26, v124, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v125, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v26, v125, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v126, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v26, v126, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v127, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v26, v127, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v128, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v26, v128, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v129, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v26, v129, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v130, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v26, v130, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v131, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v26, v131, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v132, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v26, v132, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v133, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v26, v133, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v134, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v26, v134, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v135, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v26, v135, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v136, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v26, v136, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v137, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v26, v137, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v138, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v26, v138, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v139, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v26, v139, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v140, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v26, v140, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v141, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v26, v141, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v142, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v26, v142, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v143, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v26, v143, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v144, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v26, v144, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v145, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v26, v145, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v146, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v26, v146, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v147, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v26, v147, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v148, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v26, v148, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v149, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v26, v149, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v150, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v26, v150, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v151, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v26, v151, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v152, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v26, v152, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v153, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v26, v153, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v154, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v26, v154, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v155, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v26, v155, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v156, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v26, v156, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v157, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v26, v157, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v158, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v26, v158, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v159, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v26, v159, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v160, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v26, v160, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v161, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v26, v161, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v162, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v26, v162, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v163, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v26, v163, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v201, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v26, v201, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v202, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v26, v202, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v203, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v26, v203, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v204, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v26, v204, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v205, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v26, v205, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v206, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v26, v206, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v207, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v26, v207, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v208, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v26, v208, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v209, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v26, v209, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v210, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v26, v210, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v211, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v26, v211, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v212, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v26, v212, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v213, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v26, v213, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v214, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v26, v214, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v215, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v26, v215, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v216, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v26, v216, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v217, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v26, v217, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v218, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v26, v218, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v219, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v26, v219, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v220, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v26, v220, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v221, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v26, v221, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v222, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v26, v222, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v223, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v26, v223, s[72:73]            // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+27], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+28], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+29], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+30], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+31], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+32], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+33], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+34], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+35], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+36], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+37], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+38], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+39], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+40], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+41], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+42], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+43], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+44], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+45], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+46], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+47], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+48], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+49], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+50], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+51], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+52], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+53], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+54], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+55], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+56], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+57], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+58], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+59], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+60], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+61], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+62], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+63], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+64], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+65], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+66], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+67], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+68], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+69], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+70], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+71], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+72], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+73], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+74], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+75], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+76], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+77], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+78], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+79], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+80], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+81], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+82], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+83], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+84], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+85], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+86], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+87], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+88], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+89], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+90], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+91], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+92], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+93], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+94], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+95], acc65          // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+96], acc69          // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+97], acc73          // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+98], acc77          // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+99], acc66          // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+100], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+101], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+102], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+103], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+104], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+105], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+106], acc79         // copy acc to vreg[79]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3), (1, 0, 1, 0), (1, 0, 1, 1), (1, 0, 1, 2), (1, 0, 1, 3), (1, 0, 2, 0), (1, 0, 2, 1), (1, 0, 2, 2), (1, 0, 2, 3), (1, 0, 3, 0), (1, 0, 3, 1), (1, 0, 3, 2), (1, 0, 3, 3), (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 0, 2), (2, 0, 0, 3), (2, 0, 1, 0), (2, 0, 1, 1), (2, 0, 1, 2), (2, 0, 1, 3), (2, 0, 2, 0), (2, 0, 2, 1), (2, 0, 2, 2), (2, 0, 2, 3), (2, 0, 3, 0), (2, 0, 3, 1), (2, 0, 3, 2), (2, 0, 3, 3), (3, 0, 0, 0), (3, 0, 0, 1), (3, 0, 0, 2), (3, 0, 0, 3), (3, 0, 1, 0), (3, 0, 1, 1), (3, 0, 1, 2), (3, 0, 1, 3), (3, 0, 2, 0), (3, 0, 2, 1), (3, 0, 2, 2), (3, 0, 2, 3), (3, 0, 3, 0), (3, 0, 3, 1), (3, 0, 3, 2), (3, 0, 3, 3), (4, 0, 0, 0), (4, 0, 0, 1), (4, 0, 0, 2), (4, 0, 0, 3), (4, 0, 1, 0), (4, 0, 1, 1), (4, 0, 1, 2), (4, 0, 1, 3), (4, 0, 2, 0), (4, 0, 2, 1), (4, 0, 2, 2), (4, 0, 2, 3), (4, 0, 3, 0), (4, 0, 3, 1), (4, 0, 3, 2), (4, 0, 3, 3)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v27, v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v28, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v29, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v30, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v31, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v32, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v33, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v34, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v35, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v36, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v37, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v38, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v39, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v40, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v41, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v42, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v43, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v44, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v45, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v46, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v47, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v48, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v49, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v50, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v51, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v52, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v53, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v54, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v55, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v56, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v57, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v58, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v59, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v60, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v61, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v62, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v63, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v64, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v65, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v66, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v67, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v68, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v69, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v70, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v71, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v72, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v73, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v74, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v75, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v76, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v77, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v78, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v79, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v80, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v81, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v82, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v83, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v84, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v85, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v86, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v87, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v88, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v89, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v90, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v91, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v92, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v93, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v94, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v95, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v96, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v97, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v98, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v99, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v100, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v101, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v102, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v103, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v104, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v105, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v106, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[68:69]                               // addr of next instr
s_add_i32 s70, label_KernelEnd, 4                  // target branch offset
s_add_u32 s68, s68, s70                            // add target branch offset
s_addc_u32 s69, s69, 0                             // add high and carry
s_setpc_b64 s[68:69]                               // branch to label_KernelEnd
label_GSU:
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_0MXDW6EW9K7ZNG8F     // Only branch on scc0
s_getpc_b64 s[72:73]                               // addr of next instr
s_add_i32 s74, label_SK_Partials_1, 4              // target branch offset
s_add_u32 s72, s72, s74                            // add target branch offset
s_addc_u32 s73, s73, 0                             // add high and carry
s_setpc_b64 s[72:73]                               // branch to label_SK_Partials_1
label_NoBranch_0MXDW6EW9K7ZNG8F:
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cbranch_scc1 label_SK_Store                      // Branch if started and finished tile, go to regular store code
s_add_u32 s8, s[sgprStreamKIdx], 1                 // input partial tile index
s_mul_hi_u32 s69, s[sgprStreamKIterEnd], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s70, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s68, s[sgprStreamKIterEnd], s70          // s_magic mul, div alg 2
s_add_u32 s68, s68, s69
s_and_b32 s70, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s68, s68, s70                           // sMagicDiv Alg 2
s_mul_i32 s68, s68, s[sgprItersPerTile]            // start iteration of partial tile
s_sub_u32 s9, s[sgprStreamKIterEnd], s68           // calc iterations completed by this WG
label_SK_Fixup:
s_lshl_b32 s68, s8, 2                              // flag offset based on CTA index
s_load_dword s70, s[sgprAddressFlags:sgprAddressFlags+1], s68 glc // get flag
s_waitcnt lgkmcnt(0)                               // wait for flag load
s_cmp_eq_u32 s70, 1                                // check if ready
s_cbranch_scc0 label_SK_Fixup                      // if flag not set, wait and check again
s_barrier                                          // wait for all workgroups before resetting flag
v_readfirstlane_b32 s70, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s70, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagReset              // Skip flag reset
s_store_dword s70, s[sgprAddressFlags:sgprAddressFlags+1], s68 glc // reset flag
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=22 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s62, 0x14000, s8                         // Offset to correct partials tile
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s62      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], 0       // add hi to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (1,0,0,0:vw4); (1,0,1,0:vw4); (1,0,2,0:vw4); (1,0,3,0:vw4); (2,0,0,0:vw4); (2,0,1,0:vw4); (2,0,2,0:vw4); (2,0,3,0:vw4); (3,0,0,0:vw4); (3,0,1,0:vw4); (3,0,2,0:vw4); (3,0,3,0:vw4); (4,0,0,0:vw4); (4,0,1,0:vw4); (4,0,2,0:vw4); (4,0,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v28, 4, v[vgprSerial]                // v28 = v[vgprSerial] * 16
s_mov_b32 s62, 0                                   // Init sgpr offset
buffer_load_dwordx4 v[112:115], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[116:119], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[120:123], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[124:127], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[128:131], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[132:135], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[136:139], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[140:143], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[144:147], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[148:151], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[152:155], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[156:159], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[160:163], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[204:207], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[208:211], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[212:215], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[216:219], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[220:223], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[224:227], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[228:231], v28, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 // load WS
v_accvgpr_read_b32 v[vgprValuC+32], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+33], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+34], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+35], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+36], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+37], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+38], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+39], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+44], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+45], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+46], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+47], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+48], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+49], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+50], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+51], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+52], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+53], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+54], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+55], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+56], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+57], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+58], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+59], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+60], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+61], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+62], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+63], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+64], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+65], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+66], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+67], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+68], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+69], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+70], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+71], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+72], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+73], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+74], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+75], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+76], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+77], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+78], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+79], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+80], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+81], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+82], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+83], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+84], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+85], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+86], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+87], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+88], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+89], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+90], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+91], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+92], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+93], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+94], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+95], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+96], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+97], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+98], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+99], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+100], acc65         // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+101], acc69         // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+102], acc73         // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+103], acc77         // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+104], acc66         // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+105], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+106], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+107], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+108], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+109], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+110], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+111], acc79         // copy acc to vreg[79]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(19)                                // wait C (interleaved) 19 = 20 - 0 + 0 - 1
v_add_f32 v[vgprValuC+32], v[vgprValuC+32], v112   // accum partials
v_add_f32 v[vgprValuC+33], v[vgprValuC+33], v113   // accum partials
v_add_f32 v[vgprValuC+34], v[vgprValuC+34], v114   // accum partials
v_add_f32 v[vgprValuC+35], v[vgprValuC+35], v115   // accum partials

s_waitcnt vmcnt(18)                                // wait C (interleaved) 18 = 20 - 1 + 0 - 1
v_add_f32 v[vgprValuC+36], v[vgprValuC+36], v116   // accum partials
v_add_f32 v[vgprValuC+37], v[vgprValuC+37], v117   // accum partials
v_add_f32 v[vgprValuC+38], v[vgprValuC+38], v118   // accum partials
v_add_f32 v[vgprValuC+39], v[vgprValuC+39], v119   // accum partials

s_waitcnt vmcnt(17)                                // wait C (interleaved) 17 = 20 - 2 + 0 - 1
v_add_f32 v[vgprValuC+40], v[vgprValuC+40], v120   // accum partials
v_add_f32 v[vgprValuC+41], v[vgprValuC+41], v121   // accum partials
v_add_f32 v[vgprValuC+42], v[vgprValuC+42], v122   // accum partials
v_add_f32 v[vgprValuC+43], v[vgprValuC+43], v123   // accum partials

s_waitcnt vmcnt(16)                                // wait C (interleaved) 16 = 20 - 3 + 0 - 1
v_add_f32 v[vgprValuC+44], v[vgprValuC+44], v124   // accum partials
v_add_f32 v[vgprValuC+45], v[vgprValuC+45], v125   // accum partials
v_add_f32 v[vgprValuC+46], v[vgprValuC+46], v126   // accum partials
v_add_f32 v[vgprValuC+47], v[vgprValuC+47], v127   // accum partials

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 20 - 4 + 0 - 1
v_add_f32 v[vgprValuC+48], v[vgprValuC+48], v128   // accum partials
v_add_f32 v[vgprValuC+49], v[vgprValuC+49], v129   // accum partials
v_add_f32 v[vgprValuC+50], v[vgprValuC+50], v130   // accum partials
v_add_f32 v[vgprValuC+51], v[vgprValuC+51], v131   // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 20 - 5 + 0 - 1
v_add_f32 v[vgprValuC+52], v[vgprValuC+52], v132   // accum partials
v_add_f32 v[vgprValuC+53], v[vgprValuC+53], v133   // accum partials
v_add_f32 v[vgprValuC+54], v[vgprValuC+54], v134   // accum partials
v_add_f32 v[vgprValuC+55], v[vgprValuC+55], v135   // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 20 - 6 + 0 - 1
v_add_f32 v[vgprValuC+56], v[vgprValuC+56], v136   // accum partials
v_add_f32 v[vgprValuC+57], v[vgprValuC+57], v137   // accum partials
v_add_f32 v[vgprValuC+58], v[vgprValuC+58], v138   // accum partials
v_add_f32 v[vgprValuC+59], v[vgprValuC+59], v139   // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 20 - 7 + 0 - 1
v_add_f32 v[vgprValuC+60], v[vgprValuC+60], v140   // accum partials
v_add_f32 v[vgprValuC+61], v[vgprValuC+61], v141   // accum partials
v_add_f32 v[vgprValuC+62], v[vgprValuC+62], v142   // accum partials
v_add_f32 v[vgprValuC+63], v[vgprValuC+63], v143   // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 20 - 8 + 0 - 1
v_add_f32 v[vgprValuC+64], v[vgprValuC+64], v144   // accum partials
v_add_f32 v[vgprValuC+65], v[vgprValuC+65], v145   // accum partials
v_add_f32 v[vgprValuC+66], v[vgprValuC+66], v146   // accum partials
v_add_f32 v[vgprValuC+67], v[vgprValuC+67], v147   // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 20 - 9 + 0 - 1
v_add_f32 v[vgprValuC+68], v[vgprValuC+68], v148   // accum partials
v_add_f32 v[vgprValuC+69], v[vgprValuC+69], v149   // accum partials
v_add_f32 v[vgprValuC+70], v[vgprValuC+70], v150   // accum partials
v_add_f32 v[vgprValuC+71], v[vgprValuC+71], v151   // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 20 - 10 + 0 - 1
v_add_f32 v[vgprValuC+72], v[vgprValuC+72], v152   // accum partials
v_add_f32 v[vgprValuC+73], v[vgprValuC+73], v153   // accum partials
v_add_f32 v[vgprValuC+74], v[vgprValuC+74], v154   // accum partials
v_add_f32 v[vgprValuC+75], v[vgprValuC+75], v155   // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 20 - 11 + 0 - 1
v_add_f32 v[vgprValuC+76], v[vgprValuC+76], v156   // accum partials
v_add_f32 v[vgprValuC+77], v[vgprValuC+77], v157   // accum partials
v_add_f32 v[vgprValuC+78], v[vgprValuC+78], v158   // accum partials
v_add_f32 v[vgprValuC+79], v[vgprValuC+79], v159   // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 20 - 12 + 0 - 1
v_add_f32 v[vgprValuC+80], v[vgprValuC+80], v160   // accum partials
v_add_f32 v[vgprValuC+81], v[vgprValuC+81], v161   // accum partials
v_add_f32 v[vgprValuC+82], v[vgprValuC+82], v162   // accum partials
v_add_f32 v[vgprValuC+83], v[vgprValuC+83], v163   // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 20 - 13 + 0 - 1
v_add_f32 v[vgprValuC+84], v[vgprValuC+84], v204   // accum partials
v_add_f32 v[vgprValuC+85], v[vgprValuC+85], v205   // accum partials
v_add_f32 v[vgprValuC+86], v[vgprValuC+86], v206   // accum partials
v_add_f32 v[vgprValuC+87], v[vgprValuC+87], v207   // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 20 - 14 + 0 - 1
v_add_f32 v[vgprValuC+88], v[vgprValuC+88], v208   // accum partials
v_add_f32 v[vgprValuC+89], v[vgprValuC+89], v209   // accum partials
v_add_f32 v[vgprValuC+90], v[vgprValuC+90], v210   // accum partials
v_add_f32 v[vgprValuC+91], v[vgprValuC+91], v211   // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 20 - 15 + 0 - 1
v_add_f32 v[vgprValuC+92], v[vgprValuC+92], v212   // accum partials
v_add_f32 v[vgprValuC+93], v[vgprValuC+93], v213   // accum partials
v_add_f32 v[vgprValuC+94], v[vgprValuC+94], v214   // accum partials
v_add_f32 v[vgprValuC+95], v[vgprValuC+95], v215   // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 20 - 16 + 0 - 1
v_add_f32 v[vgprValuC+96], v[vgprValuC+96], v216   // accum partials
v_add_f32 v[vgprValuC+97], v[vgprValuC+97], v217   // accum partials
v_add_f32 v[vgprValuC+98], v[vgprValuC+98], v218   // accum partials
v_add_f32 v[vgprValuC+99], v[vgprValuC+99], v219   // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 20 - 17 + 0 - 1
v_add_f32 v[vgprValuC+100], v[vgprValuC+100], v220 // accum partials
v_add_f32 v[vgprValuC+101], v[vgprValuC+101], v221 // accum partials
v_add_f32 v[vgprValuC+102], v[vgprValuC+102], v222 // accum partials
v_add_f32 v[vgprValuC+103], v[vgprValuC+103], v223 // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 20 - 18 + 0 - 1
v_add_f32 v[vgprValuC+104], v[vgprValuC+104], v224 // accum partials
v_add_f32 v[vgprValuC+105], v[vgprValuC+105], v225 // accum partials
v_add_f32 v[vgprValuC+106], v[vgprValuC+106], v226 // accum partials
v_add_f32 v[vgprValuC+107], v[vgprValuC+107], v227 // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 20 - 19 + 0 - 1
v_add_f32 v[vgprValuC+108], v[vgprValuC+108], v228 // accum partials
v_add_f32 v[vgprValuC+109], v[vgprValuC+109], v229 // accum partials
v_add_f32 v[vgprValuC+110], v[vgprValuC+110], v230 // accum partials
v_add_f32 v[vgprValuC+111], v[vgprValuC+111], v231 // accum partials
v_accvgpr_write_b32 acc0, v[vgprValuC+32]          // copy vreg[0] to acc
v_accvgpr_write_b32 acc4, v[vgprValuC+33]          // copy vreg[1] to acc
v_accvgpr_write_b32 acc8, v[vgprValuC+34]          // copy vreg[2] to acc
v_accvgpr_write_b32 acc12, v[vgprValuC+35]         // copy vreg[3] to acc
v_accvgpr_write_b32 acc1, v[vgprValuC+36]          // copy vreg[4] to acc
v_accvgpr_write_b32 acc5, v[vgprValuC+37]          // copy vreg[5] to acc
v_accvgpr_write_b32 acc9, v[vgprValuC+38]          // copy vreg[6] to acc
v_accvgpr_write_b32 acc13, v[vgprValuC+39]         // copy vreg[7] to acc
v_accvgpr_write_b32 acc2, v[vgprValuC+40]          // copy vreg[8] to acc
v_accvgpr_write_b32 acc6, v[vgprValuC+41]          // copy vreg[9] to acc
v_accvgpr_write_b32 acc10, v[vgprValuC+42]         // copy vreg[10] to acc
v_accvgpr_write_b32 acc14, v[vgprValuC+43]         // copy vreg[11] to acc
v_accvgpr_write_b32 acc3, v[vgprValuC+44]          // copy vreg[12] to acc
v_accvgpr_write_b32 acc7, v[vgprValuC+45]          // copy vreg[13] to acc
v_accvgpr_write_b32 acc11, v[vgprValuC+46]         // copy vreg[14] to acc
v_accvgpr_write_b32 acc15, v[vgprValuC+47]         // copy vreg[15] to acc
v_accvgpr_write_b32 acc16, v[vgprValuC+48]         // copy vreg[16] to acc
v_accvgpr_write_b32 acc20, v[vgprValuC+49]         // copy vreg[17] to acc
v_accvgpr_write_b32 acc24, v[vgprValuC+50]         // copy vreg[18] to acc
v_accvgpr_write_b32 acc28, v[vgprValuC+51]         // copy vreg[19] to acc
v_accvgpr_write_b32 acc17, v[vgprValuC+52]         // copy vreg[20] to acc
v_accvgpr_write_b32 acc21, v[vgprValuC+53]         // copy vreg[21] to acc
v_accvgpr_write_b32 acc25, v[vgprValuC+54]         // copy vreg[22] to acc
v_accvgpr_write_b32 acc29, v[vgprValuC+55]         // copy vreg[23] to acc
v_accvgpr_write_b32 acc18, v[vgprValuC+56]         // copy vreg[24] to acc
v_accvgpr_write_b32 acc22, v[vgprValuC+57]         // copy vreg[25] to acc
v_accvgpr_write_b32 acc26, v[vgprValuC+58]         // copy vreg[26] to acc
v_accvgpr_write_b32 acc30, v[vgprValuC+59]         // copy vreg[27] to acc
v_accvgpr_write_b32 acc19, v[vgprValuC+60]         // copy vreg[28] to acc
v_accvgpr_write_b32 acc23, v[vgprValuC+61]         // copy vreg[29] to acc
v_accvgpr_write_b32 acc27, v[vgprValuC+62]         // copy vreg[30] to acc
v_accvgpr_write_b32 acc31, v[vgprValuC+63]         // copy vreg[31] to acc
v_accvgpr_write_b32 acc32, v[vgprValuC+64]         // copy vreg[32] to acc
v_accvgpr_write_b32 acc36, v[vgprValuC+65]         // copy vreg[33] to acc
v_accvgpr_write_b32 acc40, v[vgprValuC+66]         // copy vreg[34] to acc
v_accvgpr_write_b32 acc44, v[vgprValuC+67]         // copy vreg[35] to acc
v_accvgpr_write_b32 acc33, v[vgprValuC+68]         // copy vreg[36] to acc
v_accvgpr_write_b32 acc37, v[vgprValuC+69]         // copy vreg[37] to acc
v_accvgpr_write_b32 acc41, v[vgprValuC+70]         // copy vreg[38] to acc
v_accvgpr_write_b32 acc45, v[vgprValuC+71]         // copy vreg[39] to acc
v_accvgpr_write_b32 acc34, v[vgprValuC+72]         // copy vreg[40] to acc
v_accvgpr_write_b32 acc38, v[vgprValuC+73]         // copy vreg[41] to acc
v_accvgpr_write_b32 acc42, v[vgprValuC+74]         // copy vreg[42] to acc
v_accvgpr_write_b32 acc46, v[vgprValuC+75]         // copy vreg[43] to acc
v_accvgpr_write_b32 acc35, v[vgprValuC+76]         // copy vreg[44] to acc
v_accvgpr_write_b32 acc39, v[vgprValuC+77]         // copy vreg[45] to acc
v_accvgpr_write_b32 acc43, v[vgprValuC+78]         // copy vreg[46] to acc
v_accvgpr_write_b32 acc47, v[vgprValuC+79]         // copy vreg[47] to acc
v_accvgpr_write_b32 acc48, v[vgprValuC+80]         // copy vreg[48] to acc
v_accvgpr_write_b32 acc52, v[vgprValuC+81]         // copy vreg[49] to acc
v_accvgpr_write_b32 acc56, v[vgprValuC+82]         // copy vreg[50] to acc
v_accvgpr_write_b32 acc60, v[vgprValuC+83]         // copy vreg[51] to acc
v_accvgpr_write_b32 acc49, v[vgprValuC+84]         // copy vreg[52] to acc
v_accvgpr_write_b32 acc53, v[vgprValuC+85]         // copy vreg[53] to acc
v_accvgpr_write_b32 acc57, v[vgprValuC+86]         // copy vreg[54] to acc
v_accvgpr_write_b32 acc61, v[vgprValuC+87]         // copy vreg[55] to acc
v_accvgpr_write_b32 acc50, v[vgprValuC+88]         // copy vreg[56] to acc
v_accvgpr_write_b32 acc54, v[vgprValuC+89]         // copy vreg[57] to acc
v_accvgpr_write_b32 acc58, v[vgprValuC+90]         // copy vreg[58] to acc
v_accvgpr_write_b32 acc62, v[vgprValuC+91]         // copy vreg[59] to acc
v_accvgpr_write_b32 acc51, v[vgprValuC+92]         // copy vreg[60] to acc
v_accvgpr_write_b32 acc55, v[vgprValuC+93]         // copy vreg[61] to acc
v_accvgpr_write_b32 acc59, v[vgprValuC+94]         // copy vreg[62] to acc
v_accvgpr_write_b32 acc63, v[vgprValuC+95]         // copy vreg[63] to acc
v_accvgpr_write_b32 acc64, v[vgprValuC+96]         // copy vreg[64] to acc
v_accvgpr_write_b32 acc68, v[vgprValuC+97]         // copy vreg[65] to acc
v_accvgpr_write_b32 acc72, v[vgprValuC+98]         // copy vreg[66] to acc
v_accvgpr_write_b32 acc76, v[vgprValuC+99]         // copy vreg[67] to acc
v_accvgpr_write_b32 acc65, v[vgprValuC+100]        // copy vreg[68] to acc
v_accvgpr_write_b32 acc69, v[vgprValuC+101]        // copy vreg[69] to acc
v_accvgpr_write_b32 acc73, v[vgprValuC+102]        // copy vreg[70] to acc
v_accvgpr_write_b32 acc77, v[vgprValuC+103]        // copy vreg[71] to acc
v_accvgpr_write_b32 acc66, v[vgprValuC+104]        // copy vreg[72] to acc
v_accvgpr_write_b32 acc70, v[vgprValuC+105]        // copy vreg[73] to acc
v_accvgpr_write_b32 acc74, v[vgprValuC+106]        // copy vreg[74] to acc
v_accvgpr_write_b32 acc78, v[vgprValuC+107]        // copy vreg[75] to acc
v_accvgpr_write_b32 acc67, v[vgprValuC+108]        // copy vreg[76] to acc
v_accvgpr_write_b32 acc71, v[vgprValuC+109]        // copy vreg[77] to acc
v_accvgpr_write_b32 acc75, v[vgprValuC+110]        // copy vreg[78] to acc
v_accvgpr_write_b32 acc79, v[vgprValuC+111]        // copy vreg[79] to acc
s_nop 1                                            // 2 wait states required before reading vgpr
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s61, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s62, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s61, s61, s62                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_add_u32 s62, s[sgprSKItersPerWG], 1              // Add extra iter
s_cmp_lt_u32 s8, s61                               // Check if next WG had an extra iteration
s_cselect_b32 s62, s62, s[sgprSKItersPerWG]        // Select correct number of iterations for next WG
s_add_u32 s9, s9, s62                              // next partial tile iteration
s_add_u32 s8, s8, 1                                // next partial tile index
s_cmp_lt_u32 s9, s[sgprItersPerTile]               // done loading partial tiles?
s_cbranch_scc1 label_SK_Fixup                      // Branch to continue fixup loop
label_SK_Store:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:
s_and_b32 s68, 127, s[sgprSizeI]                   // s68 = s[sgprSizeI] % 128
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Else       // jump if edges required
s_mov_b32 s71, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s70, 819, s[sgprSizeJ]                   // tmp1 = dividend * magic hi
s_lshl_b64 s[70:71], s[70:71], 16                  // left shift 16 bits
s_mul_i32 s69, s[sgprSizeJ], 13108                 // tmp0 = dividend * magic lo
s_add_u32 s70, s69, s70                            // add lo
s_addc_u32 s71, s71, 0                             // add hi
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp1 = (dividend * magic) << shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=46 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (1,0,0,0:vw4); (1,0,1,0:vw4); (1,0,2,0:vw4); (1,0,3,0:vw4); (2,0,0,0:vw4); (2,0,1,0:vw4); (2,0,2,0:vw4); (2,0,3,0:vw4); (3,0,0,0:vw4); (3,0,1,0:vw4); (3,0,2,0:vw4); (3,0,3,0:vw4); (4,0,0,0:vw4); (4,0,1,0:vw4); (4,0,2,0:vw4); (4,0,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
v_add_lshl_u32 v27, v23, v20, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=20, coord0Vgpr=20 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+32], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+33], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+34], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+35], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+36], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+37], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+38], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+39], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+44], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+45], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+46], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+47], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+48], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+49], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+50], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+51], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+52], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+53], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+54], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+55], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+56], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+57], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+58], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+59], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+60], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+61], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+62], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+63], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+64], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+65], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+66], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+67], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+68], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+69], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+70], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+71], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+72], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+73], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+74], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+75], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+76], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+77], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+78], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+79], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+80], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+81], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+82], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+83], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+84], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+85], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+86], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+87], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+88], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+89], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+90], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+91], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+92], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+93], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+94], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+95], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+96], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+97], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+98], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+99], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+100], acc65         // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+101], acc69         // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+102], acc73         // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+103], acc77         // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+104], acc66         // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+105], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+106], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+107], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+108], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+109], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+110], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+111], acc79         // copy acc to vreg[79]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0), (3, 0, 0, 0), (3, 0, 1, 0), (3, 0, 2, 0), (3, 0, 3, 0), (4, 0, 0, 0), (4, 0, 1, 0), (4, 0, 2, 0), (4, 0, 3, 0)] */
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+108:vgprValuC+108+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+108:vgprValuC+108+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+110:vgprValuC+110+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+110:vgprValuC+110+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[32:35], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[36:39], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[44:47], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[52:55], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[60:63], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[68:71], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[72:75], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[76:79], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[80:83], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[84:87], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[88:91], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[92:95], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[96:99], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[100:103], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[104:107], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[108:111], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=37 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (1,0,0,0:vw4); (1,0,1,0:vw4); (1,0,2,0:vw4); (1,0,3,0:vw4); (2,0,0,0:vw4); (2,0,1,0:vw4); (2,0,2,0:vw4); (2,0,3,0:vw4); (3,0,0,0:vw4); (3,0,1,0:vw4); (3,0,2,0:vw4); (3,0,3,0:vw4); (4,0,0,0:vw4); (4,0,1,0:vw4); (4,0,2,0:vw4); (4,0,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v26, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v27, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v27, v26, v27, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v108, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v26, v108, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v109, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v26, v109, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v110, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v26, v110, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v111, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v26, v111, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v112, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v26, v112, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v113, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v26, v113, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v114, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v26, v114, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v115, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v26, v115, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v116, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v26, v116, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v117, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v26, v117, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v118, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v26, v118, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v119, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v26, v119, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v120, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v26, v120, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v121, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v26, v121, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v122, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v26, v122, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v123, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v26, v123, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v124, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v26, v124, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v125, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v26, v125, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v126, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v26, v126, s[72:73]            // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+28], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+29], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+30], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+31], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+36], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+37], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+38], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+39], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+40], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+41], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+42], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+43], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+44], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+45], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+46], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+47], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+48], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+49], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+50], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+51], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+52], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+53], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+54], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+55], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+56], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+57], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+58], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+59], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+60], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+61], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+62], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+63], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+64], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+65], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+66], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+67], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+68], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+69], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+70], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+71], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+72], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+73], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+74], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+75], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+76], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+77], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+78], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+79], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+80], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+81], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+82], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+83], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+84], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+85], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+86], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+87], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+88], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+89], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+90], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+91], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+92], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+93], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+94], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+95], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+96], acc65          // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+97], acc69          // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+98], acc73          // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+99], acc77          // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+100], acc66         // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+101], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+102], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+103], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+104], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+105], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+106], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+107], acc79         // copy acc to vreg[79]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0), (3, 0, 0, 0), (3, 0, 1, 0), (3, 0, 2, 0), (3, 0, 3, 0), (4, 0, 0, 0), (4, 0, 1, 0), (4, 0, 2, 0), (4, 0, 3, 0)] */
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[28:31], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[32:35], v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[36:39], v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[40:43], v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[44:47], v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[48:51], v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[52:55], v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[56:59], v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[60:63], v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[64:67], v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[68:71], v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[72:75], v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[76:79], v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[80:83], v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[84:87], v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[88:91], v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[92:95], v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[96:99], v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[100:103], v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dwordx4 v[104:107], v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=95 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (1,0,0,0:vw1); (1,0,0,1:vw1); (1,0,0,2:vw1); (1,0,0,3:vw1); (1,0,1,0:vw1); (1,0,1,1:vw1); (1,0,1,2:vw1); (1,0,1,3:vw1); (1,0,2,0:vw1); (1,0,2,1:vw1); (1,0,2,2:vw1); (1,0,2,3:vw1); (1,0,3,0:vw1); (1,0,3,1:vw1); (1,0,3,2:vw1); (1,0,3,3:vw1); (2,0,0,0:vw1); (2,0,0,1:vw1); (2,0,0,2:vw1); (2,0,0,3:vw1); (2,0,1,0:vw1); (2,0,1,1:vw1); (2,0,1,2:vw1); (2,0,1,3:vw1); (2,0,2,0:vw1); (2,0,2,1:vw1); (2,0,2,2:vw1); (2,0,2,3:vw1); (2,0,3,0:vw1); (2,0,3,1:vw1); (2,0,3,2:vw1); (2,0,3,3:vw1); (3,0,0,0:vw1); (3,0,0,1:vw1); (3,0,0,2:vw1); (3,0,0,3:vw1); (3,0,1,0:vw1); (3,0,1,1:vw1); (3,0,1,2:vw1); (3,0,1,3:vw1); (3,0,2,0:vw1); (3,0,2,1:vw1); (3,0,2,2:vw1); (3,0,2,3:vw1); (3,0,3,0:vw1); (3,0,3,1:vw1); (3,0,3,2:vw1); (3,0,3,3:vw1); (4,0,0,0:vw1); (4,0,0,1:vw1); (4,0,0,2:vw1); (4,0,0,3:vw1); (4,0,1,0:vw1); (4,0,1,1:vw1); (4,0,1,2:vw1); (4,0,1,3:vw1); (4,0,2,0:vw1); (4,0,2,1:vw1); (4,0,2,2:vw1); (4,0,2,3:vw1); (4,0,3,0:vw1); (4,0,3,1:vw1); (4,0,3,2:vw1); (4,0,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v26, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v107, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v26, v107, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v108, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v26, v108, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v109, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v26, v109, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v110, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v26, v110, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v111, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v26, v111, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v112, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v26, v112, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v113, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v26, v113, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v114, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v26, v114, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v115, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v26, v115, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v116, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v26, v116, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v117, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v26, v117, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v118, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v26, v118, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v119, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v26, v119, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v120, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v26, v120, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v121, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v26, v121, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v122, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v26, v122, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v123, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v26, v123, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v124, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v124, v26, v124, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v125, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v26, v125, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v126, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v26, v126, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v127, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v26, v127, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v128, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v26, v128, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v129, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v26, v129, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v130, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v26, v130, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v131, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v26, v131, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v132, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v26, v132, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v133, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v26, v133, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v134, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v26, v134, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v135, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v26, v135, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v136, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v26, v136, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v137, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v26, v137, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v138, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v26, v138, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v139, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v26, v139, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v140, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v26, v140, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v141, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v26, v141, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v142, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v26, v142, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v143, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v26, v143, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v144, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v26, v144, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v145, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v26, v145, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v146, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v26, v146, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v147, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v26, v147, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v148, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v26, v148, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v149, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v26, v149, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v150, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v26, v150, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v151, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v26, v151, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v152, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v26, v152, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v153, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v26, v153, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v154, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v26, v154, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v155, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v26, v155, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v156, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v26, v156, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v157, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v26, v157, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v158, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v26, v158, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v159, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v26, v159, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v160, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v26, v160, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v161, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v26, v161, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v162, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v26, v162, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v163, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v26, v163, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v201, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v26, v201, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v202, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v26, v202, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v203, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v26, v203, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v204, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v26, v204, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v205, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v205, v26, v205, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v206, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v26, v206, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v207, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v207, v26, v207, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v208, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v26, v208, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v209, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v209, v26, v209, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v210, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v26, v210, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v211, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v211, v26, v211, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v212, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v26, v212, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v213, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v213, v26, v213, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v214, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v26, v214, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v215, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v215, v26, v215, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v216, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v26, v216, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v217, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v217, v26, v217, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v218, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v26, v218, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v219, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v219, v26, v219, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v220, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v26, v220, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v221, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v221, v26, v221, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v222, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v26, v222, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v223, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v223, v26, v223, s[72:73]            // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+27], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+28], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+29], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+30], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+31], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+32], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+33], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+34], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+35], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+36], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+37], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+38], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+39], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+40], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+41], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+42], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+43], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+44], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+45], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+46], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+47], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+48], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+49], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+50], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+51], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+52], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+53], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+54], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+55], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+56], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+57], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+58], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+59], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+60], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+61], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+62], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+63], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+64], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+65], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+66], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+67], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+68], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+69], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+70], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+71], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+72], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+73], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+74], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+75], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+76], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+77], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+78], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+79], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+80], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+81], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+82], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+83], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+84], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+85], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+86], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+87], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+88], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+89], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+90], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+91], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+92], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+93], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+94], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+95], acc65          // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+96], acc69          // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+97], acc73          // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+98], acc77          // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+99], acc66          // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+100], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+101], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+102], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+103], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+104], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+105], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+106], acc79         // copy acc to vreg[79]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3), (1, 0, 1, 0), (1, 0, 1, 1), (1, 0, 1, 2), (1, 0, 1, 3), (1, 0, 2, 0), (1, 0, 2, 1), (1, 0, 2, 2), (1, 0, 2, 3), (1, 0, 3, 0), (1, 0, 3, 1), (1, 0, 3, 2), (1, 0, 3, 3), (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 0, 2), (2, 0, 0, 3), (2, 0, 1, 0), (2, 0, 1, 1), (2, 0, 1, 2), (2, 0, 1, 3), (2, 0, 2, 0), (2, 0, 2, 1), (2, 0, 2, 2), (2, 0, 2, 3), (2, 0, 3, 0), (2, 0, 3, 1), (2, 0, 3, 2), (2, 0, 3, 3), (3, 0, 0, 0), (3, 0, 0, 1), (3, 0, 0, 2), (3, 0, 0, 3), (3, 0, 1, 0), (3, 0, 1, 1), (3, 0, 1, 2), (3, 0, 1, 3), (3, 0, 2, 0), (3, 0, 2, 1), (3, 0, 2, 2), (3, 0, 2, 3), (3, 0, 3, 0), (3, 0, 3, 1), (3, 0, 3, 2), (3, 0, 3, 3), (4, 0, 0, 0), (4, 0, 0, 1), (4, 0, 0, 2), (4, 0, 0, 3), (4, 0, 1, 0), (4, 0, 1, 1), (4, 0, 1, 2), (4, 0, 1, 3), (4, 0, 2, 0), (4, 0, 2, 1), (4, 0, 2, 2), (4, 0, 2, 3), (4, 0, 3, 0), (4, 0, 3, 1), (4, 0, 3, 2), (4, 0, 3, 3)] */
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha

/* apply mask, calc new C and issue writes */
buffer_store_dword v27, v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v28, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v29, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v30, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v31, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v32, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v33, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v34, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v35, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v36, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v37, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v38, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v39, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v40, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v41, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v42, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v43, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v44, v124, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v45, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v46, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v47, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v48, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v49, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v50, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v51, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v52, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v53, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v54, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v55, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v56, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v57, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v58, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v59, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v60, v140, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v61, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v62, v142, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v63, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v64, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v65, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v66, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v67, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v68, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v69, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v70, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v71, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v72, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v73, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v74, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v75, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v76, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v77, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v78, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v79, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v80, v160, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v81, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v82, v162, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v83, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v84, v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v85, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v86, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v87, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v88, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v89, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v90, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v91, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v92, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v93, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v94, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v95, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v96, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v97, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v98, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v99, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v100, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v101, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v102, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v103, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v104, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v105, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
buffer_store_dword v106, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:
s_and_b32 s68, 127, s[sgprSizeI]                   // s68 = s[sgprSizeI] % 128
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Else       // jump if edges required
s_mov_b32 s71, 0                                   // STATIC_DIV: divisor=160
s_mul_i32 s70, 819, s[sgprSizeJ]                   // tmp1 = dividend * magic hi
s_lshl_b64 s[70:71], s[70:71], 16                  // left shift 16 bits
s_mul_i32 s69, s[sgprSizeJ], 13108                 // tmp0 = dividend * magic lo
s_add_u32 s70, s69, s70                            // add lo
s_addc_u32 s71, s71, 0                             // add hi
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp1 = (dividend * magic) << shift
s_mov_b32 s69, s70                                 // quotient
s_mul_i32 s70, s69, 160                            // quotient*divisor
s_sub_u32 s68, s[sgprSizeJ], s70                   // rReg = dividend - quotient*divisor
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=22 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (1,0,0,0:vw4); (1,0,1,0:vw4); (1,0,2,0:vw4); (1,0,3,0:vw4); (2,0,0,0:vw4); (2,0,1,0:vw4); (2,0,2,0:vw4); (2,0,3,0:vw4); (3,0,0,0:vw4); (3,0,1,0:vw4); (3,0,2,0:vw4); (3,0,3,0:vw4); (4,0,0,0:vw4); (4,0,1,0:vw4); (4,0,2,0:vw4); (4,0,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v28, v22, v20, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=20, coord0Vgpr=20 (multiple bpe)
buffer_load_dwordx4 v[112:115], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[116:119], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[120:123], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[124:127], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 116                // scale StrideC *= numRows(29) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[128:131], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[132:135], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[136:139], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[140:143], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 116                // scale StrideC *= numRows(29) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[144:147], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[148:151], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[152:155], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[156:159], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 116                // scale StrideC *= numRows(29) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[160:163], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[204:207], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[208:211], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[212:215], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 116                // scale StrideC *= numRows(29) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[216:219], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[220:223], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[224:227], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dwordx4 v[228:231], v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v27, v23, v20, 2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=20, coord0Vgpr=20 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+32], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+33], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+34], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+35], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+36], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+37], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+38], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+39], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+44], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+45], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+46], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+47], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+48], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+49], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+50], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+51], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+52], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+53], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+54], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+55], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+56], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+57], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+58], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+59], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+60], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+61], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+62], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+63], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+64], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+65], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+66], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+67], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+68], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+69], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+70], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+71], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+72], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+73], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+74], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+75], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+76], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+77], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+78], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+79], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+80], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+81], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+82], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+83], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+84], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+85], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+86], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+87], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+88], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+89], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+90], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+91], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+92], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+93], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+94], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+95], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+96], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+97], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+98], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+99], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+100], acc65         // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+101], acc69         // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+102], acc73         // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+103], acc77         // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+104], acc66         // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+105], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+106], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+107], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+108], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+109], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+110], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+111], acc79         // copy acc to vreg[79]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0), (3, 0, 0, 0), (3, 0, 1, 0), (3, 0, 2, 0), (3, 0, 3, 0), (4, 0, 0, 0), (4, 0, 1, 0), (4, 0, 2, 0), (4, 0, 3, 0)] */
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+108:vgprValuC+108+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+108:vgprValuC+108+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+110:vgprValuC+110+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+110:vgprValuC+110+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(19)                                // vlcnt(19) = 20 - 1 (beta) vscnt(0) (interleaved)
v_fmac_f32 v[vgprValuC+32], v112, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+33], v113, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+34], v114, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+35], v115, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[32:35], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(18) = 20 - 2 (beta) vscnt(1) (interleaved)
v_fmac_f32 v[vgprValuC+36], v116, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+37], v117, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+38], v118, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+39], v119, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[36:39], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(17) = 20 - 3 (beta) vscnt(2) (interleaved)
v_fmac_f32 v[vgprValuC+40], v120, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+41], v121, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+42], v122, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+43], v123, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(16) = 20 - 4 (beta) vscnt(3) (interleaved)
v_fmac_f32 v[vgprValuC+44], v124, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+45], v125, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+46], v126, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+47], v127, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[44:47], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(15) = 20 - 5 (beta) vscnt(4) (interleaved)
v_fmac_f32 v[vgprValuC+48], v128, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+49], v129, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+50], v130, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+51], v131, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(14) = 20 - 6 (beta) vscnt(5) (interleaved)
v_fmac_f32 v[vgprValuC+52], v132, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+53], v133, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+54], v134, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+55], v135, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[52:55], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(13) = 20 - 7 (beta) vscnt(6) (interleaved)
v_fmac_f32 v[vgprValuC+56], v136, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+57], v137, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+58], v138, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+59], v139, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(12) = 20 - 8 (beta) vscnt(7) (interleaved)
v_fmac_f32 v[vgprValuC+60], v140, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+61], v141, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+62], v142, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+63], v143, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[60:63], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(11) = 20 - 9 (beta) vscnt(8) (interleaved)
v_fmac_f32 v[vgprValuC+64], v144, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+65], v145, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+66], v146, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+67], v147, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(10) = 20 - 10 (beta) vscnt(9) (interleaved)
v_fmac_f32 v[vgprValuC+68], v148, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+69], v149, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+70], v150, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+71], v151, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[68:71], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(9) = 20 - 11 (beta) vscnt(10) (interleaved)
v_fmac_f32 v[vgprValuC+72], v152, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+73], v153, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+74], v154, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+75], v155, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[72:75], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(8) = 20 - 12 (beta) vscnt(11) (interleaved)
v_fmac_f32 v[vgprValuC+76], v156, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+77], v157, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+78], v158, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+79], v159, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[76:79], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(7) = 20 - 13 (beta) vscnt(12) (interleaved)
v_fmac_f32 v[vgprValuC+80], v160, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+81], v161, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+82], v162, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+83], v163, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[80:83], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(6) = 20 - 14 (beta) vscnt(13) (interleaved)
v_fmac_f32 v[vgprValuC+84], v204, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+85], v205, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+86], v206, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+87], v207, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[84:87], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(5) = 20 - 15 (beta) vscnt(14) (interleaved)
v_fmac_f32 v[vgprValuC+88], v208, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+89], v209, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+90], v210, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+91], v211, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[88:91], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(4) = 20 - 16 (beta) vscnt(15) (interleaved)
v_fmac_f32 v[vgprValuC+92], v212, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+93], v213, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+94], v214, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+95], v215, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[92:95], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(3) = 20 - 17 (beta) vscnt(16) (interleaved)
v_fmac_f32 v[vgprValuC+96], v216, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+97], v217, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+98], v218, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+99], v219, s[sgprBeta]      // finalSum = sum*alpha + C*beta
s_mul_i32 s8, s[sgprStrideD1J], 116                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[96:99], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(2) = 20 - 18 (beta) vscnt(17) (interleaved)
v_fmac_f32 v[vgprValuC+100], v220, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+101], v221, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+102], v222, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+103], v223, s[sgprBeta]     // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[100:103], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(1) = 20 - 19 (beta) vscnt(18) (interleaved)
v_fmac_f32 v[vgprValuC+104], v224, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+105], v225, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+106], v226, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+107], v227, s[sgprBeta]     // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[104:107], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D

s_waitcnt vmcnt(19)                                // vlcnt(0) = 20 - 20 (beta) vscnt(19) (interleaved)
v_fmac_f32 v[vgprValuC+108], v228, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+109], v229, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+110], v230, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+111], v231, s[sgprBeta]     // finalSum = sum*alpha + C*beta
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx4 v[108:111], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=20 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (1,0,0,0:vw4); (1,0,1,0:vw4); (1,0,2,0:vw4); (1,0,3,0:vw4); (2,0,0,0:vw4); (2,0,1,0:vw4); (2,0,2,0:vw4); (2,0,3,0:vw4); (3,0,0,0:vw4); (3,0,1,0:vw4); (3,0,2,0:vw4); (3,0,3,0:vw4); (4,0,0,0:vw4); (4,0,1,0:vw4); (4,0,2,0:vw4); (4,0,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v26, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v27, v22, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v27, v26, v27, s[72:73]              // LDC clip if OOB. offset
buffer_load_dwordx4 v[108:111], v27, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v27, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v27, v26, v27, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v116, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v26, v116, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[112:115], v116, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v116, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v26, v116, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v117, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v26, v117, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[120:123], v117, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v117, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v26, v117, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v118, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v26, v118, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[124:127], v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v118, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v26, v118, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v119, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v26, v119, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[128:131], v119, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v119, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v26, v119, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v136, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v26, v136, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[132:135], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v136, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v26, v136, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v137, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v26, v137, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[140:143], v137, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v137, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v26, v137, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v138, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v26, v138, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[144:147], v138, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v138, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v26, v138, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v139, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v26, v139, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[148:151], v139, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v139, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v26, v139, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v156, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v26, v156, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[152:155], v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v156, v26, v156, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v157, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v26, v157, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[160:163], v157, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v157, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v26, v157, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v158, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v26, v158, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[204:207], v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v158, v26, v158, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v159, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v26, v159, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[208:211], v159, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v159, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v26, v159, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v201, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v26, v201, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[212:215], v201, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v201, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v201, v26, v201, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v202, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v26, v202, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[216:219], v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v26, v202, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v203, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v26, v203, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[220:223], v203, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v203, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v203, v26, v203, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v228, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v26, v228, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[224:227], v228, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v228, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v26, v228, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v229, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v26, v229, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[232:235], v229, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v229, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v229, v26, v229, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v230, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v26, v230, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[236:239], v230, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v230, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v26, v230, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v231, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v26, v231, s[72:73]            // LDC clip if OOB. offset
buffer_load_dwordx4 v[240:243], v231, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v231, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v231, v26, v231, s[72:73]            // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+28], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+29], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+30], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+31], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+36], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+37], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+38], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+39], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+40], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+41], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+42], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+43], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+44], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+45], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+46], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+47], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+48], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+49], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+50], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+51], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+52], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+53], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+54], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+55], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+56], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+57], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+58], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+59], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+60], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+61], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+62], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+63], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+64], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+65], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+66], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+67], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+68], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+69], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+70], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+71], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+72], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+73], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+74], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+75], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+76], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+77], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+78], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+79], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+80], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+81], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+82], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+83], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+84], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+85], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+86], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+87], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+88], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+89], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+90], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+91], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+92], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+93], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+94], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+95], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+96], acc65          // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+97], acc69          // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+98], acc73          // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+99], acc77          // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+100], acc66         // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+101], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+102], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+103], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+104], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+105], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+106], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+107], acc79         // copy acc to vreg[79]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0), (3, 0, 0, 0), (3, 0, 1, 0), (3, 0, 2, 0), (3, 0, 3, 0), (4, 0, 0, 0), (4, 0, 1, 0), (4, 0, 2, 0), (4, 0, 3, 0)] */
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+90:vgprValuC+90+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+90:vgprValuC+90+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+92:vgprValuC+92+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+92:vgprValuC+92+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+94:vgprValuC+94+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+94:vgprValuC+94+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+96:vgprValuC+96+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+96:vgprValuC+96+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+98:vgprValuC+98+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+98:vgprValuC+98+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+100:vgprValuC+100+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+100:vgprValuC+100+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+102:vgprValuC+102+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+102:vgprValuC+102+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+104:vgprValuC+104+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+104:vgprValuC+104+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+106:vgprValuC+106+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+106:vgprValuC+106+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+28], v108, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+29], v109, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+30], v110, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+31], v111, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[28:31], v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+32], v112, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+33], v113, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+34], v114, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+35], v115, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[32:35], v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+36], v120, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+37], v121, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+38], v122, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+39], v123, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[36:39], v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+40], v124, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+41], v125, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+42], v126, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+43], v127, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[40:43], v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+44], v128, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+45], v129, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+46], v130, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+47], v131, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[44:47], v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+48], v132, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+49], v133, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+50], v134, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+51], v135, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[48:51], v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+52], v140, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+53], v141, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+54], v142, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+55], v143, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[52:55], v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+56], v144, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+57], v145, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+58], v146, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+59], v147, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[56:59], v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+60], v148, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+61], v149, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+62], v150, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+63], v151, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[60:63], v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+64], v152, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+65], v153, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+66], v154, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+67], v155, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[64:67], v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+68], v160, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+69], v161, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+70], v162, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+71], v163, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[68:71], v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+72], v204, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+73], v205, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+74], v206, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+75], v207, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[72:75], v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+76], v208, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+77], v209, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+78], v210, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+79], v211, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[76:79], v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+80], v212, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+81], v213, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+82], v214, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+83], v215, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[80:83], v201, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+84], v216, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+85], v217, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+86], v218, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+87], v219, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[84:87], v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+88], v220, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+89], v221, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+90], v222, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+91], v223, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[88:91], v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+92], v224, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+93], v225, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+94], v226, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+95], v227, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[92:95], v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+96], v232, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+97], v233, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+98], v234, s[sgprBeta]      // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+99], v235, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[96:99], v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+100], v236, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+101], v237, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+102], v238, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+103], v239, s[sgprBeta]     // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[100:103], v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+104], v240, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+105], v241, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+106], v242, s[sgprBeta]     // finalSum = sum*alpha + C*beta
v_fmac_f32 v[vgprValuC+107], v243, s[sgprBeta]     // finalSum = sum*alpha + C*beta
buffer_store_dwordx4 v[104:107], v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=63 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (1,0,0,0:vw1); (1,0,0,1:vw1); (1,0,0,2:vw1); (1,0,0,3:vw1); (1,0,1,0:vw1); (1,0,1,1:vw1); (1,0,1,2:vw1); (1,0,1,3:vw1); (1,0,2,0:vw1); (1,0,2,1:vw1); (1,0,2,2:vw1); (1,0,2,3:vw1); (1,0,3,0:vw1); (1,0,3,1:vw1); (1,0,3,2:vw1); (1,0,3,3:vw1); (2,0,0,0:vw1); (2,0,0,1:vw1); (2,0,0,2:vw1); (2,0,0,3:vw1); (2,0,1,0:vw1); (2,0,1,1:vw1); (2,0,1,2:vw1); (2,0,1,3:vw1); (2,0,2,0:vw1); (2,0,2,1:vw1); (2,0,2,2:vw1); (2,0,2,3:vw1); (2,0,3,0:vw1); (2,0,3,1:vw1); (2,0,3,2:vw1); (2,0,3,3:vw1); (3,0,0,0:vw1); (3,0,0,1:vw1); (3,0,0,2:vw1); (3,0,0,3:vw1); (3,0,1,0:vw1); (3,0,1,1:vw1); (3,0,1,2:vw1); (3,0,1,3:vw1); (3,0,2,0:vw1); (3,0,2,1:vw1); (3,0,2,2:vw1); (3,0,2,3:vw1); (3,0,3,0:vw1); (3,0,3,1:vw1); (3,0,3,2:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v26, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v91, v22, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v91, v26, v91, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v90, v91, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v91, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v91, v26, v91, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v93, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v26, v93, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v92, v93, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v93, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v26, v93, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v95, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v95, v26, v95, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v94, v95, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v95, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v95, v26, v95, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v97, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v97, v26, v97, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v96, v97, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v97, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v97, v26, v97, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v99, v22, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v26, v99, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v98, v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v99, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v26, v99, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v101, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v26, v101, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v101, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v26, v101, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v103, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v103, v26, v103, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v102, v103, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v103, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v103, v26, v103, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v105, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v26, v105, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v104, v105, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v105, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v26, v105, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v107, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v26, v107, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v106, v107, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v107, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v26, v107, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v109, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v26, v109, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v108, v109, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v109, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v26, v109, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v111, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v26, v111, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v110, v111, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v111, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v26, v111, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v113, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v26, v113, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v112, v113, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v113, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v26, v113, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v115, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v26, v115, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v114, v115, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v115, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v26, v115, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v117, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v26, v117, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v116, v117, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v117, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v26, v117, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v119, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v26, v119, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v118, v119, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v119, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v26, v119, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v121, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v26, v121, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v120, v121, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v121, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v26, v121, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v123, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v26, v123, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v122, v123, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v123, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v26, v123, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v125, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v26, v125, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v124, v125, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v125, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v26, v125, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v127, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v26, v127, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v126, v127, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v127, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v26, v127, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v129, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v26, v129, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v128, v129, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v129, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v26, v129, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v131, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v26, v131, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v130, v131, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v131, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v26, v131, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v133, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v26, v133, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v132, v133, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v133, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v26, v133, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v135, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v26, v135, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v134, v135, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v135, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v26, v135, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v137, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v26, v137, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v136, v137, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v137, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v26, v137, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v139, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v26, v139, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v138, v139, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v139, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v26, v139, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v141, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v26, v141, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v140, v141, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v141, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v26, v141, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v143, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v26, v143, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v142, v143, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v143, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v26, v143, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v145, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v26, v145, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v144, v145, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v145, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v26, v145, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v147, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v26, v147, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v146, v147, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v147, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v26, v147, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v149, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v26, v149, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v148, v149, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v149, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v26, v149, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v151, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v26, v151, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v150, v151, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v151, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v26, v151, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v153, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v26, v153, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v152, v153, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v153, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v26, v153, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v155, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v26, v155, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v154, v155, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v155, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v26, v155, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v157, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v26, v157, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v156, v157, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v157, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v26, v157, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v159, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v26, v159, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v158, v159, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v159, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v26, v159, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v161, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v26, v161, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v160, v161, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v161, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v26, v161, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v163, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v26, v163, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v162, v163, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v163, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v26, v163, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v202, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v26, v202, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v201, v202, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v202, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v202, v26, v202, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v204, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v26, v204, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v203, v204, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v204, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v204, v26, v204, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v206, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v26, v206, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v205, v206, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v206, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v206, v26, v206, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v208, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v26, v208, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v207, v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v208, v26, v208, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v210, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v26, v210, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v209, v210, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v210, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v210, v26, v210, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v212, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v26, v212, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v211, v212, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v212, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v212, v26, v212, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v214, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v26, v214, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v213, v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v214, v26, v214, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v216, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v26, v216, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v215, v216, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v216, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v216, v26, v216, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v218, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v26, v218, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v217, v218, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v218, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v218, v26, v218, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v220, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v26, v220, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v219, v220, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v220, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v220, v26, v220, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v222, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v26, v222, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v221, v222, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v222, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v222, v26, v222, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v224, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v26, v224, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v223, v224, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v224, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v224, v26, v224, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v226, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v26, v226, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v225, v226, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v226, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v226, v26, v226, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v228, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v26, v228, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v227, v228, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v228, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v228, v26, v228, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v230, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v26, v230, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v229, v230, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v230, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v230, v26, v230, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v232, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v26, v232, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v231, v232, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v232, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v232, v26, v232, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v234, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v26, v234, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v233, v234, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v234, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v234, v26, v234, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v236, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v26, v236, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v235, v236, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v236, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v236, v26, v236, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v238, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v26, v238, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v237, v238, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v238, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v238, v26, v238, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v240, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v26, v240, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v239, v240, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v240, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v240, v26, v240, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v242, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v26, v242, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v241, v242, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v242, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v242, v26, v242, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v244, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v26, v244, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v243, v244, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v244, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v244, v26, v244, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v246, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v26, v246, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v245, v246, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v246, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v246, v26, v246, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v248, v22, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v26, v248, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v247, v248, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v248, v23, v20, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v248, v26, v248, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v250, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v26, v250, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v249, v250, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v250, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v250, v26, v250, s[72:73]            // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v252, v22, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v252, v26, v252, s[72:73]            // LDC clip if OOB. offset
buffer_load_dword v251, v252, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v252, v23, v24, 2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v252, v26, v252, s[72:73]            // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+27], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+28], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+29], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+30], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+31], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+32], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+33], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+34], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+35], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+36], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+37], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+38], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+39], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+40], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+41], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+42], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+43], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+44], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+45], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+46], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+47], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+48], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+49], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+50], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+51], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+52], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+53], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+54], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+55], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+56], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+57], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+58], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+59], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+60], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+61], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+62], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+63], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+64], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+65], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+66], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+67], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+68], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+69], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+70], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+71], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+72], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+73], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+74], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+75], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+76], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+77], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+78], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+79], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+80], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+81], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+82], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+83], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+84], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+85], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+86], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+87], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+88], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+89], acc59          // copy acc to vreg[62]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 3), (1, 0, 1, 0), (1, 0, 1, 1), (1, 0, 1, 2), (1, 0, 1, 3), (1, 0, 2, 0), (1, 0, 2, 1), (1, 0, 2, 2), (1, 0, 2, 3), (1, 0, 3, 0), (1, 0, 3, 1), (1, 0, 3, 2), (1, 0, 3, 3), (2, 0, 0, 0), (2, 0, 0, 1), (2, 0, 0, 2), (2, 0, 0, 3), (2, 0, 1, 0), (2, 0, 1, 1), (2, 0, 1, 2), (2, 0, 1, 3), (2, 0, 2, 0), (2, 0, 2, 1), (2, 0, 2, 2), (2, 0, 2, 3), (2, 0, 3, 0), (2, 0, 3, 1), (2, 0, 3, 2), (2, 0, 3, 3), (3, 0, 0, 0), (3, 0, 0, 1), (3, 0, 0, 2), (3, 0, 0, 3), (3, 0, 1, 0), (3, 0, 1, 1), (3, 0, 1, 2), (3, 0, 1, 3), (3, 0, 2, 0), (3, 0, 2, 1), (3, 0, 2, 2), (3, 0, 2, 3), (3, 0, 3, 0), (3, 0, 3, 1), (3, 0, 3, 2)] */
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+80:vgprValuC+80+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+80:vgprValuC+80+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+82:vgprValuC+82+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+82:vgprValuC+82+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+84:vgprValuC+84+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+84:vgprValuC+84+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+86:vgprValuC+86+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+86:vgprValuC+86+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+88:vgprValuC+88+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+88:vgprValuC+88+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+27], v90, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v27, v91, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+28], v92, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v28, v93, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+29], v94, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v29, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+30], v96, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v30, v97, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+31], v98, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v31, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+32], v100, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v32, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+33], v102, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v33, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+34], v104, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v34, v105, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+35], v106, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v35, v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+36], v108, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v36, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+37], v110, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v37, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+38], v112, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v38, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+39], v114, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v39, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+40], v116, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v40, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+41], v118, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v41, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+42], v120, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v42, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+43], v122, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v43, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+44], v124, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v44, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+45], v126, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v45, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+46], v128, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v46, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+47], v130, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v47, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+48], v132, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v48, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+49], v134, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v49, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+50], v136, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v50, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+51], v138, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v51, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+52], v140, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v52, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+53], v142, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v53, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+54], v144, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v54, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+55], v146, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v55, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+56], v148, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v56, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+57], v150, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v57, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+58], v152, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v58, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+59], v154, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v59, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+60], v156, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v60, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+61], v158, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v61, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+62], v160, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v62, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+63], v162, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v63, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+64], v201, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v64, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+65], v203, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v65, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+66], v205, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v66, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+67], v207, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v67, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+68], v209, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v68, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+69], v211, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v69, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+70], v213, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v70, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+71], v215, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v71, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+72], v217, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v72, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+73], v219, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v73, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+74], v221, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v74, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+75], v223, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v75, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+76], v225, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v76, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+77], v227, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v77, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+78], v229, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v78, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+79], v231, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v79, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+80], v233, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v80, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+81], v235, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v81, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+82], v237, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v82, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+83], v239, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v83, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+84], v241, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v84, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+85], v243, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v85, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+86], v245, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v86, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+87], v247, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v87, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+88], v249, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v88, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+89], v251, s[sgprBeta]      // finalSum = sum*alpha + C*beta
buffer_store_dword v89, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (3,0,3,3:vw1); (4,0,0,0:vw1); (4,0,0,1:vw1); (4,0,0,2:vw1); (4,0,0,3:vw1); (4,0,1,0:vw1); (4,0,1,1:vw1); (4,0,1,2:vw1); (4,0,1,3:vw1); (4,0,2,0:vw1); (4,0,2,1:vw1); (4,0,2,2:vw1); (4,0,2,3:vw1); (4,0,3,0:vw1); (4,0,3,1:vw1); (4,0,3,2:vw1); (4,0,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v26, BufferOOB
/* (d1,vc1,d0,vc0)=(3,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v45, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v45, v26, v45, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v44, v45, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v45, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v45, v26, v45, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v21, vcc, v21, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s68, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v22, v22, s68                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s68, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v23, v23, s68                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v47, v22, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v47, v26, v47, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v46, v47, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v47, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v47, v26, v47, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v49, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v49, v26, v49, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v48, v49, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v49, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v49, v26, v49, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v51, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v51, v26, v51, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v50, v51, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v51, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v51, v26, v51, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,0,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v53, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v53, v26, v53, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v52, v53, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v53, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v53, v26, v53, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v55, v22, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v55, v26, v55, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v54, v55, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v55, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v55, v26, v55, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v57, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v57, v26, v57, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v56, v57, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v57, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v57, v26, v57, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v59, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v59, v26, v59, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v58, v59, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v59, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v59, v26, v59, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,1,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v61, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v61, v26, v61, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v60, v61, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v61, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v61, v26, v61, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v63, v22, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v63, v26, v63, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v62, v63, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v63, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v63, v26, v63, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v65, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v65, v26, v65, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v64, v65, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v65, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v65, v26, v65, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v67, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v67, v26, v67, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v66, v67, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v67, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v67, v26, v67, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,2,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v69, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v69, v26, v69, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v68, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v69, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v69, v26, v69, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,0) */
v_add_co_u32 v21, vcc, v21, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v22, v22, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v23, v23, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[68:69], v20, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v71, v22, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v26, v71, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v70, v71, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v71, v23, v20, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v26, v71, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,1) */
v_add_co_u32 v24, vcc, v20, 1                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v73, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v26, v73, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v72, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v73, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v26, v73, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,2) */
v_add_co_u32 v24, vcc, v20, 2                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v75, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v26, v75, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v74, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v75, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v26, v75, s[72:73]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(4,3,0,3) */
v_add_co_u32 v24, vcc, v20, 3                      // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[68:69], v24, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v21, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[68:69], s[72:73]             // in0 && in1
v_add_lshl_u32 v77, v22, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v26, v77, s[72:73]              // LDC clip if OOB. offset
buffer_load_dword v76, v77, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v77, v23, v24, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v26, v77, s[72:73]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+27], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+28], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+29], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+30], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+31], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+32], acc65          // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+33], acc69          // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+34], acc73          // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+35], acc77          // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+36], acc66          // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+37], acc70          // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+38], acc74          // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+39], acc78          // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+40], acc67          // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+41], acc71          // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+42], acc75          // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+43], acc79          // copy acc to vreg[79]

/* rC *= alpha batchElements=[(3, 0, 3, 3), (4, 0, 0, 0), (4, 0, 0, 1), (4, 0, 0, 2), (4, 0, 0, 3), (4, 0, 1, 0), (4, 0, 1, 1), (4, 0, 1, 2), (4, 0, 1, 3), (4, 0, 2, 0), (4, 0, 2, 1), (4, 0, 2, 2), (4, 0, 2, 3), (4, 0, 3, 0), (4, 0, 3, 1), (4, 0, 3, 2), (4, 0, 3, 3)] */
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+28:vgprValuC+28+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+30:vgprValuC+30+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+32:vgprValuC+32+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+32:vgprValuC+32+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+34:vgprValuC+34+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+36:vgprValuC+36+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+36:vgprValuC+36+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+38:vgprValuC+38+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+38:vgprValuC+38+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+40:vgprValuC+40+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+40:vgprValuC+40+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+42:vgprValuC+42+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+42:vgprValuC+42+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+27], v44, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v27, v45, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+28], v46, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v28, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+29], v48, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v29, v49, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+30], v50, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v30, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+31], v52, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v31, v53, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+32], v54, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v32, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+33], v56, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v33, v57, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+34], v58, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v34, v59, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+35], v60, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v35, v61, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+36], v62, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v36, v63, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+37], v64, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v37, v65, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+38], v66, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v38, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+39], v68, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v39, v69, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+40], v70, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v40, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+41], v72, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v41, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+42], v74, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v42, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
v_fmac_f32 v[vgprValuC+43], v76, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_dword v43, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 nt // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s8, 0x14000, s[sgprStreamKIdx]           // Offset to correct partials tile
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s8       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], 0       // add hi to SRD

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=46 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (1,0,0,0:vw4); (1,0,1,0:vw4); (1,0,2,0:vw4); (1,0,3,0:vw4); (2,0,0,0:vw4); (2,0,1,0:vw4); (2,0,2,0:vw4); (2,0,3,0:vw4); (3,0,0,0:vw4); (3,0,1,0:vw4); (3,0,2,0:vw4); (3,0,3,0:vw4); (4,0,0,0:vw4); (4,0,1,0:vw4); (4,0,2,0:vw4); (4,0,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_accvgpr_read_b32 v[vgprValuC+32], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+33], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+34], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+35], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+36], acc1           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+37], acc5           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+38], acc9           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+39], acc13          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+40], acc2           // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+41], acc6           // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+42], acc10          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+43], acc14          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+44], acc3           // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+45], acc7           // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+46], acc11          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+47], acc15          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+48], acc16          // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+49], acc20          // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+50], acc24          // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+51], acc28          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+52], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+53], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+54], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+55], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+56], acc18          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+57], acc22          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+58], acc26          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+59], acc30          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+60], acc19          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+61], acc23          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+62], acc27          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+63], acc31          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+64], acc32          // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+65], acc36          // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+66], acc40          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+67], acc44          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+68], acc33          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+69], acc37          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+70], acc41          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+71], acc45          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+72], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+73], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+74], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+75], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+76], acc35          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+77], acc39          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+78], acc43          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+79], acc47          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+80], acc48          // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+81], acc52          // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+82], acc56          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+83], acc60          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+84], acc49          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+85], acc53          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+86], acc57          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+87], acc61          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+88], acc50          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+89], acc54          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+90], acc58          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+91], acc62          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+92], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+93], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+94], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+95], acc63          // copy acc to vreg[63]
v_accvgpr_read_b32 v[vgprValuC+96], acc64          // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+97], acc68          // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+98], acc72          // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+99], acc76          // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+100], acc65         // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+101], acc69         // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+102], acc73         // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+103], acc77         // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+104], acc66         // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+105], acc70         // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+106], acc74         // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+107], acc78         // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+108], acc67         // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+109], acc71         // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+110], acc75         // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+111], acc79         // copy acc to vreg[79]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_lshlrev_b32 v27, 4, v[vgprSerial]                // v27 = v[vgprSerial] * 16
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_dwordx4 v[32:35], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[36:39], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[40:43], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[44:47], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[48:51], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[52:55], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[56:59], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[60:63], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[64:67], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[68:71], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[72:75], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[76:79], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[80:83], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[84:87], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[88:91], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[92:95], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[96:99], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[100:103], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[104:107], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[108:111], v27, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 nt // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt vmcnt(0)                                 // wait for data store
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprStreamKIdx], 2                // flag offset based on CTA index
v_readfirstlane_b32 s61, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s61, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s61, 1                                   // flag data
s_store_dword s61, s[sgprAddressFlags:sgprAddressFlags+1], s8 glc // set flag
label_SK_SkipFlagSet:
s_waitcnt lgkmcnt(0)                               // wait for flag
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_SK_CloseLoop:
s_cmp_ge_u32 s[sgprStreamKIter], s[sgprStreamKIterEnd] // Check if done all StreamK iterations
s_cbranch_scc1 label_NoBranch_IXPKU979JKZCQDH3     // Only branch on scc0
s_getpc_b64 s[68:69]                               // addr of next instr
s_add_i32 s70, label_PersistentLoopStart, 4        // target branch offset
s_abs_i32 s70, s70                                 // abs offset
s_sub_u32 s68, s68, s70                            // sub target branch offset
s_subb_u32 s69, s69, 0                             // sub high and carry
s_setpc_b64 s[68:69]                               // branch to label_PersistentLoopStart
label_NoBranch_IXPKU979JKZCQDH3:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
