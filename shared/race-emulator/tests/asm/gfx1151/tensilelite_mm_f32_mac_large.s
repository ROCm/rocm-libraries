
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Cijk_Ailk_Bljk_S_B_UserArgs_MT32x32x16_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT4_4_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1
.globl Cijk_Ailk_Bljk_S_B_UserArgs_MT32x32x16_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT4_4_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1
.p2align 8
.type Cijk_Ailk_Bljk_S_B_UserArgs_MT32x32x16_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT4_4_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_S_B_UserArgs_MT32x32x16_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT4_4_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 96 // vgprs
  .amdhsa_next_free_sgpr 84 // sgprs
  .amdhsa_group_segment_fixed_size 12352 // lds bytes
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
/* Num VGPR   =96 */
/* Num AccVGPR=0 */
/* Num SGPR   =84 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 4 x 4 */
/* SubGroup= 8 x 8 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=1, GlobalReadVectorWidthB=1 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Cijk_Ailk_Bljk_S_B_UserArgs_MT32x32x16_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT4_4_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1
    .symbol: 'Cijk_Ailk_Bljk_S_B_UserArgs_MT32x32x16_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT4_4_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1.kd'
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
    .group_segment_fixed_size:   12352
    .kernarg_segment_align:      8
    .kernarg_segment_size:       104
    .max_flat_workgroup_size:    64
    .private_segment_fixed_size: 0
    .sgpr_count:                 84
    .sgpr_spill_count:           0
    .vgpr_count:                 96
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_S_B_UserArgs_MT32x32x16_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT4_4_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1:
label_ASM_Start:  /// Main body of the asm kernel
.macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\vgprDstIdx+0], \dividend, \magicA
    v_add_nc_u32 v[\vgprDstIdx+0], v[\vgprDstIdx+0], v[\vgprDstIdx+1]
    v_lshrrev_b32 v[\vgprDstIdx+0], \magicShift, v[\vgprDstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-16),  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 24
.set vgprLocalWriteAddrA, 18
.set vgprLocalWriteAddrB, 19
.set vgprGlobalReadOffsetA, 16
.set vgprGlobalReadOffsetB, 17
.set vgprLocalReadAddrA, 20
.set vgprLocalReadAddrB, 21
.set vgprSerial, 58

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+9
.set vgprG2LA_BASE, vgprBase+18
.set vgprG2LB_BASE, vgprBase+26
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+4
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+4
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB, vgprG2LB_BASE+0

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
.set sgprGSU, 46

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
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 32
.set MT1, 32
.set DepthU, 16
.set BpeA, 4
.set BpeALog2, 2
.set BpeB, 4
.set BpeBLog2, 2
.set BpeAGR, 4
.set BpeAGRLog2, 2
.set BpeBGR, 4
.set BpeBGRLog2, 2
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 1
.set SrdShiftLeftB, 1
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x31004000                        */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* format (7b): 4                         */
/* _unusedA (2b): 0                       */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* resource_level (1b): 1                 */
/* _unusedB (1b): 0                       */
/* LLC_noalloc (2b): 0                    */
/* oob_select (2b): 3                     */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x31004000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req, vgprOffset0I:req, vgprOffsetL:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]   // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 2, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]   // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 2, v[\vgprAddr+0]    // offset *= bytes/element
.endm
.macro MAC_4x4_X0
    // Component.MAC.MAC_F32_Plain
    v_fmac_f32 v[vgprValuC+0+0], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0]
    s_setprio 1                                        // Raise priority while processing macs
    v_fmac_f32 v[vgprValuC+1+0], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0]
    v_fmac_f32 v[vgprValuC+2+0], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+0]
    v_fmac_f32 v[vgprValuC+3+0], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+0]
    v_fmac_f32 v[vgprValuC+0+4], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+1]
    v_fmac_f32 v[vgprValuC+1+4], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+1]
    v_fmac_f32 v[vgprValuC+2+4], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+1]
    v_fmac_f32 v[vgprValuC+3+4], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+1]
    v_fmac_f32 v[vgprValuC+0+8], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+2]
    v_fmac_f32 v[vgprValuC+1+8], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+2]
    v_fmac_f32 v[vgprValuC+2+8], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+2]
    v_fmac_f32 v[vgprValuC+3+8], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+2]
    v_fmac_f32 v[vgprValuC+0+12], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+3]
    v_fmac_f32 v[vgprValuC+1+12], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+3]
    v_fmac_f32 v[vgprValuC+2+12], v[vgprValuA_X0_I0+2], v[vgprValuB_X0_I0+3]
    v_fmac_f32 v[vgprValuC+3+12], v[vgprValuA_X0_I0+3], v[vgprValuB_X0_I0+3]
    s_setprio 0                                        // Reset priority after macs
.endm
.macro MAC_4x4_X1
    // Component.MAC.MAC_F32_Plain
    v_fmac_f32 v[vgprValuC+0+0], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+0]
    s_setprio 1                                        // Raise priority while processing macs
    v_fmac_f32 v[vgprValuC+1+0], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+0]
    v_fmac_f32 v[vgprValuC+2+0], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+0]
    v_fmac_f32 v[vgprValuC+3+0], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+0]
    v_fmac_f32 v[vgprValuC+0+4], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+1]
    v_fmac_f32 v[vgprValuC+1+4], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+1]
    v_fmac_f32 v[vgprValuC+2+4], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+1]
    v_fmac_f32 v[vgprValuC+3+4], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+1]
    v_fmac_f32 v[vgprValuC+0+8], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+2]
    v_fmac_f32 v[vgprValuC+1+8], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+2]
    v_fmac_f32 v[vgprValuC+2+8], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+2]
    v_fmac_f32 v[vgprValuC+3+8], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+2]
    v_fmac_f32 v[vgprValuC+0+12], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+3]
    v_fmac_f32 v[vgprValuC+1+12], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+3]
    v_fmac_f32 v[vgprValuC+2+12], v[vgprValuA_X1_I0+2], v[vgprValuB_X1_I0+3]
    v_fmac_f32 v[vgprValuC+3+12], v[vgprValuA_X1_I0+3], v[vgprValuB_X1_I0+3]
    s_setprio 0                                        // Reset priority after macs
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_b32 s20, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s22, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s23, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s21, s20, 0x1e                          // Get arg type
s_and_b32 s20, 0x3fffffff, s20                     // Get nums of gemm
s_cmp_eq_u32 s21, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_b128 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_and_b32 s[sgprStaggerU], s22, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s22, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s21
s_mov_b32 m0, 0x3040                               // LDS clamp at 12352 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s52, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s52, s52                             // Get log(WGMXCC)
s_lshr_b32 s53, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s52, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s49, s23, s52
s_lshl_b32 s49, s49, s52
s_cmp_ge_u32 s[sgprWorkGroup0], s49
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s53, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s49, s[sgprWorkGroup0], s52
s_bfm_b32 s50, s52, 0
s_and_b32 s50, s[sgprWorkGroup0], s50
s_lshr_b32 s51, s23, s52
s_mul_i32 s50, s50, s51
s_add_u32 s[sgprWorkGroup0], s49, s50
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[6:7], s53                          // s49 = s[sgprWorkGroup0] / s53
v_rcp_f64 v[6:7], v[6:7]                           // s49 = s[sgprWorkGroup0] / s53
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s49 = s[sgprWorkGroup0] / s53
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s49 = s[sgprWorkGroup0] / s53
v_cvt_u32_f64 v6, v[6:7]                           // s49 = s[sgprWorkGroup0] / s53
v_mul_lo_u32 v7, v6, s53                           // s49 = s[sgprWorkGroup0] / s53
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s49 = s[sgprWorkGroup0] / s53
v_cmp_ge_u32 vcc_lo, v8, s53                       // s49 = s[sgprWorkGroup0] / s53
s_mov_b32 exec_lo vcc_lo                           // s49 = s[sgprWorkGroup0] / s53
v_add_nc_u32 v6, v6, 1                             // s49 = s[sgprWorkGroup0] / s53
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s53                           // s49 = s[sgprWorkGroup0] / s53
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s49 = s[sgprWorkGroup0] / s53
v_readfirstlane_b32 s49, v6                        // quotient
v_readfirstlane_b32 s50, v8                        // remainder
s_mul_i32 s49, s49, s53
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s50, s50, s52
/* temp0 = temp0 + temp1 */
s_add_u32 s49, s49, s50
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[6:7], s53                          // s50 = s23 / s53
v_rcp_f64 v[6:7], v[6:7]                           // s50 = s23 / s53
v_cvt_f64_u32 v[8:9], s23                          // s50 = s23 / s53
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s50 = s23 / s53
v_cvt_u32_f64 v6, v[6:7]                           // s50 = s23 / s53
v_mul_lo_u32 v7, v6, s53                           // s50 = s23 / s53
v_sub_nc_u32 v8, s23, v7                           // s50 = s23 / s53
v_cmp_ge_u32 vcc_lo, v8, s53                       // s50 = s23 / s53
s_mov_b32 exec_lo vcc_lo                           // s50 = s23 / s53
v_add_nc_u32 v6, v6, 1                             // s50 = s23 / s53
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s50, v6                        // quotient
s_mul_i32 s50, s50, s53
s_sub_u32 s51, s23, s50
s_cmp_gt_u32 s[sgprWorkGroup0], s50
s_cselect_b32 s50, s51, s53
s_lshr_b32 s50, s50, s52
s_bfm_b32 s51, s52, 0
s_and_b32 s51, s[sgprWorkGroup0], s51
s_mul_i32 s50, s50, s51
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s49, s50
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
/* init: add vgpr [24...65) to pool */
/* init: add vgpr [0...16) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
v_lshrrev_b32 v0, 3, v[vgprSerial]                 // 0 = Serial / 8
v_and_b32 v1, 7, v[vgprSerial]                     // 1 = Serial % 8
v_lshrrev_b32 v2, 3, v0                            // 2 = 0 / 8
v_and_b32 v3, 7, v0                                // 3 = 0 % 8

/* local read addresses: final offsets a */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(64)
v_lshlrev_b32 v0, 5, v0                            // LSU offset: lsuoffset = sgid*(MT0+PAD)
                                                   // Final Offset: lrAOffset * VW (multiplier is 1, do nothing)
v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x2  // Final Offset: add padding 0 per block 0

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(64)
v_mul_lo_u32 v0, 33, v0                            // LSU offset: lsuoffset = sgid*(MT1+PAD)
                                                   // Final Offset: lrBOffset * VW (multiplier is 1, do nothing)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x2  // Final Offset: add padding 4 per block 0

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x800, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 32 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 5, v[vgprSerial]                 // 1 = Serial / 32
v_and_b32 v0, 31, v[vgprSerial]                    // 0 = Serial % 32
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // 2 = Serial / 16
v_and_b32 v3, 15, v[vgprSerial]                    // 3 = Serial % 16
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x20, v4     // lwAL**(MTA + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x2 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x21, v5     // lwBL**(MTB + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x2 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpeDS
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x800, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=2048
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_nc_u32 v9, v7, v9                            // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc_lo, v9, 0                         // v6 = ceil(v7 / v8)
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo          // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_nc_u32 v9, v7, v9                            // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc_lo, v9, 0                         // v6 = ceil(v7 / v8)
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo          // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v6, s16                              // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v6, v6                             // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v6, v6, v7                               // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v6, v6                               // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v7, v6, s16                          // s16 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v7, s16                       // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo vcc_lo                           // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v6, 1, v6                             // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s16                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo vcc_lo                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v6, 1, v6                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 88
s_mul_i32 s54, s20, 4
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 196
s_mov_b32 s54, 0
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s55, 0
s_load_b128 s[24:27], s[48:49], s54
s_cmpk_eq_u32 s20, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 5                             // s52 = s24 / 32
s_and_b32 s50, 31, s24                             // s50 = s24 % 32
s_addc_u32 s52, s52, 0
s_lshr_b32 s53, s25, 5                             // s53 = s25 / 32
s_and_b32 s50, 31, s25                             // s50 = s25 % 32
s_addc_u32 s53, s53, 0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_and_b32 s53, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s52, s52, s53
s_add_u32 s55, s55, s52
s_cmp_lt_u32 s[sgprWorkGroup0], s55
s_cbranch_scc1 label_FOUND
s_add_u32 s54, s54, s15
s_load_b128 s[24:27], s[48:49], s54
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s20
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 5                             // s52 = s24 / 32
s_and_b32 s50, 31, s24                             // s50 = s24 % 32
s_addc_u32 s52, s52, 0
s_lshr_b32 s53, s25, 5                             // s53 = s25 / 32
s_and_b32 s50, 31, s25                             // s50 = s25 % 32
s_addc_u32 s53, s53, 0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_and_b32 s48, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s52, s52, s48
s_add_u32 s55, s55, s52

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s49, s14, 1
s_sub_u32 s48, s55, s52
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 88
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 196
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b32 s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
// Read Beta
s_load_b32 s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
label_LoadExternalStructEnd:
/* init: add vgpr [24...65) to pool */
/* init: add vgpr [0...16) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
v_lshrrev_b32 v0, 3, v[vgprSerial]                 // 0 = Serial / 8
v_and_b32 v1, 7, v[vgprSerial]                     // 1 = Serial % 8
v_lshrrev_b32 v2, 3, v0                            // 2 = 0 / 8
v_and_b32 v3, 7, v0                                // 3 = 0 % 8

/* local read addresses: final offsets a */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(64)
v_lshlrev_b32 v0, 5, v0                            // LSU offset: lsuoffset = sgid*(MT0+PAD)
                                                   // Final Offset: lrAOffset * VW (multiplier is 1, do nothing)
v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x2  // Final Offset: add padding 0 per block 0

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(64)
v_mul_lo_u32 v0, 33, v0                            // LSU offset: lsuoffset = sgid*(MT1+PAD)
                                                   // Final Offset: lrBOffset * VW (multiplier is 1, do nothing)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x2  // Final Offset: add padding 4 per block 0

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x800, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 32 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 5, v[vgprSerial]                 // 1 = Serial / 32
v_and_b32 v0, 31, v[vgprSerial]                    // 0 = Serial % 32
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // 2 = Serial / 16
v_and_b32 v3, 15, v[vgprSerial]                    // 3 = Serial % 16
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x20, v4     // lwAL**(MTA + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x2 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x21, v5     // lwBL**(MTB + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x2 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpeDS
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x800, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=2048
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_nc_u32 v9, v7, v9                            // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc_lo, v9, 0                         // v6 = ceil(v7 / v8)
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo          // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_nc_u32 v9, v7, v9                            // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc_lo, v9, 0                         // v6 = ceil(v7 / v8)
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo          // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1

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
v_cvt_f32_u32 v6, s16                              // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v6, v6                             // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v6, v6, v7                               // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v6, v6                               // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v7, v6, s16                          // s16 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v7, s16                       // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo vcc_lo                           // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v6, 1, v6                             // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s16                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo vcc_lo                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v6, 1, v6                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v6                        // quotient
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
.set sgprSrdA, 48
.set sgprSrdB, 52
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprStaggerUIter, 47
.set sgprWrapUA, 60
.set sgprWrapUB, 62
.set sgprGlobalReadIncsA, 64
.set sgprGlobalReadIncsB, 65
.set sgprScalarGlobalReadOffsetA, 66
.set sgprScalarGlobalReadOffsetB, 73
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 4  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 4  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:

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
v_cvt_f32_u32 v6, s16                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_u32_u24 v7, v6, s16                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cmp_eq_u32 vcc_lo, v7, s16                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_add_nc_u32 v6, 1, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s16                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s16                          // re-calculate remainder
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v7           // remainder
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
v_cvt_f32_u32 v6, s[sgprNumWorkGroups1]            // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v6, v6                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v6, v6, v7                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_u32_f32 v6, v6                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups1]        // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups1]     // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_nc_u32 v6, 1, v6                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups1]     // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups1]        // re-calculate remainder
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprGSUSumIdx], v6           // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v7          // remainder
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
v_cvt_f64_u32 v[6:7], s16                          // s17 = s[sgprWorkGroup0] / s16
v_rcp_f64 v[6:7], v[6:7]                           // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s17 = s[sgprWorkGroup0] / s16
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s17 = s[sgprWorkGroup0] / s16
v_cvt_u32_f64 v6, v[6:7]                           // s17 = s[sgprWorkGroup0] / s16
v_mul_lo_u32 v7, v6, s16                           // s17 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s17 = s[sgprWorkGroup0] / s16
v_cmp_ge_u32 vcc_lo, v8, s16                       // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo vcc_lo                           // s17 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v6, v6, 1                             // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v6                        // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup0], s20              // WorkGroup0=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s20, s20, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[6:7], s16                          // s18 = s[sgprNumWorkGroups0] / s16
v_rcp_f64 v[6:7], v[6:7]                           // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups0]        // s18 = s[sgprNumWorkGroups0] / s16
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_u32_f64 v6, v[6:7]                           // s18 = s[sgprNumWorkGroups0] / s16
v_mul_lo_u32 v7, v6, s16                           // s18 = s[sgprNumWorkGroups0] / s16
v_sub_nc_u32 v8, s[sgprNumWorkGroups0], v7         // s18 = s[sgprNumWorkGroups0] / s16
v_cmp_ge_u32 vcc_lo, v8, s16                       // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo vcc_lo                           // s18 = s[sgprNumWorkGroups0] / s16
v_add_nc_u32 v6, v6, 1                             // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v6                        // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups0], s19          // NumWorkGroups0=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[6:7], s18                          // s[sgprWorkGroup1] = s20 / s18
v_rcp_f64 v[6:7], v[6:7]                           // s[sgprWorkGroup1] = s20 / s18
v_cvt_f64_u32 v[8:9], s20                          // s[sgprWorkGroup1] = s20 / s18
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s[sgprWorkGroup1] = s20 / s18
v_cvt_u32_f64 v6, v[6:7]                           // s[sgprWorkGroup1] = s20 / s18
v_mul_lo_u32 v7, v6, s18                           // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v8, s20, v7                           // s[sgprWorkGroup1] = s20 / s18
v_cmp_ge_u32 vcc_lo, v8, s18                       // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s20 / s18
v_add_nc_u32 v6, v6, 1                             // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s18                           // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v8, s20, v7                           // s[sgprWorkGroup1] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v8          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s16, s16                                 // WGM
v_cvt_f64_u32 v[6:7], s16                          // s17 = s[sgprWorkGroup1] / s16
v_rcp_f64 v[6:7], v[6:7]                           // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup1]            // s17 = s[sgprWorkGroup1] / s16
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s17 = s[sgprWorkGroup1] / s16
v_cvt_u32_f64 v6, v[6:7]                           // s17 = s[sgprWorkGroup1] / s16
v_mul_lo_u32 v7, v6, s16                           // s17 = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v8, s[sgprWorkGroup1], v7             // s17 = s[sgprWorkGroup1] / s16
v_cmp_ge_u32 vcc_lo, v8, s16                       // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo vcc_lo                           // s17 = s[sgprWorkGroup1] / s16
v_add_nc_u32 v6, v6, 1                             // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v6                        // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup1], s20              // WorkGroup1=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s20, s20, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[6:7], s16                          // s18 = s[sgprNumWorkGroups1] / s16
v_rcp_f64 v[6:7], v[6:7]                           // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups1]        // s18 = s[sgprNumWorkGroups1] / s16
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_u32_f64 v6, v[6:7]                           // s18 = s[sgprNumWorkGroups1] / s16
v_mul_lo_u32 v7, v6, s16                           // s18 = s[sgprNumWorkGroups1] / s16
v_sub_nc_u32 v8, s[sgprNumWorkGroups1], v7         // s18 = s[sgprNumWorkGroups1] / s16
v_cmp_ge_u32 vcc_lo, v8, s16                       // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo vcc_lo                           // s18 = s[sgprNumWorkGroups1] / s16
v_add_nc_u32 v6, v6, 1                             // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v6                        // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups1], s19          // NumWorkGroups1=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[6:7], s18                          // s[sgprWorkGroup0] = s20 / s18
v_rcp_f64 v[6:7], v[6:7]                           // s[sgprWorkGroup0] = s20 / s18
v_cvt_f64_u32 v[8:9], s20                          // s[sgprWorkGroup0] = s20 / s18
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s[sgprWorkGroup0] = s20 / s18
v_cvt_u32_f64 v6, v[6:7]                           // s[sgprWorkGroup0] = s20 / s18
v_mul_lo_u32 v7, v6, s18                           // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v8, s20, v7                           // s[sgprWorkGroup0] = s20 / s18
v_cmp_ge_u32 vcc_lo, v8, s18                       // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup0] = s20 / s18
v_add_nc_u32 v6, v6, 1                             // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s18                           // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v8, s20, v7                           // s[sgprWorkGroup0] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup0], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v8          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s20, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s17 // wg1 += blockId * WGM
label_WGM:

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v0 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v2 */

/* global read addresses: unroll assignment a */
/* v1 */

/* global read addresses: unroll assignment b */
/* v3 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */

/* global read addresses: tile offsets b */

/* global read addresses: unroll offsets a */

/* global read addresses: unroll offsets b */

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup0], 32            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup0], 32               // WorkGroup[01] * MT
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s17, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 4 // s[LoopCounterL] = s[sgprSizesSum] / 16
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v6, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v6, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v7, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v6, v6, v7                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v6, v6                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v7, s[sgprLoopCounterL], v7           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v7, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v6, 1, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_nc_u32 v7, s[sgprLoopCounterL], v7           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v6        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v7         // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_mul_hi_u32 s17, s16, s[sgprStrideAL]             // tlu=1, scaled unroll-offset by stride
s_mul_i32 s16, s16, s[sgprStrideAL]                // tlu=1, scaled unroll-offset by stride
s_add_u32 s18, s18, s16                            // accum GsuOffset term to tilestart
s_addc_u32 s19, s19, s17                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s16, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s17, constStrideA0I, s16              // stride x (size-1)
s_mul_i32 s16, constStrideA0I, s16                 // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // sum tensor size
s_sub_u32 s16, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s17, s[sgprStrideAL], s16             // stride x (size-1)
s_mul_i32 s16, s[sgprStrideAL], s16                // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s18 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s19 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x2 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 4 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s17, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s19   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup1], 32            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 32               // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s17, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 4 // s[LoopCounterL] = s[sgprSizesSum] / 16
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v6, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v6, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v7, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v6, v6, v7                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v6, v6                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v7, s[sgprLoopCounterL], v7           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v7, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v6, 1, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_nc_u32 v7, s[sgprLoopCounterL], v7           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v6        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v7         // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x2 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 4 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s17, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 2                   // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s19   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  0,  1, 6 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideAL], 2 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+0], s[sgprScalarGlobalReadOffsetA+0], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+1], s[sgprStrideAL], 4 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+1], s[sgprScalarGlobalReadOffsetA+1], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+2], s[sgprStrideAL], 6 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+2], s[sgprScalarGlobalReadOffsetA+2], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+3], s[sgprStrideAL], 8 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+3], s[sgprScalarGlobalReadOffsetA+3], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+4], s[sgprStrideAL], 10 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+4], s[sgprScalarGlobalReadOffsetA+4], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+5], s[sgprStrideAL], 12 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+5], s[sgprScalarGlobalReadOffsetA+5], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+6], s[sgprStrideAL], 14 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+6], s[sgprScalarGlobalReadOffsetA+6], 0x2 // scalar offset *= bytes/element
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 6 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideB1J], 4 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+1], s[sgprStrideB1J], 8 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+1], s[sgprScalarGlobalReadOffsetB+1], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+2], s[sgprStrideB1J], 12 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+2], s[sgprScalarGlobalReadOffsetB+2], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+3], s[sgprStrideB1J], 16 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+3], s[sgprScalarGlobalReadOffsetB+3], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+4], s[sgprStrideB1J], 20 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+4], s[sgprScalarGlobalReadOffsetB+4], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+5], s[sgprStrideB1J], 24 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+5], s[sgprScalarGlobalReadOffsetB+5], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+6], s[sgprStrideB1J], 28 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+6], s[sgprScalarGlobalReadOffsetB+6], 0x2 // scalar offset *= bytes/element
/* ============================================================= */

/* global read addresses: increments a */
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s17, s17, DepthU*BpeAGR                  // GSU*DepthU*Bpe
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cmov_b32 s17, DepthU*BpeAGR                      // DepthU*Bpe if GSUC = 1
s_mul_i32 s[sgprGlobalReadIncsA+0], s17, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s17, s17, DepthU*BpeBGR                  // GSU*DepthU*Bpe
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeBGR, s17 // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v0, 1, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
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
buffer_load_b32 v[vgprG2LA+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b32 v[vgprG2LA+4], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b32 v[vgprG2LA+5], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b32 v[vgprG2LA+7], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 // G -> Reg 0_0_7_0
buffer_load_b32 v[vgprG2LB+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LB+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LB+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LB+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b32 v[vgprG2LB+4], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b32 v[vgprG2LB+5], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b32 v[vgprG2LB+6], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5] offen offset:0 // G -> Reg 0_0_6_0
buffer_load_b32 v[vgprG2LB+7], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6] offen offset:0 // G -> Reg 0_0_7_0

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
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s82, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s81, s82, s[sgprStrideC1J]            // ScaleC s82 by Stride
s_mul_i32 s80, s82, s[sgprStrideC1J]               // ScaleC s82 by Stride
s_lshl_b64 s[80:81], s[80:81], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s80    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s81   // add hi to SRD
s_mul_hi_u32 s81, s82, s[sgprStrideD1J]            // ScaleD s82 by Stride
s_mul_i32 s80, s82, s[sgprStrideD1J]               // ScaleD s82 by Stride
s_lshl_b64 s[80:81], s[80:81], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s80    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s81   // add hi to SRD

s_mul_hi_u32 s81, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s80, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[80:81], s[80:81], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s80        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s81       // add hi to SRD
s_mul_hi_u32 s81, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s80, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[80:81], s[80:81], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s80        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s81       // add hi to SRD

s_and_b32 s80, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s80, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s81, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s80, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s82, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s82, s82, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s83, s82, s[sgprStrideC1J]            // Free1
s_mul_i32 s82, s82, s[sgprStrideC1J]               // Free1
s_add_u32 s80, s80, s82                            // Free1
s_addc_u32 s81, s81, s83                           // Free1
s_sub_u32 s82, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s82, s82, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s83, s82, s[sgprStrideCK]             // Free2
s_mul_i32 s82, s82, s[sgprStrideCK]                // Free2
s_add_u32 s80, s80, s82                            // Free2
s_addc_u32 s81, s81, s83                           // Free2
s_lshl_b64 s[80:81], s[80:81], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s80        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s81       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* initC: remove ValuC vgpr buffer [0...16) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [24...41) from pool */
v_mov_b32 v[vgprValuC+0], 0                        // initC
v_mov_b32 v[vgprValuC+1], 0                        // initC
v_mov_b32 v[vgprValuC+2], 0                        // initC
v_mov_b32 v[vgprValuC+3], 0                        // initC
v_mov_b32 v[vgprValuC+4], 0                        // initC
v_mov_b32 v[vgprValuC+5], 0                        // initC
v_mov_b32 v[vgprValuC+6], 0                        // initC
v_mov_b32 v[vgprValuC+7], 0                        // initC
v_mov_b32 v[vgprValuC+8], 0                        // initC
v_mov_b32 v[vgprValuC+9], 0                        // initC
v_mov_b32 v[vgprValuC+10], 0                       // initC
v_mov_b32 v[vgprValuC+11], 0                       // initC
v_mov_b32 v[vgprValuC+12], 0                       // initC
v_mov_b32 v[vgprValuC+13], 0                       // initC
v_mov_b32 v[vgprValuC+14], 0                       // initC
v_mov_b32 v[vgprValuC+15], 0                       // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc1
s_getpc_b64 s[80:81]                               // addr of next instr
s_add_i32 s82, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s80, s80, s82                            // add target branch offset
s_addc_u32 s81, s81, 0                             // add high and carry
s_setpc_b64 s[80:81]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_T8JHFHKM7BO5OHXW:
s_waitcnt vmcnt(0)                                 // wait for global read

/* local write a */
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:256 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 256
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:512 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 512
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:768 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 768
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+4] offset:1024 // lwoA_0_0_4_0 = (0*LSCA) + (4*LSPA)(*MT0I+PAD) = 1024
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+5] offset:1280 // lwoA_0_0_5_0 = (0*LSCA) + (5*LSPA)(*MT0I+PAD) = 1280
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+6] offset:1536 // lwoA_0_0_6_0 = (0*LSCA) + (6*LSPA)(*MT0I+PAD) = 1536
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+7] offset:1792 // lwoA_0_0_7_0 = (0*LSCA) + (7*LSPA)(*MT0I+PAD) = 1792

/* local write b */
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:16 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 16
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:32 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 32
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:48 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 48
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:64 // lwoB_0_0_4_0 = (0 + 0*LSCB)*(MT1J+PAD) + (4*LSPB) = 64
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:80 // lwoB_0_0_5_0 = (0 + 0*LSCB)*(MT1J+PAD) + (5*LSPB) = 80
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:96 // lwoB_0_0_6_0 = (0 + 0*LSCB)*(MT1J+PAD) + (6*LSPB) = 96
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:112 // lwoB_0_0_7_0 = (0 + 0*LSCB)*(MT1J+PAD) + (7*LSPB) = 112

/* local write swap a */

/* (EPS=1) local write swap internal offset -> 8192 */

/* local write swap b */

/* (EPS=1) local write swap internal offset -> 8192 */
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local read prefetch a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:32
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:64
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:96

/* local read prefetch b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:0
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:32
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:64
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:96

/* local read inc a */
/* N/A, lro->32 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/* local read inc b */
/* N/A, lro->33 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_le_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:128
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:160
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:192
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:224
buffer_load_b32 v[vgprG2LA+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b32 v[vgprG2LA+4], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b32 v[vgprG2LA+5], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 // G -> Reg 0_0_6_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:132
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:164
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:196
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:228

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 2 self.localReadDoCntB 2 */

/* local read increment b */
/* N/A, lro->66 */
/* self.localReadDoCntA 2 self.localReadDoCntB 2 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0 for iteration == 0
MAC_4x4_X0


/* iter 1 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:256
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:288
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:320
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:352
buffer_load_b32 v[vgprG2LA+7], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 // G -> Reg 0_0_7_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:264
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:296
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:328
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:360

/* local read increment a */
/* N/A, lro->96 */
/* self.localReadDoCntA 3 self.localReadDoCntB 3 */

/* local read increment b */
/* N/A, lro->99 */
/* self.localReadDoCntA 3 self.localReadDoCntB 3 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 2 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:384
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:416
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:448
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:480
buffer_load_b32 v[vgprG2LB+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:396
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:428
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:460
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:492

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 4 self.localReadDoCntB 4 */

/* local read increment b */
/* N/A, lro->132 */
/* self.localReadDoCntA 4 self.localReadDoCntB 4 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 3 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:512
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:544
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:576
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:608
buffer_load_b32 v[vgprG2LB+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:528
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:560
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:592
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:624

/* local read increment a */
/* N/A, lro->160 */
/* self.localReadDoCntA 5 self.localReadDoCntB 5 */

/* local read increment b */
/* N/A, lro->165 */
/* self.localReadDoCntA 5 self.localReadDoCntB 5 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 4 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:640
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:672
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:704
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:736
buffer_load_b32 v[vgprG2LB+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0 // G -> Reg 0_0_2_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:660
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:692
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:724
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:756

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 6 self.localReadDoCntB 6 */

/* local read increment b */
/* N/A, lro->198 */
/* self.localReadDoCntA 6 self.localReadDoCntB 6 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 5 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:768
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:800
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:832
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:864
buffer_load_b32 v[vgprG2LB+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0 // G -> Reg 0_0_3_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:792
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:824
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:856
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:888

/* local read increment a */
/* N/A, lro->224 */
/* self.localReadDoCntA 7 self.localReadDoCntB 7 */

/* local read increment b */
/* N/A, lro->231 */
/* self.localReadDoCntA 7 self.localReadDoCntB 7 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 6 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:896
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:928
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:960
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:992
buffer_load_b32 v[vgprG2LB+4], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0 // G -> Reg 0_0_4_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:924
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:956
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:988
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1020

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 8 self.localReadDoCntB 8 */

/* local read increment b */
/* N/A, lro->264 */
/* self.localReadDoCntA 8 self.localReadDoCntB 8 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 7 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1024
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1056
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1088
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1120
buffer_load_b32 v[vgprG2LB+5], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0 // G -> Reg 0_0_5_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1056
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1088
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1120
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1152

/* local read increment a */
/* N/A, lro->288 */
/* self.localReadDoCntA 9 self.localReadDoCntB 9 */

/* local read increment b */
/* N/A, lro->297 */
/* self.localReadDoCntA 9 self.localReadDoCntB 9 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 8 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1152
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1184
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1216
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1248
buffer_load_b32 v[vgprG2LB+6], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5] offen offset:0 // G -> Reg 0_0_6_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1188
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1220
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1252
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1284

/* local read increment a */
/* N/A, lro->320 */
/* self.localReadDoCntA 10 self.localReadDoCntB 10 */

/* local read increment b */
/* N/A, lro->330 */
/* self.localReadDoCntA 10 self.localReadDoCntB 10 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 9 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1280
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1312
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1344
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1376
buffer_load_b32 v[vgprG2LB+7], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6] offen offset:0 // G -> Reg 0_0_7_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1320
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1352
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1384
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1416

/* local read increment a */
/* N/A, lro->352 */
/* self.localReadDoCntA 11 self.localReadDoCntB 11 */

/* local read increment b */
/* N/A, lro->363 */
/* self.localReadDoCntA 11 self.localReadDoCntB 11 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 10 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1408
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1440
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1472
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1504

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s80, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s81, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s80        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s81       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s80 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s81 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1452
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1484
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1516
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1548

/* local read increment a */
/* N/A, lro->384 */
/* self.localReadDoCntA 12 self.localReadDoCntB 12 */

/* local read increment b */
/* N/A, lro->396 */
/* self.localReadDoCntA 12 self.localReadDoCntB 12 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 11 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1536
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1568
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1600
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1632

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s80, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s81, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s80        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s81       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s80 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s81 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1584
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1616
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1648
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1680

/* local read increment a */
/* N/A, lro->416 */
/* self.localReadDoCntA 13 self.localReadDoCntB 13 */

/* local read increment b */
/* N/A, lro->429 */
/* self.localReadDoCntA 13 self.localReadDoCntB 13 */
/* sched write - iter 11 writesPerItem=1 */
s_waitcnt vmcnt(15)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:8192 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 8192
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X1


/* iter 12 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1664
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1696
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1728
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1760

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1716
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1748
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1780
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1812

/* local read increment a */
/* N/A, lro->448 */
/* self.localReadDoCntA 14 self.localReadDoCntB 14 */

/* local read increment b */
/* N/A, lro->462 */
/* self.localReadDoCntA 14 self.localReadDoCntB 14 */
/* sched write - iter 12 writesPerItem=1 */
s_waitcnt vmcnt(14)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:8448 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 8448
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X0


/* iter 13 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1792
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1824
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1856
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1888

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1848
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1880
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1912
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1944

/* local read increment a */
/* N/A, lro->480 */
/* self.localReadDoCntA 15 self.localReadDoCntB 15 */

/* local read increment b */
/* N/A, lro->495 */
/* self.localReadDoCntA 15 self.localReadDoCntB 15 */
/* sched write - iter 13 writesPerItem=1 */
s_waitcnt vmcnt(13)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:8704 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8704
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X1


/* iter 14 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1920
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1952
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1984
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:2016

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1980
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:2012
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:2044
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:2076
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:8960 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 8960
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+4] offset:9216 // lwoA_0_0_4_0 = (0*LSCA) + (4*LSPA)(*MT0I+PAD) = 9216
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+5] offset:9472 // lwoA_0_0_5_0 = (0*LSCA) + (5*LSPA)(*MT0I+PAD) = 9472
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+6] offset:9728 // lwoA_0_0_6_0 = (0*LSCA) + (6*LSPA)(*MT0I+PAD) = 9728
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+7] offset:9984 // lwoA_0_0_7_0 = (0*LSCA) + (7*LSPA)(*MT0I+PAD) = 9984
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:8192 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 8192
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:8208 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 8208
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:8224 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 8224
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:8240 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 8240
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:8256 // lwoB_0_0_4_0 = (0 + 0*LSCB)*(MT1J+PAD) + (4*LSPB) = 8256
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:8272 // lwoB_0_0_5_0 = (0 + 0*LSCB)*(MT1J+PAD) + (5*LSPB) = 8272
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:8288 // lwoB_0_0_6_0 = (0 + 0*LSCB)*(MT1J+PAD) + (6*LSPB) = 8288
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:8304 // lwoB_0_0_7_0 = (0 + 0*LSCB)*(MT1J+PAD) + (7*LSPB) = 8304

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 0 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 0 */

/* local read swap offsets a */

/* local read swap internal offset -> 8192 */

/* local read swap offsets b */

/* local read swap internal offset -> 8192 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(21)                              // wait for prior local read local write old=0, new=21 newLW=0 newLR=0
MAC_4x4_X0


/* iter 15 */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:8192
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:8224
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:8256
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:8288

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:8192
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:8224
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:8256
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:8288

/* local read increment a */
/* N/A, lro->32 */
/* self.localReadDoCntA 17 self.localReadDoCntB 17 */

/* local read increment b */
/* N/A, lro->33 */
/* self.localReadDoCntA 17 self.localReadDoCntB 17 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/

/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc1 label_LoopEndL_oddexit              // exit LoopL

/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:8320
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:8352
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:8384
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:8416
buffer_load_b32 v[vgprG2LA+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b32 v[vgprG2LA+4], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b32 v[vgprG2LA+5], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 // G -> Reg 0_0_5_0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 // G -> Reg 0_0_6_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:8324
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:8356
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:8388
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:8420

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 18 self.localReadDoCntB 18 */

/* local read increment b */
/* N/A, lro->66 */
/* self.localReadDoCntA 18 self.localReadDoCntB 18 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0 for iteration == 0
MAC_4x4_X0


/* iter 1 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:8448
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:8480
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:8512
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:8544
buffer_load_b32 v[vgprG2LA+7], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 // G -> Reg 0_0_7_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:8456
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:8488
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:8520
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:8552

/* local read increment a */
/* N/A, lro->96 */
/* self.localReadDoCntA 19 self.localReadDoCntB 19 */

/* local read increment b */
/* N/A, lro->99 */
/* self.localReadDoCntA 19 self.localReadDoCntB 19 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 2 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:8576
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:8608
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:8640
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:8672
buffer_load_b32 v[vgprG2LB+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:8588
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:8620
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:8652
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:8684

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 20 self.localReadDoCntB 20 */

/* local read increment b */
/* N/A, lro->132 */
/* self.localReadDoCntA 20 self.localReadDoCntB 20 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 3 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:8704
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:8736
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:8768
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:8800
buffer_load_b32 v[vgprG2LB+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:8720
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:8752
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:8784
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:8816

/* local read increment a */
/* N/A, lro->160 */
/* self.localReadDoCntA 21 self.localReadDoCntB 21 */

/* local read increment b */
/* N/A, lro->165 */
/* self.localReadDoCntA 21 self.localReadDoCntB 21 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 4 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:8832
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:8864
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:8896
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:8928
buffer_load_b32 v[vgprG2LB+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0 // G -> Reg 0_0_2_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:8852
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:8884
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:8916
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:8948

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 22 self.localReadDoCntB 22 */

/* local read increment b */
/* N/A, lro->198 */
/* self.localReadDoCntA 22 self.localReadDoCntB 22 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 5 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:8960
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:8992
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:9024
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:9056
buffer_load_b32 v[vgprG2LB+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0 // G -> Reg 0_0_3_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:8984
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:9016
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:9048
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:9080

/* local read increment a */
/* N/A, lro->224 */
/* self.localReadDoCntA 23 self.localReadDoCntB 23 */

/* local read increment b */
/* N/A, lro->231 */
/* self.localReadDoCntA 23 self.localReadDoCntB 23 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 6 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:9088
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:9120
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:9152
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:9184
buffer_load_b32 v[vgprG2LB+4], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0 // G -> Reg 0_0_4_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:9116
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:9148
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:9180
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:9212

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 24 self.localReadDoCntB 24 */

/* local read increment b */
/* N/A, lro->264 */
/* self.localReadDoCntA 24 self.localReadDoCntB 24 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 7 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:9216
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:9248
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:9280
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:9312
buffer_load_b32 v[vgprG2LB+5], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0 // G -> Reg 0_0_5_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:9248
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:9280
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:9312
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:9344

/* local read increment a */
/* N/A, lro->288 */
/* self.localReadDoCntA 25 self.localReadDoCntB 25 */

/* local read increment b */
/* N/A, lro->297 */
/* self.localReadDoCntA 25 self.localReadDoCntB 25 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 8 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:9344
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:9376
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:9408
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:9440
buffer_load_b32 v[vgprG2LB+6], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5] offen offset:0 // G -> Reg 0_0_6_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:9380
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:9412
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:9444
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:9476

/* local read increment a */
/* N/A, lro->320 */
/* self.localReadDoCntA 26 self.localReadDoCntB 26 */

/* local read increment b */
/* N/A, lro->330 */
/* self.localReadDoCntA 26 self.localReadDoCntB 26 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 9 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:9472
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:9504
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:9536
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:9568
buffer_load_b32 v[vgprG2LB+7], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6] offen offset:0 // G -> Reg 0_0_7_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:9512
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:9544
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:9576
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:9608

/* local read increment a */
/* N/A, lro->352 */
/* self.localReadDoCntA 27 self.localReadDoCntB 27 */

/* local read increment b */
/* N/A, lro->363 */
/* self.localReadDoCntA 27 self.localReadDoCntB 27 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 10 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:9600
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:9632
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:9664
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:9696

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s80, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s81, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s80        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s81       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s80 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s81 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:9644
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:9676
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:9708
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:9740

/* local read increment a */
/* N/A, lro->384 */
/* self.localReadDoCntA 28 self.localReadDoCntB 28 */

/* local read increment b */
/* N/A, lro->396 */
/* self.localReadDoCntA 28 self.localReadDoCntB 28 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 11 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:9728
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:9760
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:9792
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:9824

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s80, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s81, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s80        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s81       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s80 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s81 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:9776
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:9808
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:9840
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:9872

/* local read increment a */
/* N/A, lro->416 */
/* self.localReadDoCntA 29 self.localReadDoCntB 29 */

/* local read increment b */
/* N/A, lro->429 */
/* self.localReadDoCntA 29 self.localReadDoCntB 29 */
/* sched write - iter 11 writesPerItem=1 */
s_waitcnt vmcnt(15)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X1


/* iter 12 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:9856
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:9888
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:9920
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:9952

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:9908
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:9940
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:9972
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:10004

/* local read increment a */
/* N/A, lro->448 */
/* self.localReadDoCntA 30 self.localReadDoCntB 30 */

/* local read increment b */
/* N/A, lro->462 */
/* self.localReadDoCntA 30 self.localReadDoCntB 30 */
/* sched write - iter 12 writesPerItem=1 */
s_waitcnt vmcnt(14)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:256 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 256
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X0


/* iter 13 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:9984
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:10016
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:10048
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:10080

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:10040
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:10072
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:10104
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:10136

/* local read increment a */
/* N/A, lro->480 */
/* self.localReadDoCntA 31 self.localReadDoCntB 31 */

/* local read increment b */
/* N/A, lro->495 */
/* self.localReadDoCntA 31 self.localReadDoCntB 31 */
/* sched write - iter 13 writesPerItem=1 */
s_waitcnt vmcnt(13)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:512 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 512
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=0 newLR=0
MAC_4x4_X1


/* iter 14 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:10112
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:10144
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:10176
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:10208

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:10172
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:10204
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:10236
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:10268
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(12)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:768 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 768
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(11)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+4] offset:1024 // lwoA_0_0_4_0 = (0*LSCA) + (4*LSPA)(*MT0I+PAD) = 1024
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(10)                                // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+5] offset:1280 // lwoA_0_0_5_0 = (0*LSCA) + (5*LSPA)(*MT0I+PAD) = 1280
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(9)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+6] offset:1536 // lwoA_0_0_6_0 = (0*LSCA) + (6*LSPA)(*MT0I+PAD) = 1536
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(8)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+7] offset:1792 // lwoA_0_0_7_0 = (0*LSCA) + (7*LSPA)(*MT0I+PAD) = 1792
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:16 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 16
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:32 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 32
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:48 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 48
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:64 // lwoB_0_0_4_0 = (0 + 0*LSCB)*(MT1J+PAD) + (4*LSPB) = 64
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:80 // lwoB_0_0_5_0 = (0 + 0*LSCB)*(MT1J+PAD) + (5*LSPB) = 80
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:96 // lwoB_0_0_6_0 = (0 + 0*LSCB)*(MT1J+PAD) + (6*LSPB) = 96
/* sched write - iter 14 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:112 // lwoB_0_0_7_0 = (0 + 0*LSCB)*(MT1J+PAD) + (7*LSPB) = 112

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 8192 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 8192 */

/* local read swap offsets a */

/* local read swap internal offset -> 0 */

/* local read swap offsets b */

/* local read swap internal offset -> 0 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(21)                              // wait for prior local read local write old=0, new=21 newLW=0 newLR=0
MAC_4x4_X0


/* iter 15 */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:32
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:64
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:96

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:0
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:32
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:64
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:96

/* local read increment a */
/* N/A, lro->32 */
/* self.localReadDoCntA 33 self.localReadDoCntB 33 */

/* local read increment b */
/* N/A, lro->33 */
/* self.localReadDoCntA 33 self.localReadDoCntB 33 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1              // counterL==1
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL_evenexit:  /// unroll loop eveniter exit
s_branch label_LoopEndL                            // exit unroll loopL (and skip second exit code)
label_LoopEndL_oddexit:  /// unroll loop odditer exit

/* Select high bank of LDS */
v_xor_b32 v[vgprLocalReadAddrA], 0x2000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x2000, v[vgprLocalReadAddrB] // swap Red Blk
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */
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

s_and_b32 s80, 31, s[sgprSizeI]                    // s80 = s[sgprSizeI] % 32
s_add_u32 s81, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s81                // wg0 >= nwg0-1 ?
s_cselect_b32 s80, s80, 0                          // set rem
s_cmpk_gt_u32 s80, 0                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s80, 31, s[sgprSizeJ]                    // s80 = s[sgprSizeJ] % 32
s_add_u32 s81, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s81                // wg1 >= nwg1-1
s_cselect_b32 s80, s80, 0                          // set rem
s_cmpk_gt_u32 s80, 0                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required

s_and_b32 s81, 15, s[sgprSizesSum+0]               // s81 = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s81, 0                                // numIterL == 0
s_cbranch_scc0 label_OptNLL_End                    // skip if tail loop required

/* iter 0 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:128
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:160
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:192
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:224

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:132
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:164
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:196
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:228

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 34 self.localReadDoCntB 34 */

/* local read increment b */
/* N/A, lro->66 */
/* self.localReadDoCntA 34 self.localReadDoCntB 34 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0 for iteration == 0
MAC_4x4_X0


/* iter 1 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:256
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:288
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:320
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:352

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:264
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:296
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:328
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:360

/* local read increment a */
/* N/A, lro->96 */
/* self.localReadDoCntA 35 self.localReadDoCntB 35 */

/* local read increment b */
/* N/A, lro->99 */
/* self.localReadDoCntA 35 self.localReadDoCntB 35 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 2 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:384
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:416
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:448
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:480

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:396
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:428
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:460
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:492

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 36 self.localReadDoCntB 36 */

/* local read increment b */
/* N/A, lro->132 */
/* self.localReadDoCntA 36 self.localReadDoCntB 36 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 3 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:512
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:544
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:576
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:608

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:528
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:560
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:592
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:624

/* local read increment a */
/* N/A, lro->160 */
/* self.localReadDoCntA 37 self.localReadDoCntB 37 */

/* local read increment b */
/* N/A, lro->165 */
/* self.localReadDoCntA 37 self.localReadDoCntB 37 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 4 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:640
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:672
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:704
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:736

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:660
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:692
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:724
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:756

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 38 self.localReadDoCntB 38 */

/* local read increment b */
/* N/A, lro->198 */
/* self.localReadDoCntA 38 self.localReadDoCntB 38 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 5 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:768
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:800
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:832
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:864

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:792
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:824
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:856
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:888

/* local read increment a */
/* N/A, lro->224 */
/* self.localReadDoCntA 39 self.localReadDoCntB 39 */

/* local read increment b */
/* N/A, lro->231 */
/* self.localReadDoCntA 39 self.localReadDoCntB 39 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 6 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:896
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:928
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:960
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:992

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:924
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:956
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:988
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1020

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 40 self.localReadDoCntB 40 */

/* local read increment b */
/* N/A, lro->264 */
/* self.localReadDoCntA 40 self.localReadDoCntB 40 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 7 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1024
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1056
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1088
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1120

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1056
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1088
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1120
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1152

/* local read increment a */
/* N/A, lro->288 */
/* self.localReadDoCntA 41 self.localReadDoCntB 41 */

/* local read increment b */
/* N/A, lro->297 */
/* self.localReadDoCntA 41 self.localReadDoCntB 41 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 8 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1152
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1184
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1216
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1248

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1188
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1220
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1252
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1284

/* local read increment a */
/* N/A, lro->320 */
/* self.localReadDoCntA 42 self.localReadDoCntB 42 */

/* local read increment b */
/* N/A, lro->330 */
/* self.localReadDoCntA 42 self.localReadDoCntB 42 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 9 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1280
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1312
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1344
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1376

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1320
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1352
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1384
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1416

/* local read increment a */
/* N/A, lro->352 */
/* self.localReadDoCntA 43 self.localReadDoCntB 43 */

/* local read increment b */
/* N/A, lro->363 */
/* self.localReadDoCntA 43 self.localReadDoCntB 43 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 10 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1408
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1440
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1472
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1504

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1452
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1484
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1516
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1548

/* local read increment a */
/* N/A, lro->384 */
/* self.localReadDoCntA 44 self.localReadDoCntB 44 */

/* local read increment b */
/* N/A, lro->396 */
/* self.localReadDoCntA 44 self.localReadDoCntB 44 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 11 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1536
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1568
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1600
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1632

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1584
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1616
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1648
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1680

/* local read increment a */
/* N/A, lro->416 */
/* self.localReadDoCntA 45 self.localReadDoCntB 45 */

/* local read increment b */
/* N/A, lro->429 */
/* self.localReadDoCntA 45 self.localReadDoCntB 45 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 12 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1664
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1696
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1728
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1760

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1716
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1748
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1780
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1812

/* local read increment a */
/* N/A, lro->448 */
/* self.localReadDoCntA 46 self.localReadDoCntB 46 */

/* local read increment b */
/* N/A, lro->462 */
/* self.localReadDoCntA 46 self.localReadDoCntB 46 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 13 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1792
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1824
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1856
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1888

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1848
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1880
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1912
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1944

/* local read increment a */
/* N/A, lro->480 */
/* self.localReadDoCntA 47 self.localReadDoCntB 47 */

/* local read increment b */
/* N/A, lro->495 */
/* self.localReadDoCntA 47 self.localReadDoCntB 47 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 14 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1920
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1952
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1984
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:2016

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1980
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:2012
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:2044
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:2076
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 15 (last unrolled loop) */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->512 */
/* self.localReadDoCntA 49 self.localReadDoCntB 49 */

/* local read increment b */
/* N/A, lro->528 */
/* self.localReadDoCntA 49 self.localReadDoCntB 49 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_4x4_X1

label_toPGR1end_OptNLL:
/* Stores for OptNLL */
label_Summation_End_OptNLL:
/* endSummation: add vgpr [16...58) to pool */
/* load store sgprs */
/* computeStoreVgprs */
v_lshrrev_b32 v17, 3, v[vgprSerial]                // 17 = Serial / 8
v_and_b32 v16, 7, v[vgprSerial]                    // 16 = Serial % 8
                                                   // v16 = v16 * 1 (multiplier is 1, do nothing)
v_mul_lo_u32 v18, v17, s[sgprStrideC1J]            // rowStart vgpr
v_mul_lo_u32 v19, v17, s[sgprStrideD1J]            // rowStart vgpr

s_mul_i32 s30, 0x20, s[sgprWorkGroup0]             // s30 = wg0*MT0
v_add_nc_u32 v16, s30, v16                         // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s32, 0x20, s[sgprWorkGroup1]             // <- wg1*MT1
v_add_nc_u32 v17, s32, v17                         // coord1 = tid1*VW + wg1*MT1

/******************************************/
/* Global Write Elements                  */
/******************************************/
label_GW_B0:
label_GW_B0_FD0:
label_GW_B0_FD0_VW1_Then:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
v_add_lshl_u32 v23, v19, v16, 0x2                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=16, coord0Vgpr=16

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v2, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v3, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v4, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v5, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v6, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v7, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v8, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v9, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v10, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v11, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v12, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v13, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v14, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v15, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_branch label_GW_End                              // jump to end
label_GW_End:

s_endpgm                                           // Kernel End
label_OptNLL_End:
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:128
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:160
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:192
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:224

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:132
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:164
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:196
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:228

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 34 self.localReadDoCntB 34 */

/* local read increment b */
/* N/A, lro->66 */
/* self.localReadDoCntA 34 self.localReadDoCntB 34 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0 for iteration == 0
MAC_4x4_X0


/* iter 1 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:256
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:288
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:320
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:352

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:264
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:296
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:328
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:360

/* local read increment a */
/* N/A, lro->96 */
/* self.localReadDoCntA 35 self.localReadDoCntB 35 */

/* local read increment b */
/* N/A, lro->99 */
/* self.localReadDoCntA 35 self.localReadDoCntB 35 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 2 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:384
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:416
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:448
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:480

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:396
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:428
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:460
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:492

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 36 self.localReadDoCntB 36 */

/* local read increment b */
/* N/A, lro->132 */
/* self.localReadDoCntA 36 self.localReadDoCntB 36 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 3 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:512
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:544
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:576
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:608

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:528
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:560
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:592
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:624

/* local read increment a */
/* N/A, lro->160 */
/* self.localReadDoCntA 37 self.localReadDoCntB 37 */

/* local read increment b */
/* N/A, lro->165 */
/* self.localReadDoCntA 37 self.localReadDoCntB 37 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 4 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:640
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:672
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:704
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:736

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:660
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:692
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:724
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:756

/* local read increment a */
/* N/A, lro->192 */
/* self.localReadDoCntA 38 self.localReadDoCntB 38 */

/* local read increment b */
/* N/A, lro->198 */
/* self.localReadDoCntA 38 self.localReadDoCntB 38 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 5 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:768
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:800
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:832
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:864

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:792
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:824
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:856
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:888

/* local read increment a */
/* N/A, lro->224 */
/* self.localReadDoCntA 39 self.localReadDoCntB 39 */

/* local read increment b */
/* N/A, lro->231 */
/* self.localReadDoCntA 39 self.localReadDoCntB 39 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 6 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:896
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:928
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:960
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:992

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:924
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:956
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:988
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1020

/* local read increment a */
/* N/A, lro->256 */
/* self.localReadDoCntA 40 self.localReadDoCntB 40 */

/* local read increment b */
/* N/A, lro->264 */
/* self.localReadDoCntA 40 self.localReadDoCntB 40 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 7 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1024
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1056
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1088
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1120

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1056
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1088
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1120
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1152

/* local read increment a */
/* N/A, lro->288 */
/* self.localReadDoCntA 41 self.localReadDoCntB 41 */

/* local read increment b */
/* N/A, lro->297 */
/* self.localReadDoCntA 41 self.localReadDoCntB 41 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 8 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1152
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1184
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1216
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1248

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1188
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1220
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1252
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1284

/* local read increment a */
/* N/A, lro->320 */
/* self.localReadDoCntA 42 self.localReadDoCntB 42 */

/* local read increment b */
/* N/A, lro->330 */
/* self.localReadDoCntA 42 self.localReadDoCntB 42 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 9 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1280
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1312
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1344
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1376

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1320
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1352
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1384
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1416

/* local read increment a */
/* N/A, lro->352 */
/* self.localReadDoCntA 43 self.localReadDoCntB 43 */

/* local read increment b */
/* N/A, lro->363 */
/* self.localReadDoCntA 43 self.localReadDoCntB 43 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 10 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1408
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1440
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1472
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1504

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1452
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1484
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1516
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1548

/* local read increment a */
/* N/A, lro->384 */
/* self.localReadDoCntA 44 self.localReadDoCntB 44 */

/* local read increment b */
/* N/A, lro->396 */
/* self.localReadDoCntA 44 self.localReadDoCntB 44 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 11 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1536
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1568
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1600
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1632

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1584
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1616
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1648
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1680

/* local read increment a */
/* N/A, lro->416 */
/* self.localReadDoCntA 45 self.localReadDoCntB 45 */

/* local read increment b */
/* N/A, lro->429 */
/* self.localReadDoCntA 45 self.localReadDoCntB 45 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 12 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1664
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1696
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1728
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:1760

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1716
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:1748
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:1780
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:1812

/* local read increment a */
/* N/A, lro->448 */
/* self.localReadDoCntA 46 self.localReadDoCntB 46 */

/* local read increment b */
/* N/A, lro->462 */
/* self.localReadDoCntA 46 self.localReadDoCntB 46 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 13 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:1792
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:1824
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:1856
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:1888

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:1848
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:1880
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:1912
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:1944

/* local read increment a */
/* N/A, lro->480 */
/* self.localReadDoCntA 47 self.localReadDoCntB 47 */

/* local read increment b */
/* N/A, lro->495 */
/* self.localReadDoCntA 47 self.localReadDoCntB 47 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X1


/* iter 14 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:1920
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:1952
ds_load_b32 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:1984
ds_load_b32 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:2016

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:1980
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:2012
ds_load_b32 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB+0] offset:2044
ds_load_b32 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB+0] offset:2076
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=0
MAC_4x4_X0


/* iter 15 (last unrolled loop) */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->512 */
/* self.localReadDoCntA 49 self.localReadDoCntB 49 */

/* local read increment b */
/* N/A, lro->528 */
/* self.localReadDoCntA 49 self.localReadDoCntB 49 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_4x4_X1

label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [24...41) to pool */

/* Tail: add address/G2L vgpr [41...58) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */
v_and_b32 v[vgprLocalWriteAddrA], 0xf01fff, v[vgprLocalWriteAddrA] // reset to Red

/* local write reset offsets b */
v_and_b32 v[vgprLocalWriteAddrB], 0xf01fff, v[vgprLocalWriteAddrB] // reset to Red
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (8,8,0) */
.set vgprG2LA_BASE, 24
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 32
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_and_b32 s80, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s81, s[sgprSizesSum], 4                 // s81 = s[sgprSizesSum] / 16
s_and_b32 s82, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v40, s82                             // s80 = s81 / s82
v_rcp_iflag_f32 v40, v40                           // s80 = s81 / s82
v_cvt_f32_u32 v41, s81                             // s80 = s81 / s82
v_mul_f32 v40, v40, v41                            // s80 = s81 / s82
v_cvt_u32_f32 v40, v40                             // s80 = s81 / s82
v_mul_u32_u24 v41, v40, s82                        // s80 = s81 / s82
v_sub_nc_u32 v41, s81, v41                         // s80 = s81 / s82
v_cmp_eq_u32 vcc_lo, v41, s82                      // s80 = s81 / s82
s_mov_b32 exec_lo vcc_lo                           // s80 = s81 / s82
v_add_nc_u32 v40, 1, v40                           // s80 = s81 / s82
v_mov_b32 v41, 0                                   // s[sgprGSUSumIdx+1] = s81 % s82
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v41, s82                      // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v40, v40, 1                           // quotient - 1
v_mul_u32_u24 v41, v40, s82                        // re-calculate remainder
v_sub_nc_u32 v41, s81, v41                         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s80, v40                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v41        // remainder
s_sub_u32 s81, s82, 1                              // GSU-1
s_cmp_eq_u32 s80, 0                                // quotient == 0
s_cselect_b32 s80, s[sgprGSUSumIdx+1], s81         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s80                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s80, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s80, 0
s_cbranch_scc0 label_Negative_S4FDBQ587JJL6NOU
s_mul_hi_u32 s81, s80, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s80, s80, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_L43KTIIJOOEN7J6P
label_Negative_S4FDBQ587JJL6NOU:
s_abs_i32 s80, s80
s_mul_hi_u32 s81, s80, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s80, s80, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s80, s80, 0xffffffff
s_xor_b32 s81, s81, 0xffffffff
s_add_u32 s80, s80, 0x1
s_addc_u32 s81, s81, 0
label_MultiplyDone_L43KTIIJOOEN7J6P:
s_sub_u32 s80, s80, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s81, s81, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s80        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s81       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s80 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s81 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s80, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s80, 0
s_cbranch_scc0 label_Negative_UR8VN3A1SJCPC6PO
s_mul_hi_u32 s81, s80, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s80, s80, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_HYY06MPL0TYYIAT2
label_Negative_UR8VN3A1SJCPC6PO:
s_abs_i32 s80, s80
s_mul_hi_u32 s81, s80, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s80, s80, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s80, s80, 0xffffffff
s_xor_b32 s81, s81, 0xffffffff
s_add_u32 s80, s80, 0x1
s_addc_u32 s81, s81, 0
label_MultiplyDone_HYY06MPL0TYYIAT2:
s_sub_u32 s80, s80, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s81, s81, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s80        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s81       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s80 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s81 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* Tail global read A */
/* g2l=0, load component 0 */
buffer_load_b32 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=1, load component 0 */
buffer_load_b32 v[vgprG2LA+1+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // load one buffer value
/* g2l=2, load component 0 */
buffer_load_b32 v[vgprG2LA+2+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // load one buffer value
/* g2l=3, load component 0 */
buffer_load_b32 v[vgprG2LA+3+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_b32 v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0 // load one buffer value
/* g2l=5, load component 0 */
buffer_load_b32 v[vgprG2LA+5+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0 // load one buffer value
/* g2l=6, load component 0 */
buffer_load_b32 v[vgprG2LA+6+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0 // load one buffer value
/* g2l=7, load component 0 */
buffer_load_b32 v[vgprG2LA+7+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0 // load one buffer value

/* Update M0 for DTLDS */

/* Tail global read B */
/* g2l=0, load component 0 */
buffer_load_b32 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=1, load component 0 */
buffer_load_b32 v[vgprG2LB+1+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // load one buffer value
/* g2l=2, load component 0 */
buffer_load_b32 v[vgprG2LB+2+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0 // load one buffer value
/* g2l=3, load component 0 */
buffer_load_b32 v[vgprG2LB+3+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_b32 v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0 // load one buffer value
/* g2l=5, load component 0 */
buffer_load_b32 v[vgprG2LB+5+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0 // load one buffer value
/* g2l=6, load component 0 */
buffer_load_b32 v[vgprG2LB+6+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5] offen offset:0 // load one buffer value
/* g2l=7, load component 0 */
buffer_load_b32 v[vgprG2LB+7+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6] offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)                                 // 2wait for global read
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local write a */
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:256 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 256
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:512 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 512
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:768 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 768
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+4] offset:1024 // lwoA_0_0_4_0 = (0*LSCA) + (4*LSPA)(*MT0I+PAD) = 1024
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+5] offset:1280 // lwoA_0_0_5_0 = (0*LSCA) + (5*LSPA)(*MT0I+PAD) = 1280
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+6] offset:1536 // lwoA_0_0_6_0 = (0*LSCA) + (6*LSPA)(*MT0I+PAD) = 1536
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+7] offset:1792 // lwoA_0_0_7_0 = (0*LSCA) + (7*LSPA)(*MT0I+PAD) = 1792

/* local write b */
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:16 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 16
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:32 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 32
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:48 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 48
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:64 // lwoB_0_0_4_0 = (0 + 0*LSCB)*(MT1J+PAD) + (4*LSPB) = 64
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:80 // lwoB_0_0_5_0 = (0 + 0*LSCB)*(MT1J+PAD) + (5*LSPB) = 80
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:96 // lwoB_0_0_6_0 = (0 + 0*LSCB)*(MT1J+PAD) + (6*LSPB) = 96
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:112 // lwoB_0_0_7_0 = (0 + 0*LSCB)*(MT1J+PAD) + (7*LSPB) = 112

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 24
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+4
.set vgprValuB_X0_I0_BASE, 32
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+4

/* Tail: local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA+0], 0x1fff, v[vgprLocalReadAddrA+0] // reset Red,Blk -> Red

/* Tail: local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB+0], 0x1fff, v[vgprLocalReadAddrB+0] // reset Red,Blk -> Red

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:32
ds_load_b32 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:64
ds_load_b32 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:96

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:0
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:32
ds_load_b32 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB+0] offset:64
ds_load_b32 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB+0] offset:96

/* local read inc a */
s_mov_b32 s8, 128                                  // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc_lo, s8, v[vgprLocalReadAddrA+0] // lrA += 128 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s8, 132                                  // inc
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, s8, v[vgprLocalReadAddrB+0] // lrB += 132 ((MT+PAD)*bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
MAC_4x4_X0


/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_X1_I0, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_X1_I0, UNDEF

/* Tail: add MISC Vgpr [16...24) to pool */
label_Summation_End_8S4L1KCK9VFC7AQU:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprScalarGlobalReadOffsetA, UNDEF
.set sgprScalarGlobalReadOffsetB, UNDEF
/* load store sgprs */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v17, 3, v[vgprSerial]                // 17 = Serial / 8
v_and_b32 v16, 7, v[vgprSerial]                    // 16 = Serial % 8
                                                   // v16 = v16 * 1 (multiplier is 1, do nothing)
v_mul_lo_u32 v18, v17, s[sgprStrideC1J]            // rowStart vgpr
v_mul_lo_u32 v19, v17, s[sgprStrideD1J]            // rowStart vgpr

s_mul_i32 s30, 0x20, s[sgprWorkGroup0]             // s30 = wg0*MT0
v_add_nc_u32 v16, s30, v16                         // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s32, 0x20, s[sgprWorkGroup1]             // <- wg1*MT1
v_add_nc_u32 v17, s32, v17                         // coord1 = tid1*VW + wg1*MT1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
label_GW_B0_1:
label_GW_B0_FD0_1:
s_and_b32 s30, 31, s[sgprSizeI]                    // s30 = s[sgprSizeI] % 32
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Else_1          // jump if edges required
s_and_b32 s30, 31, s[sgprSizeJ]                    // s30 = s[sgprSizeJ] % 32
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Then_1          // jump if edges required
label_GW_B0_FD0_VW1_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
v_add_lshl_u32 v23, v19, v16, 0x2                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=16, coord0Vgpr=16

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v2, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v3, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v4, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v5, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v6, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v7, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v8, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v9, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v10, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v11, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v12, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v13, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v14, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v15, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_NonEdgeEnd_1:
label_GW_B0_FD0_VW1_Else_1:
label_GW_B0_FD0_VW1_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=72 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v22, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v23, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v23, v22, v23, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v24, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v24, v22, v24, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v25, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v25, v22, v25, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v26, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v22, v26, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v27, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v22, v27, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v28, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v22, v28, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v29, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v29, v22, v29, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v30, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v22, v30, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v31, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v22, v31, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v32, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v22, v32, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v33, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v33, v22, v33, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v34, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v22, v34, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v35, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v35, v22, v35, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v36, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v22, v36, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v37, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, v22, v37, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v38, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v22, v38, s32                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v24, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v2, v25, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v3, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v4, v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v5, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v6, v29, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v7, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v8, v31, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v9, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v10, v33, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v11, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v12, v35, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v13, v36, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v14, v37, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v15, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
s_getpc_b64 s[30:31]                               // addr of next instr
s_add_i32 s32, label_KernelEnd, 4                  // target branch offset
s_add_u32 s30, s30, s32                            // add target branch offset
s_addc_u32 s31, s31, 0                             // add high and carry
s_setpc_b64 s[30:31]                               // branch to label_KernelEnd
label_GSU_4:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1                         // Branch if Beta is not zero

label_GW_B0_2:
label_GW_B0_FD0_2:
s_and_b32 s30, 31, s[sgprSizeI]                    // s30 = s[sgprSizeI] % 32
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Else_2          // jump if edges required
s_and_b32 s30, 31, s[sgprSizeJ]                    // s30 = s[sgprSizeJ] % 32
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Then_2          // jump if edges required
label_GW_B0_FD0_VW1_NonEdge_2:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
v_add_lshl_u32 v23, v19, v16, 0x2                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=16, coord0Vgpr=16

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10+1], s[sgprAlpha], v[vgprValuC+10+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12+1], s[sgprAlpha], v[vgprValuC+12+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14+1], s[sgprAlpha], v[vgprValuC+14+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v2, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v3, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v4, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v5, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v6, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v7, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v8, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v9, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v10, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v11, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v12, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v13, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v14, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b32 v15, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW1_NonEdgeEnd_2:
label_GW_B0_FD0_VW1_Else_2:
label_GW_B0_FD0_VW1_Then_2:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=72 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v22, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v23, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v23, v22, v23, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v24, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v24, v22, v24, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v25, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v25, v22, v25, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v26, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v22, v26, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v27, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v22, v27, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v28, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v22, v28, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v29, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v29, v22, v29, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v30, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v22, v30, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v31, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v22, v31, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v32, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v22, v32, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v33, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v33, v22, v33, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v34, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v22, v34, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v35, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v35, v22, v35, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v36, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v22, v36, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v37, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, v22, v37, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v38, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v22, v38, s32                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10+1], s[sgprAlpha], v[vgprValuC+10+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12+1], s[sgprAlpha], v[vgprValuC+12+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14+1], s[sgprAlpha], v[vgprValuC+14+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v24, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v2, v25, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v3, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v4, v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v5, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v6, v29, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v7, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v8, v31, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v9, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v10, v33, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v11, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v12, v35, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v13, v36, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v14, v37, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v15, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_branch label_GW_End_2                            // jump to end
label_GW_B1:
label_GW_B1_FD0:
s_and_b32 s30, 31, s[sgprSizeI]                    // s30 = s[sgprSizeI] % 32
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_Else            // jump if edges required
s_and_b32 s30, 31, s[sgprSizeJ]                    // s30 = s[sgprSizeJ] % 32
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_Then            // jump if edges required
label_GW_B1_FD0_VW1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=70 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v24, v18, v16, 0x2                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=16, coord0Vgpr=16
buffer_load_b32 v25, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_b32 v26, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_b32 v27, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_b32 v28, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:96 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 32                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_b32 v29, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_b32 v30, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
buffer_load_b32 v31, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
buffer_load_b32 v32, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:96 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 32                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_b32 v33, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_b32 v34, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
buffer_load_b32 v35, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
buffer_load_b32 v36, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:96 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 32                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_b32 v37, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_b32 v38, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
buffer_load_b32 v39, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
buffer_load_b32 v40, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:96 // load C
v_add_lshl_u32 v23, v19, v16, 0x2                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=16, coord0Vgpr=16

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10+1], s[sgprAlpha], v[vgprValuC+10+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12+1], s[sgprAlpha], v[vgprValuC+12+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14+1], s[sgprAlpha], v[vgprValuC+14+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // vlcnt(15) = 16 - 1 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+0], v25, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v0, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 16 - 2 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+1], v26, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v1, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 16 - 3 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+2], v27, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v2, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 16 - 4 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+3], v28, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v3, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 16 - 5 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+4], v29, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v4, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 16 - 6 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+5], v30, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v5, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 16 - 7 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+6], v31, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v6, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 16 - 8 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+7], v32, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v7, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 16 - 9 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+8], v33, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v8, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 16 - 10 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+9], v34, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v9, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 16 - 11 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+10], v35, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v10, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 16 - 12 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+11], v36, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v11, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 16 - 13 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+12], v37, s[sgprBeta]       // finalSum = sum*alpha + C*beta
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v12, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 16 - 14 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+13], v38, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v13, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 16 - 15 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+14], v39, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v14, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 16 - 16 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+15], v40, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v15, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW1_NonEdgeEnd:
label_GW_B1_FD0_VW1_Else:
label_GW_B1_FD0_VW1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=35 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (1,2,0,0:vw1); (1,3,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (2,2,0,0:vw1); (2,3,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (3,2,0,0:vw1); (3,3,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v22, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v24, v18, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v24, v22, v24, s32                   // LDC clip if OOB. offset
buffer_load_b32 v23, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v24, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v24, v22, v24, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v26, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v22, v26, s32                   // LDC clip if OOB. offset
buffer_load_b32 v25, v26, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v26, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v22, v26, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v28, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v22, v28, s32                   // LDC clip if OOB. offset
buffer_load_b32 v27, v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v28, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v22, v28, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v30, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v22, v30, s32                   // LDC clip if OOB. offset
buffer_load_b32 v29, v30, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v30, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v22, v30, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v32, v18, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v22, v32, s32                   // LDC clip if OOB. offset
buffer_load_b32 v31, v32, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v32, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v22, v32, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v34, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v22, v34, s32                   // LDC clip if OOB. offset
buffer_load_b32 v33, v34, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v34, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v22, v34, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v36, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v22, v36, s32                   // LDC clip if OOB. offset
buffer_load_b32 v35, v36, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v36, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v22, v36, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v38, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v22, v38, s32                   // LDC clip if OOB. offset
buffer_load_b32 v37, v38, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v38, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v22, v38, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v40, v18, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v22, v40, s32                   // LDC clip if OOB. offset
buffer_load_b32 v39, v40, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v40, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v22, v40, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v42, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v22, v42, s32                   // LDC clip if OOB. offset
buffer_load_b32 v41, v42, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v42, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v22, v42, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v44, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v22, v44, s32                   // LDC clip if OOB. offset
buffer_load_b32 v43, v44, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v44, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v22, v44, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v46, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, v22, v46, s32                   // LDC clip if OOB. offset
buffer_load_b32 v45, v46, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v46, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v46, v22, v46, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v17, vcc_lo, v17, 8                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v18, v18, s30                         // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v19, v19, s30                         // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v16, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v48, v18, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, v22, v48, s32                   // LDC clip if OOB. offset
buffer_load_b32 v47, v48, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v48, v19, v16, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, v22, v48, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v20, vcc_lo, v16, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v50, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v22, v50, s32                   // LDC clip if OOB. offset
buffer_load_b32 v49, v50, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v50, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v22, v50, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
v_add_co_u32 v20, vcc_lo, v16, 16                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v52, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v52, v22, v52, s32                   // LDC clip if OOB. offset
buffer_load_b32 v51, v52, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v52, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v52, v22, v52, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
v_add_co_u32 v20, vcc_lo, v16, 24                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v20, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s32, v17, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v54, v18, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v54, v22, v54, s32                   // LDC clip if OOB. offset
buffer_load_b32 v53, v54, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v54, v19, v20, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v54, v22, v54, s32                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha (pk)
v_mul_f32 v[vgprValuC+10+1], s[sgprAlpha], v[vgprValuC+10+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha (pk)
v_mul_f32 v[vgprValuC+12+1], s[sgprAlpha], v[vgprValuC+12+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14+1], s[sgprAlpha], v[vgprValuC+14+1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+0], v23, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v0, v24, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+1], v25, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v1, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+2], v27, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v2, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+3], v29, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v3, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+4], v31, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v4, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+5], v33, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v5, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+6], v35, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v6, v36, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+7], v37, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v7, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+8], v39, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v8, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+9], v41, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v9, v42, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+10], v43, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v10, v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+11], v45, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v11, v46, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+12], v47, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v12, v48, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+13], v49, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v13, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+14], v51, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v14, v52, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+15], v53, s[sgprBeta]       // finalSum = sum*alpha + C*beta
buffer_store_b32 v15, v54, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_branch label_GW_End_2                            // jump to end
label_GW_End_2:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
