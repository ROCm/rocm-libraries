
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Cijk_Ailk_Bljk_S_B_UserArgs_MT16x16x8_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT2_2_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1
.globl Cijk_Ailk_Bljk_S_B_UserArgs_MT16x16x8_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT2_2_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1
.p2align 8
.type Cijk_Ailk_Bljk_S_B_UserArgs_MT16x16x8_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT2_2_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_S_B_UserArgs_MT16x16x8_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT2_2_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 48 // vgprs
  .amdhsa_next_free_sgpr 72 // sgprs
  .amdhsa_group_segment_fixed_size 3104 // lds bytes
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
/* Num VGPR   =48 */
/* Num AccVGPR=0 */
/* Num SGPR   =72 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 2 x 2 */
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
  - .name: Cijk_Ailk_Bljk_S_B_UserArgs_MT16x16x8_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT2_2_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1
    .symbol: 'Cijk_Ailk_Bljk_S_B_UserArgs_MT16x16x8_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT2_2_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1.kd'
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
    .group_segment_fixed_size:   3104
    .kernarg_segment_align:      8
    .kernarg_segment_size:       104
    .max_flat_workgroup_size:    64
    .private_segment_fixed_size: 0
    .sgpr_count:                 72
    .sgpr_spill_count:           0
    .vgpr_count:                 48
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_S_B_UserArgs_MT16x16x8_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB1_LPM0_LRVW1_LWPMn1_MIAV0_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR1_PKA0_SGROB0_SIA1_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TT2_2_TLDS0_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG8_8_1:
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
/* ValuC range: [0-4),  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 12
.set vgprLocalWriteAddrA, 6
.set vgprLocalWriteAddrB, 7
.set vgprGlobalReadOffsetA, 4
.set vgprGlobalReadOffsetB, 5
.set vgprLocalReadAddrA, 8
.set vgprLocalReadAddrB, 9
.set vgprSerial, 26

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+5
.set vgprG2LA_BASE, vgprBase+10
.set vgprG2LB_BASE, vgprBase+12
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+2
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+2
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

.set MT0, 16
.set MT1, 16
.set DepthU, 8
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
.macro MAC_2x2_X0
    // Component.MAC.MAC_F32_Plain
    v_fmac_f32 v[vgprValuC+0+0], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0]
    s_setprio 1                                        // Raise priority while processing macs
    v_fmac_f32 v[vgprValuC+1+0], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+0]
    v_fmac_f32 v[vgprValuC+0+2], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+1]
    v_fmac_f32 v[vgprValuC+1+2], v[vgprValuA_X0_I0+1], v[vgprValuB_X0_I0+1]
    s_setprio 0                                        // Reset priority after macs
.endm
.macro MAC_2x2_X1
    // Component.MAC.MAC_F32_Plain
    v_fmac_f32 v[vgprValuC+0+0], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+0]
    s_setprio 1                                        // Raise priority while processing macs
    v_fmac_f32 v[vgprValuC+1+0], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+0]
    v_fmac_f32 v[vgprValuC+0+2], v[vgprValuA_X1_I0+0], v[vgprValuB_X1_I0+1]
    v_fmac_f32 v[vgprValuC+1+2], v[vgprValuA_X1_I0+1], v[vgprValuB_X1_I0+1]
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
s_mov_b32 m0, 0xc20                                // LDS clamp at 3104 bytes
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
v_cvt_f64_u32 v[14:15], s53                        // s49 = s[sgprWorkGroup0] / s53
v_rcp_f64 v[14:15], v[14:15]                       // s49 = s[sgprWorkGroup0] / s53
v_cvt_f64_u32 v[16:17], s[sgprWorkGroup0]          // s49 = s[sgprWorkGroup0] / s53
v_mul_f64 v[14:15], v[14:15], v[16:17]             // s49 = s[sgprWorkGroup0] / s53
v_cvt_u32_f64 v14, v[14:15]                        // s49 = s[sgprWorkGroup0] / s53
v_mul_lo_u32 v15, v14, s53                         // s49 = s[sgprWorkGroup0] / s53
v_sub_nc_u32 v16, s[sgprWorkGroup0], v15           // s49 = s[sgprWorkGroup0] / s53
v_cmp_ge_u32 vcc_lo, v16, s53                      // s49 = s[sgprWorkGroup0] / s53
s_mov_b32 exec_lo vcc_lo                           // s49 = s[sgprWorkGroup0] / s53
v_add_nc_u32 v14, v14, 1                           // s49 = s[sgprWorkGroup0] / s53
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v15, v14, s53                         // s49 = s[sgprWorkGroup0] / s53
v_sub_nc_u32 v16, s[sgprWorkGroup0], v15           // s49 = s[sgprWorkGroup0] / s53
v_readfirstlane_b32 s49, v14                       // quotient
v_readfirstlane_b32 s50, v16                       // remainder
s_mul_i32 s49, s49, s53
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s50, s50, s52
/* temp0 = temp0 + temp1 */
s_add_u32 s49, s49, s50
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[14:15], s53                        // s50 = s23 / s53
v_rcp_f64 v[14:15], v[14:15]                       // s50 = s23 / s53
v_cvt_f64_u32 v[16:17], s23                        // s50 = s23 / s53
v_mul_f64 v[14:15], v[14:15], v[16:17]             // s50 = s23 / s53
v_cvt_u32_f64 v14, v[14:15]                        // s50 = s23 / s53
v_mul_lo_u32 v15, v14, s53                         // s50 = s23 / s53
v_sub_nc_u32 v16, s23, v15                         // s50 = s23 / s53
v_cmp_ge_u32 vcc_lo, v16, s53                      // s50 = s23 / s53
s_mov_b32 exec_lo vcc_lo                           // s50 = s23 / s53
v_add_nc_u32 v14, v14, 1                           // s50 = s23 / s53
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s50, v14                       // quotient
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
/* init: add vgpr [12...33) to pool */
/* init: add vgpr [0...4) to pool */
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
v_lshlrev_b32 v0, 4, v0                            // LSU offset: lsuoffset = sgid*(MT0+PAD)
                                                   // Final Offset: lrAOffset * VW (multiplier is 1, do nothing)
v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x2  // Final Offset: add padding 0 per block 0

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(64)
v_mul_lo_u32 v0, 17, v0                            // LSU offset: lsuoffset = sgid*(MT1+PAD)
                                                   // Final Offset: lrBOffset * VW (multiplier is 1, do nothing)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x2  // Final Offset: add padding 4 per block 0

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 4, v[vgprSerial]                 // 1 = Serial / 16
v_and_b32 v0, 15, v[vgprSerial]                    // 0 = Serial % 16
v_mov_b32 v12, v1                                  // copy for GlobalSplitU
/* LVCB = 8 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 3, v[vgprSerial]                 // 2 = Serial / 8
v_and_b32 v3, 7, v[vgprSerial]                     // 3 = Serial % 8
v_mov_b32 v13, v3                                  // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v12 */
/* lwaUnrollAssignmentB = v13 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v12    // lwAL**(MTA + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x2 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x11, v13    // lwBL**(MTB + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x2 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpeDS
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x200, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=512
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
v_mov_b32 v16, MT0                                 // set MT0 into sgpr
v_mov_b32 v15, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v14, v16                             // v14 = ceil(v15 / v16)
v_rcp_iflag_f32 v14, v14                           // v14 = ceil(v15 / v16)
v_cvt_f32_u32 v17, v15                             // v14 = ceil(v15 / v16)
v_mul_f32 v14, v14, v17                            // v14 = ceil(v15 / v16)
v_cvt_u32_f32 v14, v14                             // v14 = ceil(v15 / v16)
v_mul_u32_u24 v17, v14, v16                        // v14 = ceil(v15 / v16)
v_sub_nc_u32 v17, v15, v17                         // v14 = ceil(v15 / v16)
v_cmp_ne_u32 vcc_lo, v17, 0                        // v14 = ceil(v15 / v16)
v_add_co_ci_u32 v14, vcc_lo, v14, 0, vcc_lo        // ceil
v_mov_b32 v16, MT1                                 // set MT1 into sgpr
v_mov_b32 v15, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v14     // set back to numWorkGroup0
v_cvt_f32_u32 v14, v16                             // v14 = ceil(v15 / v16)
v_rcp_iflag_f32 v14, v14                           // v14 = ceil(v15 / v16)
v_cvt_f32_u32 v17, v15                             // v14 = ceil(v15 / v16)
v_mul_f32 v14, v14, v17                            // v14 = ceil(v15 / v16)
v_cvt_u32_f32 v14, v14                             // v14 = ceil(v15 / v16)
v_mul_u32_u24 v17, v14, v16                        // v14 = ceil(v15 / v16)
v_sub_nc_u32 v17, v15, v17                         // v14 = ceil(v15 / v16)
v_cmp_ne_u32 vcc_lo, v17, 0                        // v14 = ceil(v15 / v16)
v_add_co_ci_u32 v14, vcc_lo, v14, 0, vcc_lo        // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v14     // set back to numWorkGroup1

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v14, s16                             // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v14, v14                           // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v15, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v14, v14, v15                            // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v14, v14                             // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v15, v14, s16                        // s16 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v15, s[sgprWorkGroup0], v15           // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v15, s16                      // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo vcc_lo                           // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v14, 1, v14                           // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v15, s16                      // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v14, v14, 1                           // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v14                       // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v14, s[sgprNumWorkGroups0]           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v14, v14                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v15, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v14, v14, v15                            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v14, v14                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v15, v14, s[sgprNumWorkGroups0]      // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v15, s[sgprWorkGroup0], v15           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v15, s[sgprNumWorkGroups0]    // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo vcc_lo                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v14, 1, v14                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v15, s[sgprNumWorkGroups0]    // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v14, v14, 1                           // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v14                       // quotient
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
s_lshr_b32 s52, s24, 4                             // s52 = s24 / 16
s_and_b32 s50, 15, s24                             // s50 = s24 % 16
s_addc_u32 s52, s52, 0
s_lshr_b32 s53, s25, 4                             // s53 = s25 / 16
s_and_b32 s50, 15, s25                             // s50 = s25 % 16
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
s_lshr_b32 s52, s24, 4                             // s52 = s24 / 16
s_and_b32 s50, 15, s24                             // s50 = s24 % 16
s_addc_u32 s52, s52, 0
s_lshr_b32 s53, s25, 4                             // s53 = s25 / 16
s_and_b32 s50, 15, s25                             // s50 = s25 % 16
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
/* init: add vgpr [12...33) to pool */
/* init: add vgpr [0...4) to pool */
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
v_lshlrev_b32 v0, 4, v0                            // LSU offset: lsuoffset = sgid*(MT0+PAD)
                                                   // Final Offset: lrAOffset * VW (multiplier is 1, do nothing)
v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x2  // Final Offset: add padding 0 per block 0

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(64)
v_mul_lo_u32 v0, 17, v0                            // LSU offset: lsuoffset = sgid*(MT1+PAD)
                                                   // Final Offset: lrBOffset * VW (multiplier is 1, do nothing)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x2  // Final Offset: add padding 4 per block 0

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 4, v[vgprSerial]                 // 1 = Serial / 16
v_and_b32 v0, 15, v[vgprSerial]                    // 0 = Serial % 16
v_mov_b32 v12, v1                                  // copy for GlobalSplitU
/* LVCB = 8 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 3, v[vgprSerial]                 // 2 = Serial / 8
v_and_b32 v3, 7, v[vgprSerial]                     // 3 = Serial % 8
v_mov_b32 v13, v3                                  // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v12 */
/* lwaUnrollAssignmentB = v13 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v12    // lwAL**(MTA + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x2 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x11, v13    // lwBL**(MTB + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x2 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpeDS
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x200, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=512
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
v_mov_b32 v16, MT0                                 // set MT0 into sgpr
v_mov_b32 v15, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v14, v16                             // v14 = ceil(v15 / v16)
v_rcp_iflag_f32 v14, v14                           // v14 = ceil(v15 / v16)
v_cvt_f32_u32 v17, v15                             // v14 = ceil(v15 / v16)
v_mul_f32 v14, v14, v17                            // v14 = ceil(v15 / v16)
v_cvt_u32_f32 v14, v14                             // v14 = ceil(v15 / v16)
v_mul_u32_u24 v17, v14, v16                        // v14 = ceil(v15 / v16)
v_sub_nc_u32 v17, v15, v17                         // v14 = ceil(v15 / v16)
v_cmp_ne_u32 vcc_lo, v17, 0                        // v14 = ceil(v15 / v16)
v_add_co_ci_u32 v14, vcc_lo, v14, 0, vcc_lo        // ceil
v_mov_b32 v16, MT1                                 // set MT1 into sgpr
v_mov_b32 v15, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v14     // set back to numWorkGroup0
v_cvt_f32_u32 v14, v16                             // v14 = ceil(v15 / v16)
v_rcp_iflag_f32 v14, v14                           // v14 = ceil(v15 / v16)
v_cvt_f32_u32 v17, v15                             // v14 = ceil(v15 / v16)
v_mul_f32 v14, v14, v17                            // v14 = ceil(v15 / v16)
v_cvt_u32_f32 v14, v14                             // v14 = ceil(v15 / v16)
v_mul_u32_u24 v17, v14, v16                        // v14 = ceil(v15 / v16)
v_sub_nc_u32 v17, v15, v17                         // v14 = ceil(v15 / v16)
v_cmp_ne_u32 vcc_lo, v17, 0                        // v14 = ceil(v15 / v16)
v_add_co_ci_u32 v14, vcc_lo, v14, 0, vcc_lo        // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v14     // set back to numWorkGroup1

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
v_cvt_f32_u32 v14, s16                             // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v14, v14                           // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v15, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v14, v14, v15                            // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v14, v14                             // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v15, v14, s16                        // s16 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v15, s[sgprWorkGroup0], v15           // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v15, s16                      // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo vcc_lo                           // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v14, 1, v14                           // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v15, s16                      // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v14, v14, 1                           // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v14                       // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v14, s[sgprNumWorkGroups0]           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v14, v14                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v15, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v14, v14, v15                            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v14, v14                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v15, v14, s[sgprNumWorkGroups0]      // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v15, s[sgprWorkGroup0], v15           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v15, s[sgprNumWorkGroups0]    // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo vcc_lo                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v14, 1, v14                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v15, s[sgprNumWorkGroups0]    // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v14, v14, 1                           // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v14                       // quotient
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
.set sgprScalarGlobalReadOffsetB, 67
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
v_cvt_f32_u32 v14, s16                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_rcp_iflag_f32 v14, v14                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_f32_u32 v15, s[sgprWorkGroup1]               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_f32 v14, v14, v15                            // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_u32_f32 v14, v14                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_u32_u24 v15, v14, s16                        // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v15, s[sgprWorkGroup1], v15           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cmp_eq_u32 vcc_lo, v15, s16                      // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_add_nc_u32 v14, 1, v14                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mov_b32 v15, 0                                   // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v15, s16                      // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v14, v14, 1                           // quotient - 1
v_mul_u32_u24 v15, v14, s16                        // re-calculate remainder
v_sub_nc_u32 v15, s[sgprWorkGroup1], v15           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v14         // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v15          // remainder
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
v_cvt_f32_u32 v14, s[sgprNumWorkGroups1]           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v14, v14                           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_f32_u32 v15, s[sgprWorkGroup1]               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v14, v14, v15                            // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_u32_f32 v14, v14                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v15, v14, s[sgprNumWorkGroups1]      // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_sub_nc_u32 v15, s[sgprWorkGroup1], v15           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmp_eq_u32 vcc_lo, v15, s[sgprNumWorkGroups1]    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_nc_u32 v14, 1, v14                           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v15, 0                                   // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v15, s[sgprNumWorkGroups1]    // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v14, v14, 1                           // quotient - 1
v_mul_u32_u24 v15, v14, s[sgprNumWorkGroups1]      // re-calculate remainder
v_sub_nc_u32 v15, s[sgprWorkGroup1], v15           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprGSUSumIdx], v14          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v15         // remainder
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
v_cvt_f64_u32 v[14:15], s16                        // s17 = s[sgprWorkGroup0] / s16
v_rcp_f64 v[14:15], v[14:15]                       // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[16:17], s[sgprWorkGroup0]          // s17 = s[sgprWorkGroup0] / s16
v_mul_f64 v[14:15], v[14:15], v[16:17]             // s17 = s[sgprWorkGroup0] / s16
v_cvt_u32_f64 v14, v[14:15]                        // s17 = s[sgprWorkGroup0] / s16
v_mul_lo_u32 v15, v14, s16                         // s17 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v16, s[sgprWorkGroup0], v15           // s17 = s[sgprWorkGroup0] / s16
v_cmp_ge_u32 vcc_lo, v16, s16                      // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo vcc_lo                           // s17 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v14, v14, 1                           // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v14                       // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup0], s20              // WorkGroup0=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s20, s20, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[14:15], s16                        // s18 = s[sgprNumWorkGroups0] / s16
v_rcp_f64 v[14:15], v[14:15]                       // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_f64_u32 v[16:17], s[sgprNumWorkGroups0]      // s18 = s[sgprNumWorkGroups0] / s16
v_mul_f64 v[14:15], v[14:15], v[16:17]             // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_u32_f64 v14, v[14:15]                        // s18 = s[sgprNumWorkGroups0] / s16
v_mul_lo_u32 v15, v14, s16                         // s18 = s[sgprNumWorkGroups0] / s16
v_sub_nc_u32 v16, s[sgprNumWorkGroups0], v15       // s18 = s[sgprNumWorkGroups0] / s16
v_cmp_ge_u32 vcc_lo, v16, s16                      // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo vcc_lo                           // s18 = s[sgprNumWorkGroups0] / s16
v_add_nc_u32 v14, v14, 1                           // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v14                       // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups0], s19          // NumWorkGroups0=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[14:15], s18                        // s[sgprWorkGroup1] = s20 / s18
v_rcp_f64 v[14:15], v[14:15]                       // s[sgprWorkGroup1] = s20 / s18
v_cvt_f64_u32 v[16:17], s20                        // s[sgprWorkGroup1] = s20 / s18
v_mul_f64 v[14:15], v[14:15], v[16:17]             // s[sgprWorkGroup1] = s20 / s18
v_cvt_u32_f64 v14, v[14:15]                        // s[sgprWorkGroup1] = s20 / s18
v_mul_lo_u32 v15, v14, s18                         // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v16, s20, v15                         // s[sgprWorkGroup1] = s20 / s18
v_cmp_ge_u32 vcc_lo, v16, s18                      // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s20 / s18
v_add_nc_u32 v14, v14, 1                           // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v15, v14, s18                         // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v16, s20, v15                         // s[sgprWorkGroup1] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup1], v14         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v16         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s16, s16                                 // WGM
v_cvt_f64_u32 v[14:15], s16                        // s17 = s[sgprWorkGroup1] / s16
v_rcp_f64 v[14:15], v[14:15]                       // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[16:17], s[sgprWorkGroup1]          // s17 = s[sgprWorkGroup1] / s16
v_mul_f64 v[14:15], v[14:15], v[16:17]             // s17 = s[sgprWorkGroup1] / s16
v_cvt_u32_f64 v14, v[14:15]                        // s17 = s[sgprWorkGroup1] / s16
v_mul_lo_u32 v15, v14, s16                         // s17 = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v16, s[sgprWorkGroup1], v15           // s17 = s[sgprWorkGroup1] / s16
v_cmp_ge_u32 vcc_lo, v16, s16                      // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo vcc_lo                           // s17 = s[sgprWorkGroup1] / s16
v_add_nc_u32 v14, v14, 1                           // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v14                       // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup1], s20              // WorkGroup1=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s20, s20, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[14:15], s16                        // s18 = s[sgprNumWorkGroups1] / s16
v_rcp_f64 v[14:15], v[14:15]                       // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_f64_u32 v[16:17], s[sgprNumWorkGroups1]      // s18 = s[sgprNumWorkGroups1] / s16
v_mul_f64 v[14:15], v[14:15], v[16:17]             // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_u32_f64 v14, v[14:15]                        // s18 = s[sgprNumWorkGroups1] / s16
v_mul_lo_u32 v15, v14, s16                         // s18 = s[sgprNumWorkGroups1] / s16
v_sub_nc_u32 v16, s[sgprNumWorkGroups1], v15       // s18 = s[sgprNumWorkGroups1] / s16
v_cmp_ge_u32 vcc_lo, v16, s16                      // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo vcc_lo                           // s18 = s[sgprNumWorkGroups1] / s16
v_add_nc_u32 v14, v14, 1                           // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v14                       // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups1], s19          // NumWorkGroups1=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[14:15], s18                        // s[sgprWorkGroup0] = s20 / s18
v_rcp_f64 v[14:15], v[14:15]                       // s[sgprWorkGroup0] = s20 / s18
v_cvt_f64_u32 v[16:17], s20                        // s[sgprWorkGroup0] = s20 / s18
v_mul_f64 v[14:15], v[14:15], v[16:17]             // s[sgprWorkGroup0] = s20 / s18
v_cvt_u32_f64 v14, v[14:15]                        // s[sgprWorkGroup0] = s20 / s18
v_mul_lo_u32 v15, v14, s18                         // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v16, s20, v15                         // s[sgprWorkGroup0] = s20 / s18
v_cmp_ge_u32 vcc_lo, v16, s18                      // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup0] = s20 / s18
v_add_nc_u32 v14, v14, 1                           // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v15, v14, s18                         // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v16, s20, v15                         // s[sgprWorkGroup0] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup0], v14         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v16         // remainder
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
s_mul_hi_u32 s19, s[sgprWorkGroup0], 16            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup0], 16               // WorkGroup[01] * MT
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s17, 8, s[sgprGSUSumIdx]              // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 8, s[sgprGSUSumIdx]                 // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 3 // s[LoopCounterL] = s[sgprSizesSum] / 8
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v14, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v14, v14                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v15, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v14, v14, v15                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v14, v14                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v15, v14, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v15, s[sgprLoopCounterL], v15         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v15, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v14, 1, v14                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v15, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v15, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v14, v14, 1                           // quotient - 1
v_mul_u32_u24 v15, v14, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v15, s[sgprLoopCounterL], v15         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v14       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v15        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 8                           // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 8                              // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
s_mul_hi_u32 s19, s[sgprWorkGroup1], 16            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s17, 8, s[sgprGSUSumIdx]              // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 8, s[sgprGSUSumIdx]                 // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 3 // s[LoopCounterL] = s[sgprSizesSum] / 8
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v14, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v14, v14                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v15, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v14, v14, v15                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v14, v14                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v15, v14, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v15, s[sgprLoopCounterL], v15         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v15, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v14, 1, v14                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v15, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v15, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v14, v14, 1                           // quotient - 1
v_mul_u32_u24 v15, v14, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v15, s[sgprLoopCounterL], v15         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v14       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v15        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 8                           // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 8                              // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  0,  1, 14 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideAL], 4 // compute offset diff (scaled unrollDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+0], s[sgprScalarGlobalReadOffsetA+0], 0x2 // scalar offset *= bytes/element
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 14 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideB1J], 8 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], 0x2 // scalar offset *= bytes/element
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
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 3 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 8
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
buffer_load_b32 v[vgprG2LB+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LB+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0

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


s_mul_i32 s70, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s69, s70, s[sgprStrideC1J]            // ScaleC s70 by Stride
s_mul_i32 s68, s70, s[sgprStrideC1J]               // ScaleC s70 by Stride
s_lshl_b64 s[68:69], s[68:69], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s68    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s69   // add hi to SRD
s_mul_hi_u32 s69, s70, s[sgprStrideD1J]            // ScaleD s70 by Stride
s_mul_i32 s68, s70, s[sgprStrideD1J]               // ScaleD s70 by Stride
s_lshl_b64 s[68:69], s[68:69], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s68    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s69   // add hi to SRD

s_mul_hi_u32 s69, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s68, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[68:69], s[68:69], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s69       // add hi to SRD
s_mul_hi_u32 s69, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s68, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[68:69], s[68:69], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s68        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s69       // add hi to SRD

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

/* initC: remove ValuC vgpr buffer [0...4) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [12...21) from pool */
v_mov_b32 v[vgprValuC+0], 0                        // initC
v_mov_b32 v[vgprValuC+1], 0                        // initC
v_mov_b32 v[vgprValuC+2], 0                        // initC
v_mov_b32 v[vgprValuC+3], 0                        // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc1
s_getpc_b64 s[68:69]                               // addr of next instr
s_add_i32 s70, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s68, s68, s70                            // add target branch offset
s_addc_u32 s69, s69, 0                             // add high and carry
s_setpc_b64 s[68:69]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_T8JHFHKM7BO5OHXW:
s_waitcnt vmcnt(0)                                 // wait for global read

/* local write a */
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:256 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 256

/* local write b */
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:32 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 32

/* local write swap a */

/* (EPS=1) local write swap internal offset -> 2048 */

/* local write swap b */

/* (EPS=1) local write swap internal offset -> 2048 */
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local read prefetch a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:32

/* local read prefetch b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:0
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:32

/* local read inc a */
/* N/A, lro->16 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/* local read inc b */
/* N/A, lro->17 */
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
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:64
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:96
buffer_load_b32 v[vgprG2LA+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LB+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:68
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:100

/* local read increment a */
/* N/A, lro->32 */
/* self.localReadDoCntA 2 self.localReadDoCntB 2 */

/* local read increment b */
/* N/A, lro->34 */
/* self.localReadDoCntA 2 self.localReadDoCntB 2 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0 for iteration == 0
MAC_2x2_X0


/* iter 1 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:128
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:160
buffer_load_b32 v[vgprG2LB+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:136
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:168

/* local read increment a */
/* N/A, lro->48 */
/* self.localReadDoCntA 3 self.localReadDoCntB 3 */

/* local read increment b */
/* N/A, lro->51 */
/* self.localReadDoCntA 3 self.localReadDoCntB 3 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


/* iter 2 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:192
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:224

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s69       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:204
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:236

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 4 self.localReadDoCntB 4 */

/* local read increment b */
/* N/A, lro->68 */
/* self.localReadDoCntA 4 self.localReadDoCntB 4 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X0


/* iter 3 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:256
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:288

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s69       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:272
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:304

/* local read increment a */
/* N/A, lro->80 */
/* self.localReadDoCntA 5 self.localReadDoCntB 5 */

/* local read increment b */
/* N/A, lro->85 */
/* self.localReadDoCntA 5 self.localReadDoCntB 5 */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:2048 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 2048
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=0
MAC_2x2_X1


/* iter 4 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:320
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:352

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:340
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:372

/* local read increment a */
/* N/A, lro->96 */
/* self.localReadDoCntA 6 self.localReadDoCntB 6 */

/* local read increment b */
/* N/A, lro->102 */
/* self.localReadDoCntA 6 self.localReadDoCntB 6 */
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:2304 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 2304
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=0
MAC_2x2_X0


/* iter 5 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:384
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:416

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:408
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:440

/* local read increment a */
/* N/A, lro->112 */
/* self.localReadDoCntA 7 self.localReadDoCntB 7 */

/* local read increment b */
/* N/A, lro->119 */
/* self.localReadDoCntA 7 self.localReadDoCntB 7 */
/* sched write - iter 5 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:2048 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 2048
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=0
MAC_2x2_X1


/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:448
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:480

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:476
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:508
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:2080 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2080

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 0 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 0 */

/* local read swap offsets a */

/* local read swap internal offset -> 2048 */

/* local read swap offsets b */

/* local read swap internal offset -> 2048 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=0
MAC_2x2_X0


/* iter 7 */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:2048
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:2080

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:2048
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:2080

/* local read increment a */
/* N/A, lro->16 */
/* self.localReadDoCntA 9 self.localReadDoCntB 9 */

/* local read increment b */
/* N/A, lro->17 */
/* self.localReadDoCntA 9 self.localReadDoCntB 9 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


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
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:2112
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:2144
buffer_load_b32 v[vgprG2LA+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LB+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:2116
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:2148

/* local read increment a */
/* N/A, lro->32 */
/* self.localReadDoCntA 10 self.localReadDoCntB 10 */

/* local read increment b */
/* N/A, lro->34 */
/* self.localReadDoCntA 10 self.localReadDoCntB 10 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0 for iteration == 0
MAC_2x2_X0


/* iter 1 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:2176
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:2208
buffer_load_b32 v[vgprG2LB+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:2184
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:2216

/* local read increment a */
/* N/A, lro->48 */
/* self.localReadDoCntA 11 self.localReadDoCntB 11 */

/* local read increment b */
/* N/A, lro->51 */
/* self.localReadDoCntA 11 self.localReadDoCntB 11 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


/* iter 2 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:2240
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:2272

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s68        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s69       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:2252
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:2284

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 12 self.localReadDoCntB 12 */

/* local read increment b */
/* N/A, lro->68 */
/* self.localReadDoCntA 12 self.localReadDoCntB 12 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X0


/* iter 3 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:2304
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:2336

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s68, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s69, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s68        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s69       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:2320
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:2352

/* local read increment a */
/* N/A, lro->80 */
/* self.localReadDoCntA 13 self.localReadDoCntB 13 */

/* local read increment b */
/* N/A, lro->85 */
/* self.localReadDoCntA 13 self.localReadDoCntB 13 */
/* sched write - iter 3 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=0
MAC_2x2_X1


/* iter 4 */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:2368
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:2400

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:2388
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:2420

/* local read increment a */
/* N/A, lro->96 */
/* self.localReadDoCntA 14 self.localReadDoCntB 14 */

/* local read increment b */
/* N/A, lro->102 */
/* self.localReadDoCntA 14 self.localReadDoCntB 14 */
/* sched write - iter 4 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:256 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 256
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=0
MAC_2x2_X0


/* iter 5 */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:2432
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:2464

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:2456
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:2488

/* local read increment a */
/* N/A, lro->112 */
/* self.localReadDoCntA 15 self.localReadDoCntB 15 */

/* local read increment b */
/* N/A, lro->119 */
/* self.localReadDoCntA 15 self.localReadDoCntB 15 */
/* sched write - iter 5 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=0
MAC_2x2_X1


/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:2496
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:2528

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:2524
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:2556
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:32 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 32

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 2048 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 2048 */

/* local read swap offsets a */

/* local read swap internal offset -> 0 */

/* local read swap offsets b */

/* local read swap internal offset -> 0 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=0
MAC_2x2_X0


/* iter 7 */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:32

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:0
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:32

/* local read increment a */
/* N/A, lro->16 */
/* self.localReadDoCntA 17 self.localReadDoCntB 17 */

/* local read increment b */
/* N/A, lro->17 */
/* self.localReadDoCntA 17 self.localReadDoCntB 17 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


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
v_xor_b32 v[vgprLocalReadAddrA], 0x800, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x800, v[vgprLocalReadAddrB] // swap Red Blk
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

s_and_b32 s68, 15, s[sgprSizeI]                    // s68 = s[sgprSizeI] % 16
s_add_u32 s69, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s69                // wg0 >= nwg0-1 ?
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s68, 15, s[sgprSizeJ]                    // s68 = s[sgprSizeJ] % 16
s_add_u32 s69, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s69                // wg1 >= nwg1-1
s_cselect_b32 s68, s68, 0                          // set rem
s_cmpk_gt_u32 s68, 0                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required

s_and_b32 s69, 7, s[sgprSizesSum+0]                // s69 = s[sgprSizesSum+0] % 8
s_cmp_eq_u32 s69, 0                                // numIterL == 0
s_cbranch_scc0 label_OptNLL_End                    // skip if tail loop required

/* iter 0 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:64
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:96

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:68
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:100

/* local read increment a */
/* N/A, lro->32 */
/* self.localReadDoCntA 18 self.localReadDoCntB 18 */

/* local read increment b */
/* N/A, lro->34 */
/* self.localReadDoCntA 18 self.localReadDoCntB 18 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0 for iteration == 0
MAC_2x2_X0


/* iter 1 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:128
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:160

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:136
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:168

/* local read increment a */
/* N/A, lro->48 */
/* self.localReadDoCntA 19 self.localReadDoCntB 19 */

/* local read increment b */
/* N/A, lro->51 */
/* self.localReadDoCntA 19 self.localReadDoCntB 19 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


/* iter 2 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:192
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:224

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:204
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:236

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 20 self.localReadDoCntB 20 */

/* local read increment b */
/* N/A, lro->68 */
/* self.localReadDoCntA 20 self.localReadDoCntB 20 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X0


/* iter 3 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:256
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:288

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:272
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:304

/* local read increment a */
/* N/A, lro->80 */
/* self.localReadDoCntA 21 self.localReadDoCntB 21 */

/* local read increment b */
/* N/A, lro->85 */
/* self.localReadDoCntA 21 self.localReadDoCntB 21 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


/* iter 4 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:320
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:352

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:340
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:372

/* local read increment a */
/* N/A, lro->96 */
/* self.localReadDoCntA 22 self.localReadDoCntB 22 */

/* local read increment b */
/* N/A, lro->102 */
/* self.localReadDoCntA 22 self.localReadDoCntB 22 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X0


/* iter 5 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:384
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:416

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:408
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:440

/* local read increment a */
/* N/A, lro->112 */
/* self.localReadDoCntA 23 self.localReadDoCntB 23 */

/* local read increment b */
/* N/A, lro->119 */
/* self.localReadDoCntA 23 self.localReadDoCntB 23 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


/* iter 6 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:448
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:480

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:476
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:508
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X0


/* iter 7 (last unrolled loop) */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 25 self.localReadDoCntB 25 */

/* local read increment b */
/* N/A, lro->136 */
/* self.localReadDoCntA 25 self.localReadDoCntB 25 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_2x2_X1

label_toPGR1end_OptNLL:
/* Stores for OptNLL */
label_Summation_End_OptNLL:
/* endSummation: add vgpr [4...26) to pool */
/* load store sgprs */
/* computeStoreVgprs */
v_lshrrev_b32 v5, 3, v[vgprSerial]                 // 5 = Serial / 8
v_and_b32 v4, 7, v[vgprSerial]                     // 4 = Serial % 8
                                                   // v4 = v4 * 1 (multiplier is 1, do nothing)
v_mul_lo_u32 v6, v5, s[sgprStrideC1J]              // rowStart vgpr
v_mul_lo_u32 v7, v5, s[sgprStrideD1J]              // rowStart vgpr

s_mul_i32 s30, 0x10, s[sgprWorkGroup0]             // s30 = wg0*MT0
v_add_nc_u32 v4, s30, v4                           // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s32, 0x10, s[sgprWorkGroup1]             // <- wg1*MT1
v_add_nc_u32 v5, s32, v5                           // coord1 = tid1*VW + wg1*MT1

/******************************************/
/* Global Write Elements                  */
/******************************************/
label_GW_B0:
label_GW_B0_FD0:
label_GW_B0_FD0_VW1_Then:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=4 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_lshl_u32 v11, v7, v4, 0x2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=4, coord0Vgpr=4

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v2, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v3, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
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
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:64
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:96

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:68
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:100

/* local read increment a */
/* N/A, lro->32 */
/* self.localReadDoCntA 18 self.localReadDoCntB 18 */

/* local read increment b */
/* N/A, lro->34 */
/* self.localReadDoCntA 18 self.localReadDoCntB 18 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0 for iteration == 0
MAC_2x2_X0


/* iter 1 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:128
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:160

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:136
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:168

/* local read increment a */
/* N/A, lro->48 */
/* self.localReadDoCntA 19 self.localReadDoCntB 19 */

/* local read increment b */
/* N/A, lro->51 */
/* self.localReadDoCntA 19 self.localReadDoCntB 19 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


/* iter 2 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:192
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:224

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:204
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:236

/* local read increment a */
/* N/A, lro->64 */
/* self.localReadDoCntA 20 self.localReadDoCntB 20 */

/* local read increment b */
/* N/A, lro->68 */
/* self.localReadDoCntA 20 self.localReadDoCntB 20 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X0


/* iter 3 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:256
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:288

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:272
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:304

/* local read increment a */
/* N/A, lro->80 */
/* self.localReadDoCntA 21 self.localReadDoCntB 21 */

/* local read increment b */
/* N/A, lro->85 */
/* self.localReadDoCntA 21 self.localReadDoCntB 21 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


/* iter 4 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:320
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:352

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:340
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:372

/* local read increment a */
/* N/A, lro->96 */
/* self.localReadDoCntA 22 self.localReadDoCntB 22 */

/* local read increment b */
/* N/A, lro->102 */
/* self.localReadDoCntA 22 self.localReadDoCntB 22 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X0


/* iter 5 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:384
ds_load_b32 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:416

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:408
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:440

/* local read increment a */
/* N/A, lro->112 */
/* self.localReadDoCntA 23 self.localReadDoCntB 23 */

/* local read increment b */
/* N/A, lro->119 */
/* self.localReadDoCntA 23 self.localReadDoCntB 23 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X1


/* iter 6 (last unrolled loop) */

/* local read a */
ds_load_b32 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:448
ds_load_b32 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:480

/* local read b */
ds_load_b32 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB+0] offset:476
ds_load_b32 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB+0] offset:508
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=0
MAC_2x2_X0


/* iter 7 (last unrolled loop) */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->128 */
/* self.localReadDoCntA 25 self.localReadDoCntB 25 */

/* local read increment b */
/* N/A, lro->136 */
/* self.localReadDoCntA 25 self.localReadDoCntB 25 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
MAC_2x2_X1

label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [12...21) to pool */

/* Tail: add address/G2L vgpr [21...26) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */
v_and_b32 v[vgprLocalWriteAddrA], 0xf007ff, v[vgprLocalWriteAddrA] // reset to Red

/* local write reset offsets b */
v_and_b32 v[vgprLocalWriteAddrB], 0xf007ff, v[vgprLocalWriteAddrB] // reset to Red
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (2,2,0) */
.set vgprG2LA_BASE, 12
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 14
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 7, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 8
s_and_b32 s68, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s69, s[sgprSizesSum], 3                 // s69 = s[sgprSizesSum] / 8
s_and_b32 s70, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v16, s70                             // s68 = s69 / s70
v_rcp_iflag_f32 v16, v16                           // s68 = s69 / s70
v_cvt_f32_u32 v17, s69                             // s68 = s69 / s70
v_mul_f32 v16, v16, v17                            // s68 = s69 / s70
v_cvt_u32_f32 v16, v16                             // s68 = s69 / s70
v_mul_u32_u24 v17, v16, s70                        // s68 = s69 / s70
v_sub_nc_u32 v17, s69, v17                         // s68 = s69 / s70
v_cmp_eq_u32 vcc_lo, v17, s70                      // s68 = s69 / s70
s_mov_b32 exec_lo vcc_lo                           // s68 = s69 / s70
v_add_nc_u32 v16, 1, v16                           // s68 = s69 / s70
v_mov_b32 v17, 0                                   // s[sgprGSUSumIdx+1] = s69 % s70
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v17, s70                      // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v16, v16, 1                           // quotient - 1
v_mul_u32_u24 v17, v16, s70                        // re-calculate remainder
v_sub_nc_u32 v17, s69, v17                         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s68, v16                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v17        // remainder
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

/* Tail global read A */
/* g2l=0, load component 0 */
buffer_load_b32 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=1, load component 0 */
buffer_load_b32 v[vgprG2LA+1+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // load one buffer value

/* Update M0 for DTLDS */

/* Tail global read B */
/* g2l=0, load component 0 */
buffer_load_b32 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=1, load component 0 */
buffer_load_b32 v[vgprG2LB+1+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // load one buffer value
s_waitcnt vmcnt(0)                                 // 2wait for global read
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local write a */
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:256 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 256

/* local write b */
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_store_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:32 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 32

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 12
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+2
.set vgprValuB_X0_I0_BASE, 16
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+2

/* Tail: local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA+0], 0x7ff, v[vgprLocalReadAddrA+0] // reset Red,Blk -> Red

/* Tail: local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB+0], 0x7ff, v[vgprLocalReadAddrB+0] // reset Red,Blk -> Red

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

/* local read b */
ds_load_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:0
ds_load_b32 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:32

/* local read inc a */
s_mov_b32 s8, 64                                   // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc_lo, s8, v[vgprLocalReadAddrA+0] // lrA += 64 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s8, 68                                   // inc
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, s8, v[vgprLocalReadAddrB+0] // lrB += 68 ((MT+PAD)*bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
MAC_2x2_X0


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

/* Tail: add MISC Vgpr [4...12) to pool */
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
v_lshrrev_b32 v5, 3, v[vgprSerial]                 // 5 = Serial / 8
v_and_b32 v4, 7, v[vgprSerial]                     // 4 = Serial % 8
                                                   // v4 = v4 * 1 (multiplier is 1, do nothing)
v_mul_lo_u32 v6, v5, s[sgprStrideC1J]              // rowStart vgpr
v_mul_lo_u32 v7, v5, s[sgprStrideD1J]              // rowStart vgpr

s_mul_i32 s30, 0x10, s[sgprWorkGroup0]             // s30 = wg0*MT0
v_add_nc_u32 v4, s30, v4                           // coord0 = tid0*VW + wg0*MT0
s_mul_i32 s32, 0x10, s[sgprWorkGroup1]             // <- wg1*MT1
v_add_nc_u32 v5, s32, v5                           // coord1 = tid1*VW + wg1*MT1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
label_GW_B0_1:
label_GW_B0_FD0_1:
s_and_b32 s30, 15, s[sgprSizeI]                    // s30 = s[sgprSizeI] % 16
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Else_1          // jump if edges required
s_and_b32 s30, 15, s[sgprSizeJ]                    // s30 = s[sgprSizeJ] % 16
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Then_1          // jump if edges required
label_GW_B0_FD0_VW1_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=4 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_lshl_u32 v11, v7, v4, 0x2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=4, coord0Vgpr=4

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v2, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v3, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_NonEdgeEnd_1:
label_GW_B0_FD0_VW1_Else_1:
label_GW_B0_FD0_VW1_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=36 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v10, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s30, v4, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v11, v7, v4, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v11, v10, v11, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v8, vcc_lo, v4, 8                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v12, v7, v8, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v10, v12, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v5, vcc_lo, v5, 8                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v6, v6, s30                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v7, v7, s30                           // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v4, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v13, v7, v4, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v13, v10, v13, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v8, vcc_lo, v4, 8                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v14, v7, v8, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v14, v10, v14, s32                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v12, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v2, v13, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v3, v14, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
s_and_b32 s30, 15, s[sgprSizeI]                    // s30 = s[sgprSizeI] % 16
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Else_2          // jump if edges required
s_and_b32 s30, 15, s[sgprSizeJ]                    // s30 = s[sgprSizeJ] % 16
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Then_2          // jump if edges required
label_GW_B0_FD0_VW1_NonEdge_2:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=4 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_lshl_u32 v11, v7, v4, 0x2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=4, coord0Vgpr=4

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v2, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v3, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW1_NonEdgeEnd_2:
label_GW_B0_FD0_VW1_Else_2:
label_GW_B0_FD0_VW1_Then_2:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=36 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v10, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s30, v4, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v11, v7, v4, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v11, v10, v11, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v8, vcc_lo, v4, 8                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v12, v7, v8, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v10, v12, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v5, vcc_lo, v5, 8                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v6, v6, s30                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v7, v7, s30                           // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v4, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v13, v7, v4, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v13, v10, v13, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v8, vcc_lo, v4, 8                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v14, v7, v8, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v14, v10, v14, s32                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
buffer_store_b32 v0, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v1, v12, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v2, v13, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v3, v14, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_branch label_GW_End_2                            // jump to end
label_GW_B1:
label_GW_B1_FD0:
s_and_b32 s30, 15, s[sgprSizeI]                    // s30 = s[sgprSizeI] % 16
s_add_u32 s31, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s31                // wg0 >= nwg0-1 ?
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_Else            // jump if edges required
s_and_b32 s30, 15, s[sgprSizeJ]                    // s30 = s[sgprSizeJ] % 16
s_add_u32 s31, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s31                // wg1 >= nwg1-1
s_cselect_b32 s30, s30, 0                          // set rem
s_cmpk_gt_u32 s30, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_Then            // jump if edges required
label_GW_B1_FD0_VW1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=34 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v12, v6, v4, 0x2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=4, coord0Vgpr=4
buffer_load_b32 v13, v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_b32 v14, v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s8, s[sgprStrideC1J], 32                 // scale StrideC *= numRows(8) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_b32 v15, v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_b32 v16, v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
v_add_lshl_u32 v11, v7, v4, 0x2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=4, coord0Vgpr=4

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 4 - 1 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+0], v13, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v0, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 4 - 2 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+1], v14, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v1, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 4 - 3 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+2], v15, s[sgprBeta]        // finalSum = sum*alpha + C*beta
s_mul_i32 s8, s[sgprStrideD1J], 32                 // scale StrideD *= numRows(8) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b32 v2, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 4 - 4 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+3], v16, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v3, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW1_NonEdgeEnd:
label_GW_B1_FD0_VW1_Else:
label_GW_B1_FD0_VW1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=17 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v10, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s30, v4, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v12, v6, v4, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v10, v12, s32                   // LDC clip if OOB. offset
buffer_load_b32 v11, v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v12, v7, v4, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v10, v12, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v8, vcc_lo, v4, 8                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v14, v6, v8, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v14, v10, v14, s32                   // LDC clip if OOB. offset
buffer_load_b32 v13, v14, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v14, v7, v8, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v14, v10, v14, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v5, vcc_lo, v5, 8                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s30, s[sgprStrideC1J], 8                 // scale stride
v_add_nc_i32 v6, v6, s30                           // ROWINC- Move cinRowPtr to next row
s_mul_i32 s30, s[sgprStrideD1J], 8                 // scale stride
v_add_nc_i32 v7, v7, s30                           // Move coutRowPtrD to next row
v_cmp_lt_u32 s30, v4, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v16, v6, v4, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v16, v10, v16, s32                   // LDC clip if OOB. offset
buffer_load_b32 v15, v16, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v16, v7, v4, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v16, v10, v16, s32                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v8, vcc_lo, v4, 8                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s30, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s32, v5, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s32, s30, s32                            // in0 && in1
v_add_lshl_u32 v18, v6, v8, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v18, v10, v18, s32                   // LDC clip if OOB. offset
buffer_load_b32 v17, v18, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v18, v7, v8, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v18, v10, v18, s32                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fmac_f32 v[vgprValuC+0], v11, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v0, v12, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+1], v13, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v1, v14, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+2], v15, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v2, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fmac_f32 v[vgprValuC+3], v17, s[sgprBeta]        // finalSum = sum*alpha + C*beta
buffer_store_b32 v3, v18, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_branch label_GW_End_2                            // jump to end
label_GW_End_2:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
