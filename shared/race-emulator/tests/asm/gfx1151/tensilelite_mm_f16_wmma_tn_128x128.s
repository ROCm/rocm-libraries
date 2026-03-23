
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x128x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1
.globl Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x128x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1
.p2align 8
.type Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x128x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x128x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 64 // sgprs
  .amdhsa_group_segment_fixed_size 10240 // lds bytes
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
/* Num VGPR   =256 */
/* Num AccVGPR=0 */
/* Num SGPR   =64 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 32 x 4 */
/* SubGroup= 4 x 32 */
/* VectorWidthA=4 */
/* VectorWidthB=4 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8 */
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
  - .name: Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x128x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1
    .symbol: 'Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x128x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1.kd'
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
    .group_segment_fixed_size:   10240
    .kernarg_segment_align:      8
    .kernarg_segment_size:       104
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 64
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_HHS_BH_UserArgs_MT128x128x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1:
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
/* ValuC range: [0-128), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 136
.set vgprLocalWriteAddrA, 130
.set vgprLocalWriteAddrB, 131
.set vgprGlobalReadOffsetA, 128
.set vgprGlobalReadOffsetB, 129
.set vgprLocalReadAddrA, 132
.set vgprLocalReadAddrB, 133
.set vgprSerial, 201

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+33
.set vgprG2LA_BASE, vgprBase+0
.set vgprG2LB_BASE, vgprBase+33
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
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
.set sgprNumWorkGroups0, 14
.set sgprNumWorkGroups1, 15
.set sgprSizesFree, 16
.set sgprSizesSum, 19
.set sgprAddressD, 20
.set sgprAddressC, 22
.set sgprAddressA, 24
.set sgprAddressB, 26
.set sgprStridesD, 28
.set sgprStridesC, 30
.set sgprStridesA, 32
.set sgprStridesB, 34
.set sgprAlpha, 36
.set sgprBeta, 37
.set sgprGSU, 38

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
.set DepthU, 16
.set BpeA, 2
.set BpeALog2, 1
.set BpeB, 2
.set BpeBLog2, 1
.set BpeAGR, 2
.set BpeAGRLog2, 1
.set BpeBGR, 2
.set BpeBGRLog2, 1
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
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
.macro GLOBAL_OFFSET_A vgprAddr:req, vgprOffsetL:req, vgprOffset0I:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideA0I], v[\vgprOffset0I] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]   // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 1, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]   // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 1, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_b32 s39, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s41, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s42, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s40, s39, 0x1e                          // Get arg type
s_and_b32 s39, 0x3fffffff, s39                     // Get nums of gemm
s_cmp_eq_u32 s40, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[16:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_b128 s[32:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_b64 s[36:37], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_and_b32 s[sgprStaggerU], s41, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s41, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s40
s_mov_b32 m0, 0x2800                               // LDS clamp at 10240 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s48, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s48, s48                             // Get log(WGMXCC)
s_lshr_b32 s49, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s48, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s45, s42, s48
s_lshl_b32 s45, s45, s48
s_cmp_ge_u32 s[sgprWorkGroup0], s45
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s49, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s45, s[sgprWorkGroup0], s48
s_bfm_b32 s46, s48, 0
s_and_b32 s46, s[sgprWorkGroup0], s46
s_lshr_b32 s47, s42, s48
s_mul_i32 s46, s46, s47
s_add_u32 s[sgprWorkGroup0], s45, s46
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[6:7], s49                          // s45 = s[sgprWorkGroup0] / s49
v_rcp_f64 v[6:7], v[6:7]                           // s45 = s[sgprWorkGroup0] / s49
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s45 = s[sgprWorkGroup0] / s49
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s45 = s[sgprWorkGroup0] / s49
v_cvt_u32_f64 v6, v[6:7]                           // s45 = s[sgprWorkGroup0] / s49
v_mul_lo_u32 v7, v6, s49                           // s45 = s[sgprWorkGroup0] / s49
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s45 = s[sgprWorkGroup0] / s49
v_cmp_ge_u32 vcc_lo, v8, s49                       // s45 = s[sgprWorkGroup0] / s49
s_mov_b32 exec_lo vcc_lo                           // s45 = s[sgprWorkGroup0] / s49
v_add_nc_u32 v6, v6, 1                             // s45 = s[sgprWorkGroup0] / s49
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s49                           // s45 = s[sgprWorkGroup0] / s49
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s45 = s[sgprWorkGroup0] / s49
v_readfirstlane_b32 s45, v6                        // quotient
v_readfirstlane_b32 s46, v8                        // remainder
s_mul_i32 s45, s45, s49
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s46, s46, s48
/* temp0 = temp0 + temp1 */
s_add_u32 s45, s45, s46
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[6:7], s49                          // s46 = s42 / s49
v_rcp_f64 v[6:7], v[6:7]                           // s46 = s42 / s49
v_cvt_f64_u32 v[8:9], s42                          // s46 = s42 / s49
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s46 = s42 / s49
v_cvt_u32_f64 v6, v[6:7]                           // s46 = s42 / s49
v_mul_lo_u32 v7, v6, s49                           // s46 = s42 / s49
v_sub_nc_u32 v8, s42, v7                           // s46 = s42 / s49
v_cmp_ge_u32 vcc_lo, v8, s49                       // s46 = s42 / s49
s_mov_b32 exec_lo vcc_lo                           // s46 = s42 / s49
v_add_nc_u32 v6, v6, 1                             // s46 = s42 / s49
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s46, v6                        // quotient
s_mul_i32 s46, s46, s49
s_sub_u32 s47, s42, s46
s_cmp_gt_u32 s[sgprWorkGroup0], s46
s_cselect_b32 s46, s47, s49
s_lshr_b32 s46, s46, s48
s_bfm_b32 s47, s48, 0
s_and_b32 s47, s[sgprWorkGroup0], s47
s_mul_i32 s46, s46, s47
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s45, s46
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s40, 0
s_cbranch_scc0 label_MultiGemm
/* init: add vgpr [136...337) to pool */
/* init: add vgpr [0...128) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 4, v0                            // 1. N offset: nOffset = nIdx * nStride(16)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v0, 2, v0                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 10, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 4, v1                            // 1. N offset: nOffset = nIdx * nStride(16)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v1, 2, v1                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 10, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s41, 16                                  // LSU offset: stride = lsuStride(16) when umlds==True
v_mul_lo_u32 v2, s41, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v2, v0, 0x1  // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(16) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s41, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 128

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x1400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 2 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 1, v[vgprSerial]                 // 0 = Serial / 2
v_and_b32 v1, 1, v[vgprSerial]                     // 1 = Serial % 2
/* unroll *= glvw */
v_lshlrev_b32 v1, 3, v1                            // v1 = v1 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 2 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 1, v[vgprSerial]                 // 2 = Serial / 2
v_and_b32 v3, 1, v[vgprSerial]                     // 3 = Serial % 2
/* unroll *= glvw */
v_lshlrev_b32 v3, 3, v3                            // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v0     // lwAL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrA]        // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 5, v[vgprLocalWriteAddrA] // padding 32 per block 128

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2     // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrB]        // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 5, v[vgprLocalWriteAddrB] // padding 32 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x1400, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=5120
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
s_mul_i32 s40, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s41, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s40, s40, s41
v_cvt_f32_u32 v6, s40                              // s40 = s[sgprWorkGroup0] / s40
v_rcp_iflag_f32 v6, v6                             // s40 = s[sgprWorkGroup0] / s40
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s40 = s[sgprWorkGroup0] / s40
v_mul_f32 v6, v6, v7                               // s40 = s[sgprWorkGroup0] / s40
v_cvt_u32_f32 v6, v6                               // s40 = s[sgprWorkGroup0] / s40
v_mul_u32_u24 v7, v6, s40                          // s40 = s[sgprWorkGroup0] / s40
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s40 = s[sgprWorkGroup0] / s40
v_cmp_eq_u32 vcc_lo, v7, s40                       // s40 = s[sgprWorkGroup0] / s40
s_mov_b32 exec_lo vcc_lo                           // s40 = s[sgprWorkGroup0] / s40
v_add_nc_u32 v6, 1, v6                             // s40 = s[sgprWorkGroup0] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s40                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s40, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s40
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s40, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s40, s40, s[sgprWorkGroup2]
s_mul_i32 s40, s40, s41
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s40
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo vcc_lo                           // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v6, 1, v6                             // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s40, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s40
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s40, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s40
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 88
s_mul_i32 s46, s39, 4
s_mov_b64 s[40:41], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 196
s_mov_b32 s46, 0
s_mov_b64 s[40:41], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s47, 0
s_load_b128 s[16:19], s[40:41], s46
s_cmpk_eq_u32 s39, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s44, s16, 7                             // s44 = s16 / 128
s_and_b32 s42, 127, s16                            // s42 = s16 % 128
s_addc_u32 s44, s44, 0
s_lshr_b32 s45, s17, 7                             // s45 = s17 / 128
s_and_b32 s42, 127, s17                            // s42 = s17 % 128
s_addc_u32 s45, s45, 0
s_mul_i32 s44, s44, s45
s_mul_i32 s44, s44, s18
s_and_b32 s45, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s44, s44, s45
s_add_u32 s47, s47, s44
s_cmp_lt_u32 s[sgprWorkGroup0], s47
s_cbranch_scc1 label_FOUND
s_add_u32 s46, s46, s15
s_load_b128 s[16:19], s[40:41], s46
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s39
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s44, s16, 7                             // s44 = s16 / 128
s_and_b32 s42, 127, s16                            // s42 = s16 % 128
s_addc_u32 s44, s44, 0
s_lshr_b32 s45, s17, 7                             // s45 = s17 / 128
s_and_b32 s42, 127, s17                            // s42 = s17 % 128
s_addc_u32 s45, s45, 0
s_mul_i32 s44, s44, s45
s_mul_i32 s44, s44, s18
s_and_b32 s40, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s44, s44, s40
s_add_u32 s47, s47, s44

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s41, s14, 1
s_sub_u32 s40, s47, s44
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s40
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s39, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s41, s41, 88
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s41
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[36:37], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s41, s41, 196
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s41
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b512 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b32 s36, s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
// Read Beta
s_load_b32 s37, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
label_LoadExternalStructEnd:
/* init: add vgpr [136...337) to pool */
/* init: add vgpr [0...128) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 4, v0                            // 1. N offset: nOffset = nIdx * nStride(16)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v0, 2, v0                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 10, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 4, v1                            // 1. N offset: nOffset = nIdx * nStride(16)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v1, 2, v1                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 10, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s41, 16                                  // LSU offset: stride = lsuStride(16) when umlds==True
v_mul_lo_u32 v2, s41, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v2, v0, 0x1  // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(16) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s41, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 128

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x1400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 2 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 1, v[vgprSerial]                 // 0 = Serial / 2
v_and_b32 v1, 1, v[vgprSerial]                     // 1 = Serial % 2
/* unroll *= glvw */
v_lshlrev_b32 v1, 3, v1                            // v1 = v1 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 2 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 1, v[vgprSerial]                 // 2 = Serial / 2
v_and_b32 v3, 1, v[vgprSerial]                     // 3 = Serial % 2
/* unroll *= glvw */
v_lshlrev_b32 v3, 3, v3                            // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v0     // lwAL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrA]        // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 5, v[vgprLocalWriteAddrA] // padding 32 per block 128

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2     // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrB]        // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 5, v[vgprLocalWriteAddrB] // padding 32 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x1400, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=5120
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
s_mul_i32 s40, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s41, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s40, s40, s41
v_cvt_f32_u32 v6, s40                              // s40 = s[sgprWorkGroup0] / s40
v_rcp_iflag_f32 v6, v6                             // s40 = s[sgprWorkGroup0] / s40
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s40 = s[sgprWorkGroup0] / s40
v_mul_f32 v6, v6, v7                               // s40 = s[sgprWorkGroup0] / s40
v_cvt_u32_f32 v6, v6                               // s40 = s[sgprWorkGroup0] / s40
v_mul_u32_u24 v7, v6, s40                          // s40 = s[sgprWorkGroup0] / s40
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s40 = s[sgprWorkGroup0] / s40
v_cmp_eq_u32 vcc_lo, v7, s40                       // s40 = s[sgprWorkGroup0] / s40
s_mov_b32 exec_lo vcc_lo                           // s40 = s[sgprWorkGroup0] / s40
v_add_nc_u32 v6, 1, v6                             // s40 = s[sgprWorkGroup0] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s40                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s40, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s40
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s40, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s40, s40, s[sgprWorkGroup2]
s_mul_i32 s40, s40, s41
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s40
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo vcc_lo                           // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v6, 1, v6                             // s40 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s40, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s40
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s40, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s40

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprSrdA, 40
.set sgprSrdB, 44
.set sgprShadowLimitA, 48
.set sgprShadowLimitB, 50
.set sgprStaggerUIter, 39
.set sgprWrapUA, 52
.set sgprWrapUB, 54
.set sgprGlobalReadIncsA, 56
.set sgprGlobalReadIncsB, 57
.set sgprScalarGlobalReadOffsetA, 58
.set sgprScalarGlobalReadOffsetB, 59
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
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
s_and_b32 s40, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s40, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s40, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s40, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v6, s40                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_mul_u32_u24 v7, v6, s40                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_cmp_eq_u32 vcc_lo, v7, s40                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_add_nc_u32 v6, 1, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s40
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s40                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s40                          // re-calculate remainder
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
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 1
label_GSU_End:
/* WGM Calculation */
s_mov_b32 s40, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s40, s40                            // Restore WGM
s_cmp_gt_i32 s40, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s40, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s40, s40                                 // abs(WGM)
v_cvt_f64_u32 v[6:7], s40                          // s41 = s[sgprWorkGroup0] / s40
v_rcp_f64 v[6:7], v[6:7]                           // s41 = s[sgprWorkGroup0] / s40
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s41 = s[sgprWorkGroup0] / s40
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s41 = s[sgprWorkGroup0] / s40
v_cvt_u32_f64 v6, v[6:7]                           // s41 = s[sgprWorkGroup0] / s40
v_mul_lo_u32 v7, v6, s40                           // s41 = s[sgprWorkGroup0] / s40
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s41 = s[sgprWorkGroup0] / s40
v_cmp_ge_u32 vcc_lo, v8, s40                       // s41 = s[sgprWorkGroup0] / s40
s_mov_b32 exec_lo vcc_lo                           // s41 = s[sgprWorkGroup0] / s40
v_add_nc_u32 v6, v6, 1                             // s41 = s[sgprWorkGroup0] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s41, v6                        // quotient
s_mul_i32 s44, s41, s40                            // quotient * non-magic divisor
s_sub_u32 s44, s[sgprWorkGroup0], s44              // WorkGroup0=remainder
s_mul_i32 s44, s44, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s44, s44, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[6:7], s40                          // s42 = s[sgprNumWorkGroups0] / s40
v_rcp_f64 v[6:7], v[6:7]                           // s42 = s[sgprNumWorkGroups0] / s40
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups0]        // s42 = s[sgprNumWorkGroups0] / s40
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s42 = s[sgprNumWorkGroups0] / s40
v_cvt_u32_f64 v6, v[6:7]                           // s42 = s[sgprNumWorkGroups0] / s40
v_mul_lo_u32 v7, v6, s40                           // s42 = s[sgprNumWorkGroups0] / s40
v_sub_nc_u32 v8, s[sgprNumWorkGroups0], v7         // s42 = s[sgprNumWorkGroups0] / s40
v_cmp_ge_u32 vcc_lo, v8, s40                       // s42 = s[sgprNumWorkGroups0] / s40
s_mov_b32 exec_lo vcc_lo                           // s42 = s[sgprNumWorkGroups0] / s40
v_add_nc_u32 v6, v6, 1                             // s42 = s[sgprNumWorkGroups0] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s42, v6                        // quotient
s_mul_i32 s43, s40, s42                            // quotient * non-magic divisor
s_sub_u32 s43, s[sgprNumWorkGroups0], s43          // NumWorkGroups0=remainder
s_cmp_eq_u32 s43, 0                                // remainder == 0 ?
s_cmov_b32 s43, s40                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s41, s42                              // blockId >= numFullBlocks ?
s_cselect_b32 s42, s43, s40
v_cvt_f64_u32 v[6:7], s42                          // s[sgprWorkGroup1] = s44 / s42
v_rcp_f64 v[6:7], v[6:7]                           // s[sgprWorkGroup1] = s44 / s42
v_cvt_f64_u32 v[8:9], s44                          // s[sgprWorkGroup1] = s44 / s42
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s[sgprWorkGroup1] = s44 / s42
v_cvt_u32_f64 v6, v[6:7]                           // s[sgprWorkGroup1] = s44 / s42
v_mul_lo_u32 v7, v6, s42                           // s[sgprWorkGroup1] = s44 / s42
v_sub_nc_u32 v8, s44, v7                           // s[sgprWorkGroup1] = s44 / s42
v_cmp_ge_u32 vcc_lo, v8, s42                       // s[sgprWorkGroup1] = s44 / s42
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s44 / s42
v_add_nc_u32 v6, v6, 1                             // s[sgprWorkGroup1] = s44 / s42
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s42                           // s[sgprWorkGroup1] = s44 / s42
v_sub_nc_u32 v8, s44, v7                           // s[sgprWorkGroup1] = s44 / s42
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v8          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s42 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s44, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s41, s41, s40                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s41 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s40, s40                                 // WGM
v_cvt_f64_u32 v[6:7], s40                          // s41 = s[sgprWorkGroup1] / s40
v_rcp_f64 v[6:7], v[6:7]                           // s41 = s[sgprWorkGroup1] / s40
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup1]            // s41 = s[sgprWorkGroup1] / s40
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s41 = s[sgprWorkGroup1] / s40
v_cvt_u32_f64 v6, v[6:7]                           // s41 = s[sgprWorkGroup1] / s40
v_mul_lo_u32 v7, v6, s40                           // s41 = s[sgprWorkGroup1] / s40
v_sub_nc_u32 v8, s[sgprWorkGroup1], v7             // s41 = s[sgprWorkGroup1] / s40
v_cmp_ge_u32 vcc_lo, v8, s40                       // s41 = s[sgprWorkGroup1] / s40
s_mov_b32 exec_lo vcc_lo                           // s41 = s[sgprWorkGroup1] / s40
v_add_nc_u32 v6, v6, 1                             // s41 = s[sgprWorkGroup1] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s41, v6                        // quotient
s_mul_i32 s44, s41, s40                            // quotient * non-magic divisor
s_sub_u32 s44, s[sgprWorkGroup1], s44              // WorkGroup1=remainder
s_mul_i32 s44, s44, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s44, s44, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[6:7], s40                          // s42 = s[sgprNumWorkGroups1] / s40
v_rcp_f64 v[6:7], v[6:7]                           // s42 = s[sgprNumWorkGroups1] / s40
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups1]        // s42 = s[sgprNumWorkGroups1] / s40
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s42 = s[sgprNumWorkGroups1] / s40
v_cvt_u32_f64 v6, v[6:7]                           // s42 = s[sgprNumWorkGroups1] / s40
v_mul_lo_u32 v7, v6, s40                           // s42 = s[sgprNumWorkGroups1] / s40
v_sub_nc_u32 v8, s[sgprNumWorkGroups1], v7         // s42 = s[sgprNumWorkGroups1] / s40
v_cmp_ge_u32 vcc_lo, v8, s40                       // s42 = s[sgprNumWorkGroups1] / s40
s_mov_b32 exec_lo vcc_lo                           // s42 = s[sgprNumWorkGroups1] / s40
v_add_nc_u32 v6, v6, 1                             // s42 = s[sgprNumWorkGroups1] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s42, v6                        // quotient
s_mul_i32 s43, s40, s42                            // quotient * non-magic divisor
s_sub_u32 s43, s[sgprNumWorkGroups1], s43          // NumWorkGroups1=remainder
s_cmp_eq_u32 s43, 0                                // remainder == 0 ?
s_cmov_b32 s43, s40                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s41, s42                              // blockId >= numFullBlocks ?
s_cselect_b32 s42, s43, s40
v_cvt_f64_u32 v[6:7], s42                          // s[sgprWorkGroup0] = s44 / s42
v_rcp_f64 v[6:7], v[6:7]                           // s[sgprWorkGroup0] = s44 / s42
v_cvt_f64_u32 v[8:9], s44                          // s[sgprWorkGroup0] = s44 / s42
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s[sgprWorkGroup0] = s44 / s42
v_cvt_u32_f64 v6, v[6:7]                           // s[sgprWorkGroup0] = s44 / s42
v_mul_lo_u32 v7, v6, s42                           // s[sgprWorkGroup0] = s44 / s42
v_sub_nc_u32 v8, s44, v7                           // s[sgprWorkGroup0] = s44 / s42
v_cmp_ge_u32 vcc_lo, v8, s42                       // s[sgprWorkGroup0] = s44 / s42
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup0] = s44 / s42
v_add_nc_u32 v6, v6, 1                             // s[sgprWorkGroup0] = s44 / s42
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s42                           // s[sgprWorkGroup0] = s44 / s42
v_sub_nc_u32 v8, s44, v7                           // s[sgprWorkGroup0] = s44 / s42
v_readfirstlane_b32 s[sgprWorkGroup0], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v8          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s42 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s44, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s41, s41, s40                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s41 // wg1 += blockId * WGM
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
s_mul_hi_u32 s63, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s62, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s63, s62, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s62, s62, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_and_b32 s60, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s61, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s60, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
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
s_mul_i32 s61, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s60, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s61, s61, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s60, s60, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s60, s60, s61                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s61, s60, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s60, s60, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_add_u32 s62, s62, s60                            // accum GsuOffset term to tilestart
s_addc_u32 s63, s63, s61                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s60, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s61, constStrideAL, s60               // stride x (size-1)
s_mul_i32 s60, constStrideAL, s60                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // sum tensor size
s_sub_u32 s60, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s61, s[sgprStrideA0I], s60            // stride x (size-1)
s_mul_i32 s60, s[sgprStrideA0I], s60               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s62 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s63 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s61, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s60, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s62, s62, s60                            // accum wg term to tilestart
s_addc_u32 s63, s63, s61                           // accum wg term to tilestart
s_lshl_b64 s[62:63], s[62:63], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s62    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s63   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s63, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s62, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s63, s62, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s62, s62, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s60, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s61, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s60, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
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
s_mul_i32 s61, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s60, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s61, s61, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s60, s60, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s60, s60, s61                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s61, s60, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s60, s60, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_B_End:
s_add_u32 s62, s62, s60                            // accum GsuOffset term to tilestart
s_addc_u32 s63, s63, s61                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s60, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s61, constStrideBL, s60               // stride x (size-1)
s_mul_i32 s60, constStrideBL, s60                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // sum tensor size
s_sub_u32 s60, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s61, s[sgprStrideB1J], s60            // stride x (size-1)
s_mul_i32 s60, s[sgprStrideB1J], s60               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s62 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s63 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s61, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s60, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s62, s62, s60                            // accum wg term to tilestart
s_addc_u32 s63, s63, s61                           // accum wg term to tilestart
s_lshl_b64 s[62:63], s[62:63], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s62    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s63   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  1,  0, 6 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideA0I], 64 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+0], s[sgprScalarGlobalReadOffsetA+0], 0x1 // scalar offset *= bytes/element
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 6 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideB1J], 64 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], 0x1 // scalar offset *= bytes/element
/* ============================================================= */

/* global read addresses: increments a */
s_and_b32 s61, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s61, s61, DepthU*BpeAGR                  // GSU*DepthU*Bpe
s_and_b32 s60, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeAGR, s61 // incrA (unrollIdx)

/* global read addresses: increments b */
s_and_b32 s61, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s61, s61, DepthU*BpeBGR                  // GSU*DepthU*Bpe
s_and_b32 s60, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeBGR, s61 // incrB (unrollIdx)
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...128) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [136...201) from pool */
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
v_mov_b32 v[vgprValuC+16], 0                       // initC
v_mov_b32 v[vgprValuC+17], 0                       // initC
v_mov_b32 v[vgprValuC+18], 0                       // initC
v_mov_b32 v[vgprValuC+19], 0                       // initC
v_mov_b32 v[vgprValuC+20], 0                       // initC
v_mov_b32 v[vgprValuC+21], 0                       // initC
v_mov_b32 v[vgprValuC+22], 0                       // initC
v_mov_b32 v[vgprValuC+23], 0                       // initC
v_mov_b32 v[vgprValuC+24], 0                       // initC
v_mov_b32 v[vgprValuC+25], 0                       // initC
v_mov_b32 v[vgprValuC+26], 0                       // initC
v_mov_b32 v[vgprValuC+27], 0                       // initC
v_mov_b32 v[vgprValuC+28], 0                       // initC
v_mov_b32 v[vgprValuC+29], 0                       // initC
v_mov_b32 v[vgprValuC+30], 0                       // initC
v_mov_b32 v[vgprValuC+31], 0                       // initC
v_mov_b32 v[vgprValuC+32], 0                       // initC
v_mov_b32 v[vgprValuC+33], 0                       // initC
v_mov_b32 v[vgprValuC+34], 0                       // initC
v_mov_b32 v[vgprValuC+35], 0                       // initC
v_mov_b32 v[vgprValuC+36], 0                       // initC
v_mov_b32 v[vgprValuC+37], 0                       // initC
v_mov_b32 v[vgprValuC+38], 0                       // initC
v_mov_b32 v[vgprValuC+39], 0                       // initC
v_mov_b32 v[vgprValuC+40], 0                       // initC
v_mov_b32 v[vgprValuC+41], 0                       // initC
v_mov_b32 v[vgprValuC+42], 0                       // initC
v_mov_b32 v[vgprValuC+43], 0                       // initC
v_mov_b32 v[vgprValuC+44], 0                       // initC
v_mov_b32 v[vgprValuC+45], 0                       // initC
v_mov_b32 v[vgprValuC+46], 0                       // initC
v_mov_b32 v[vgprValuC+47], 0                       // initC
v_mov_b32 v[vgprValuC+48], 0                       // initC
v_mov_b32 v[vgprValuC+49], 0                       // initC
v_mov_b32 v[vgprValuC+50], 0                       // initC
v_mov_b32 v[vgprValuC+51], 0                       // initC
v_mov_b32 v[vgprValuC+52], 0                       // initC
v_mov_b32 v[vgprValuC+53], 0                       // initC
v_mov_b32 v[vgprValuC+54], 0                       // initC
v_mov_b32 v[vgprValuC+55], 0                       // initC
v_mov_b32 v[vgprValuC+56], 0                       // initC
v_mov_b32 v[vgprValuC+57], 0                       // initC
v_mov_b32 v[vgprValuC+58], 0                       // initC
v_mov_b32 v[vgprValuC+59], 0                       // initC
v_mov_b32 v[vgprValuC+60], 0                       // initC
v_mov_b32 v[vgprValuC+61], 0                       // initC
v_mov_b32 v[vgprValuC+62], 0                       // initC
v_mov_b32 v[vgprValuC+63], 0                       // initC
v_mov_b32 v[vgprValuC+64], 0                       // initC
v_mov_b32 v[vgprValuC+65], 0                       // initC
v_mov_b32 v[vgprValuC+66], 0                       // initC
v_mov_b32 v[vgprValuC+67], 0                       // initC
v_mov_b32 v[vgprValuC+68], 0                       // initC
v_mov_b32 v[vgprValuC+69], 0                       // initC
v_mov_b32 v[vgprValuC+70], 0                       // initC
v_mov_b32 v[vgprValuC+71], 0                       // initC
v_mov_b32 v[vgprValuC+72], 0                       // initC
v_mov_b32 v[vgprValuC+73], 0                       // initC
v_mov_b32 v[vgprValuC+74], 0                       // initC
v_mov_b32 v[vgprValuC+75], 0                       // initC
v_mov_b32 v[vgprValuC+76], 0                       // initC
v_mov_b32 v[vgprValuC+77], 0                       // initC
v_mov_b32 v[vgprValuC+78], 0                       // initC
v_mov_b32 v[vgprValuC+79], 0                       // initC
v_mov_b32 v[vgprValuC+80], 0                       // initC
v_mov_b32 v[vgprValuC+81], 0                       // initC
v_mov_b32 v[vgprValuC+82], 0                       // initC
v_mov_b32 v[vgprValuC+83], 0                       // initC
v_mov_b32 v[vgprValuC+84], 0                       // initC
v_mov_b32 v[vgprValuC+85], 0                       // initC
v_mov_b32 v[vgprValuC+86], 0                       // initC
v_mov_b32 v[vgprValuC+87], 0                       // initC
v_mov_b32 v[vgprValuC+88], 0                       // initC
v_mov_b32 v[vgprValuC+89], 0                       // initC
v_mov_b32 v[vgprValuC+90], 0                       // initC
v_mov_b32 v[vgprValuC+91], 0                       // initC
v_mov_b32 v[vgprValuC+92], 0                       // initC
v_mov_b32 v[vgprValuC+93], 0                       // initC
v_mov_b32 v[vgprValuC+94], 0                       // initC
v_mov_b32 v[vgprValuC+95], 0                       // initC
v_mov_b32 v[vgprValuC+96], 0                       // initC
v_mov_b32 v[vgprValuC+97], 0                       // initC
v_mov_b32 v[vgprValuC+98], 0                       // initC
v_mov_b32 v[vgprValuC+99], 0                       // initC
v_mov_b32 v[vgprValuC+100], 0                      // initC
v_mov_b32 v[vgprValuC+101], 0                      // initC
v_mov_b32 v[vgprValuC+102], 0                      // initC
v_mov_b32 v[vgprValuC+103], 0                      // initC
v_mov_b32 v[vgprValuC+104], 0                      // initC
v_mov_b32 v[vgprValuC+105], 0                      // initC
v_mov_b32 v[vgprValuC+106], 0                      // initC
v_mov_b32 v[vgprValuC+107], 0                      // initC
v_mov_b32 v[vgprValuC+108], 0                      // initC
v_mov_b32 v[vgprValuC+109], 0                      // initC
v_mov_b32 v[vgprValuC+110], 0                      // initC
v_mov_b32 v[vgprValuC+111], 0                      // initC
v_mov_b32 v[vgprValuC+112], 0                      // initC
v_mov_b32 v[vgprValuC+113], 0                      // initC
v_mov_b32 v[vgprValuC+114], 0                      // initC
v_mov_b32 v[vgprValuC+115], 0                      // initC
v_mov_b32 v[vgprValuC+116], 0                      // initC
v_mov_b32 v[vgprValuC+117], 0                      // initC
v_mov_b32 v[vgprValuC+118], 0                      // initC
v_mov_b32 v[vgprValuC+119], 0                      // initC
v_mov_b32 v[vgprValuC+120], 0                      // initC
v_mov_b32 v[vgprValuC+121], 0                      // initC
v_mov_b32 v[vgprValuC+122], 0                      // initC
v_mov_b32 v[vgprValuC+123], 0                      // initC
v_mov_b32 v[vgprValuC+124], 0                      // initC
v_mov_b32 v[vgprValuC+125], 0                      // initC
v_mov_b32 v[vgprValuC+126], 0                      // initC
v_mov_b32 v[vgprValuC+127], 0                      // initC
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_and_b32 s60, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s60, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v202, s[sgprGSUSumIdx+1]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v202, v202                         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v203, s[sgprLoopCounterL]            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v202, v202, v203                         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v202, v202                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v203, v202, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v203, s[sgprLoopCounterL], v203       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v203, s[sgprGSUSumIdx+1]      // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v202, 1, v202                         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v203, 0                                  // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v203, s[sgprGSUSumIdx+1]      // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v202, v202, 1                         // quotient - 1
v_mul_u32_u24 v203, v202, s[sgprGSUSumIdx+1]       // re-calculate remainder
v_sub_nc_u32 v203, s[sgprLoopCounterL], v203       // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v202      // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v203       // remainder
s_add_u32 s60, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s60                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s62, s[sgprStaggerU], 0x1f00
s_lshr_b32 s62, s62, 0x8
s_and_b32 s63, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s60, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s61, s60, s62                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s61           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s60, s60, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s61, s60, 1                              // staggerU mask
s_cmp_ge_u32 s60, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s61, 0          // set Mask
s_cmp_eq_u32 s63, 0x0
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s60, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s63, 0x2000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s60, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s63, 0x4000
s_cbranch_scc0 label_StaggerUMapping_3
s_mov_b32 s60, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s63, 0x6000
s_cbranch_scc0 label_StaggerUMapping_4
s_mul_i32 s61, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s60, s60, s61
s_add_u32 s60, s60, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s63, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s60, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s60 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s62 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s61, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s60, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s61, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s60, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 1 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

/* localReadInitPointers */

/******************************************/
/* End setupNewTile                       */
/******************************************/

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */
buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s60, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s61, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s60, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s61, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt vmcnt(0)                                 // 5wait for global read
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier                                          // PGR=0, prior iter done reading lds

/* local write a */
ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:2560 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2560

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:2560 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2560
s_waitcnt lgkmcnt(0)                               // 2prefetch wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:15, lwEndMfmaIndex:15  */
/*  numMfmaForLR:15, syncPlrMfmaIndex:0  */
/*  mfmaIndex:0  */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:48 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:80 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:112 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:48 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(8)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[0:7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[0:7] // left value = v[0+0:7+0]
/*  mfmaIndex:1  */
s_waitcnt lgkmcnt(6)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[8:15], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[8:15] // left value = v[8+0:15+0]
/*  mfmaIndex:2  */
s_waitcnt lgkmcnt(4)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[16:23], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[16:23] // left value = v[16+0:23+0]
/*  mfmaIndex:3  */
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(3)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[24:31], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[24:31] // left value = v[24+0:31+0]
/*  mfmaIndex:4  */
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:80 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[32:39], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[32:39] // left value = v[32+0:39+0]
/*  mfmaIndex:5  */
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
v_wmma_f32_16x16x16_f16 v[40:47], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[40:47] // left value = v[40+0:47+0]
/*  mfmaIndex:6  */
v_wmma_f32_16x16x16_f16 v[48:55], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[48:55] // left value = v[48+0:55+0]
/*  mfmaIndex:7  */
v_wmma_f32_16x16x16_f16 v[56:63], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[56:63] // left value = v[56+0:63+0]
/*  mfmaIndex:8  */
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:112 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
s_waitcnt lgkmcnt(2)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[64:71], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[64:71] // left value = v[64+0:71+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[72:79], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[72:79] // left value = v[72+0:79+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[80:87], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[80:87] // left value = v[80+0:87+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[88:95], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[88:95] // left value = v[88+0:95+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_f16 v[96:103], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[96:103] // left value = v[96+0:103+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[104:111], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[104:111] // left value = v[104+0:111+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 5 */
v_wmma_f32_16x16x16_f16 v[112:119], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[112:119] // left value = v[112+0:119+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_wmma_f32_16x16x16_f16 v[120:127], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[120:127] // left value = v[120+0:127+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=8 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/* Tail: add ValuA/B vgpr buffer [136...201) to pool */

/* Tail: add address/G2L vgpr [201...201) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (8,8,0) */
.set vgprG2LA_BASE, 136
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 144
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_and_b32 s60, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s61, s[sgprSizesSum], 4                 // s61 = s[sgprSizesSum] / 16
s_and_b32 s62, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v152, s62                            // s60 = s61 / s62
v_rcp_iflag_f32 v152, v152                         // s60 = s61 / s62
v_cvt_f32_u32 v153, s61                            // s60 = s61 / s62
v_mul_f32 v152, v152, v153                         // s60 = s61 / s62
v_cvt_u32_f32 v152, v152                           // s60 = s61 / s62
v_mul_u32_u24 v153, v152, s62                      // s60 = s61 / s62
v_sub_nc_u32 v153, s61, v153                       // s60 = s61 / s62
v_cmp_eq_u32 vcc_lo, v153, s62                     // s60 = s61 / s62
s_mov_b32 exec_lo vcc_lo                           // s60 = s61 / s62
v_add_nc_u32 v152, 1, v152                         // s60 = s61 / s62
v_mov_b32 v153, 0                                  // s[sgprGSUSumIdx+1] = s61 % s62
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v153, s62                     // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v152, v152, 1                         // quotient - 1
v_mul_u32_u24 v153, v152, s62                      // re-calculate remainder
v_sub_nc_u32 v153, s61, v153                       // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s60, v152                      // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v153       // remainder
s_sub_u32 s61, s62, 1                              // GSU-1
s_cmp_eq_u32 s60, 0                                // quotient == 0
s_cselect_b32 s60, s[sgprGSUSumIdx+1], s61         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s60                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s60, 2, s[sgprStaggerUIter]
s_cmp_ge_i32 s60, 0
s_cbranch_scc0 label_Negative_T8JHFHKM7BO5OHXW
s_mul_hi_u32 s61, s60, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s60, s60, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_YSQ29IP70005TTFS
label_Negative_T8JHFHKM7BO5OHXW:
s_abs_i32 s60, s60
s_mul_hi_u32 s61, s60, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s60, s60, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s60, s60, 0xffffffff
s_xor_b32 s61, s61, 0xffffffff
s_add_u32 s60, s60, 0x1
s_addc_u32 s61, s61, 0
label_MultiplyDone_YSQ29IP70005TTFS:
s_sub_u32 s60, s60, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s61, s61, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s60, 2, s[sgprStaggerUIter]
s_cmp_ge_i32 s60, 0
s_cbranch_scc0 label_Negative_S4FDBQ587JJL6NOU
s_mul_hi_u32 s61, s60, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s60, s60, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_L43KTIIJOOEN7J6P
label_Negative_S4FDBQ587JJL6NOU:
s_abs_i32 s60, s60
s_mul_hi_u32 s61, s60, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s60, s60, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s60, s60, 0xffffffff
s_xor_b32 s61, s61, 0xffffffff
s_add_u32 s60, s60, 0x1
s_addc_u32 s61, s61, 0
label_MultiplyDone_L43KTIIJOOEN7J6P:
s_sub_u32 s60, s60, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s61, s61, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s60        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s61       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s60 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s61 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* Tail global read A */
/* g2l=0, load component 0 */
buffer_load_d16_b16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_d16_hi_b16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_d16_b16 v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_d16_hi_b16 v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:6 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_d16_b16 v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_d16_hi_b16 v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:10 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_d16_b16 v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_d16_hi_b16 v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:14 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_d16_b16 v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_d16_hi_b16 v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:2 // load one buffer value
/* g2l=4, load component 2 */
buffer_load_d16_b16 v[vgprG2LA+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:4 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_d16_hi_b16 v[vgprG2LA+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:6 // load one buffer value
/* g2l=4, load component 4 */
buffer_load_d16_b16 v[vgprG2LA+4+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:8 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_d16_hi_b16 v[vgprG2LA+4+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:10 // load one buffer value
/* g2l=4, load component 6 */
buffer_load_d16_b16 v[vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:12 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_d16_hi_b16 v[vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:14 // load one buffer value

/* Update M0 for DTLDS */

/* Tail global read B */
/* g2l=0, load component 0 */
buffer_load_d16_b16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_d16_hi_b16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:2 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_d16_b16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_d16_hi_b16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:6 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_d16_b16 v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_d16_hi_b16 v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:10 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_d16_b16 v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_d16_hi_b16 v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:14 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_d16_b16 v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_d16_hi_b16 v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:2 // load one buffer value
/* g2l=4, load component 2 */
buffer_load_d16_b16 v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:4 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_d16_hi_b16 v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:6 // load one buffer value
/* g2l=4, load component 4 */
buffer_load_d16_b16 v[vgprG2LB+4+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:8 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_d16_hi_b16 v[vgprG2LB+4+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:10 // load one buffer value
/* g2l=4, load component 6 */
buffer_load_d16_b16 v[vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:12 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_d16_hi_b16 v[vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:14 // load one buffer value
s_waitcnt vmcnt(0)                                 // 2wait for global read
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier

/* local write a */
ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:2560 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 2560

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:2560 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 2560

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
s_waitcnt lgkmcnt(0)                               // extra navi wait
s_barrier
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 136
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuB_X0_I0_BASE, 168
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA] offset:48 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+16:vgprValuA_X0_I0+16+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+20:vgprValuA_X0_I0+20+3], v[vgprLocalReadAddrA] offset:80 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+24:vgprValuA_X0_I0+24+3], v[vgprLocalReadAddrA] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuA_X0_I0+28:vgprValuA_X0_I0+28+3], v[vgprLocalReadAddrA] offset:112 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB] offset:48 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB] offset:80 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB] offset:112 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0

/* local read inc a */
s_mov_b32 s60, 32                                  // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc_lo, s60, v[vgprLocalReadAddrA+0] // lrA += 32 (bpeDS)

/* local read inc b */
                                                   // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, s60, v[vgprLocalReadAddrB+0] // lrB += 32 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
s_sub_i32 s60, s[sgprLoopCounterL], 1              // calculate 64bit groups index
s_lshr_b32 s61, s60, 2                             // calculate 64bit groups index
s_and_b32 s60, s60, 3                              // calculate shift value
s_sub_i32 s60, 3, s60                              // calculate shift value
s_lshl_b32 s60, s60, 4                             // calculate shift value
v_cmp_eq_i32 s62, s61, 0                           // handle this 64bit group: part 1
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+0+0:vgprValuA_X0_I0+0+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0], v[vgprValuA_X0_I0+0+0+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+1], v[vgprValuA_X0_I0+0+0+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+8+0:vgprValuA_X0_I0+8+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+0], v[vgprValuA_X0_I0+8+0+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+0+1], v[vgprValuA_X0_I0+8+0+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+16+0:vgprValuA_X0_I0+16+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+0], v[vgprValuA_X0_I0+16+0+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+0+1], v[vgprValuA_X0_I0+16+0+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+24+0:vgprValuA_X0_I0+24+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+0], v[vgprValuA_X0_I0+24+0+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+0+1], v[vgprValuA_X0_I0+24+0+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+0+0:vgprValuB_X0_I0+0+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0], v[vgprValuB_X0_I0+0+0+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+1], v[vgprValuB_X0_I0+0+0+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+8+0:vgprValuB_X0_I0+8+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+0], v[vgprValuB_X0_I0+8+0+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+0+1], v[vgprValuB_X0_I0+8+0+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+16+0:vgprValuB_X0_I0+16+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+0], v[vgprValuB_X0_I0+16+0+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+0+1], v[vgprValuB_X0_I0+16+0+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+24+0:vgprValuB_X0_I0+24+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+0], v[vgprValuB_X0_I0+24+0+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+0+1], v[vgprValuB_X0_I0+24+0+1], v203, s62 // shift if in this 64b group
v_cmp_eq_i32 s62, s61, 1                           // handle this 64bit group: part 1
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+0+2:vgprValuA_X0_I0+0+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+0], v[vgprValuA_X0_I0+0+2+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+1], v[vgprValuA_X0_I0+0+2+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+8+2:vgprValuA_X0_I0+8+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+8+2+0], v[vgprValuA_X0_I0+8+2+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+2+1], v[vgprValuA_X0_I0+8+2+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+16+2:vgprValuA_X0_I0+16+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+16+2+0], v[vgprValuA_X0_I0+16+2+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+2+1], v[vgprValuA_X0_I0+16+2+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+24+2:vgprValuA_X0_I0+24+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+24+2+0], v[vgprValuA_X0_I0+24+2+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+2+1], v[vgprValuA_X0_I0+24+2+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+0+2:vgprValuB_X0_I0+0+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+0], v[vgprValuB_X0_I0+0+2+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+1], v[vgprValuB_X0_I0+0+2+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+8+2:vgprValuB_X0_I0+8+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+8+2+0], v[vgprValuB_X0_I0+8+2+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+2+1], v[vgprValuB_X0_I0+8+2+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+16+2:vgprValuB_X0_I0+16+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+16+2+0], v[vgprValuB_X0_I0+16+2+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+2+1], v[vgprValuB_X0_I0+16+2+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+24+2:vgprValuB_X0_I0+24+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+24+2+0], v[vgprValuB_X0_I0+24+2+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+2+1], v[vgprValuB_X0_I0+24+2+1], v203, s62 // shift if in this 64b group
v_cmp_lt_i32 s62, s61, 1                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+0], v[vgprValuA_X0_I0+0+2+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+1], v[vgprValuA_X0_I0+0+2+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+2+0], v[vgprValuA_X0_I0+8+2+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+2+1], v[vgprValuA_X0_I0+8+2+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+2+0], v[vgprValuA_X0_I0+16+2+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+2+1], v[vgprValuA_X0_I0+16+2+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+2+0], v[vgprValuA_X0_I0+24+2+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+2+1], v[vgprValuA_X0_I0+24+2+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+0], v[vgprValuB_X0_I0+0+2+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+1], v[vgprValuB_X0_I0+0+2+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+2+0], v[vgprValuB_X0_I0+8+2+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+2+1], v[vgprValuB_X0_I0+8+2+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+2+0], v[vgprValuB_X0_I0+16+2+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+2+1], v[vgprValuB_X0_I0+16+2+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+2+0], v[vgprValuB_X0_I0+24+2+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+2+1], v[vgprValuB_X0_I0+24+2+1], 0, s62 // shift if in this 64b group
v_cmp_eq_i32 s62, s61, 2                           // handle this 64bit group: part 1
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+0], v[vgprValuA_X0_I0+0+4+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+1], v[vgprValuA_X0_I0+0+4+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+8+4:vgprValuA_X0_I0+8+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+8+4+0], v[vgprValuA_X0_I0+8+4+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+4+1], v[vgprValuA_X0_I0+8+4+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+16+4:vgprValuA_X0_I0+16+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+16+4+0], v[vgprValuA_X0_I0+16+4+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+4+1], v[vgprValuA_X0_I0+16+4+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+24+4:vgprValuA_X0_I0+24+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+24+4+0], v[vgprValuA_X0_I0+24+4+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+4+1], v[vgprValuA_X0_I0+24+4+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+0], v[vgprValuB_X0_I0+0+4+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+1], v[vgprValuB_X0_I0+0+4+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+8+4:vgprValuB_X0_I0+8+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+8+4+0], v[vgprValuB_X0_I0+8+4+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+4+1], v[vgprValuB_X0_I0+8+4+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+16+4:vgprValuB_X0_I0+16+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+16+4+0], v[vgprValuB_X0_I0+16+4+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+4+1], v[vgprValuB_X0_I0+16+4+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+24+4:vgprValuB_X0_I0+24+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+24+4+0], v[vgprValuB_X0_I0+24+4+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+4+1], v[vgprValuB_X0_I0+24+4+1], v203, s62 // shift if in this 64b group
v_cmp_lt_i32 s62, s61, 2                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+0], v[vgprValuA_X0_I0+0+4+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+1], v[vgprValuA_X0_I0+0+4+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+4+0], v[vgprValuA_X0_I0+8+4+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+4+1], v[vgprValuA_X0_I0+8+4+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+4+0], v[vgprValuA_X0_I0+16+4+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+4+1], v[vgprValuA_X0_I0+16+4+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+4+0], v[vgprValuA_X0_I0+24+4+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+4+1], v[vgprValuA_X0_I0+24+4+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+0], v[vgprValuB_X0_I0+0+4+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+1], v[vgprValuB_X0_I0+0+4+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+4+0], v[vgprValuB_X0_I0+8+4+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+4+1], v[vgprValuB_X0_I0+8+4+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+4+0], v[vgprValuB_X0_I0+16+4+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+4+1], v[vgprValuB_X0_I0+16+4+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+4+0], v[vgprValuB_X0_I0+24+4+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+4+1], v[vgprValuB_X0_I0+24+4+1], 0, s62 // shift if in this 64b group
v_cmp_eq_i32 s62, s61, 3                           // handle this 64bit group: part 1
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+0+6:vgprValuA_X0_I0+0+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+0], v[vgprValuA_X0_I0+0+6+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+1], v[vgprValuA_X0_I0+0+6+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+8+6:vgprValuA_X0_I0+8+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+8+6+0], v[vgprValuA_X0_I0+8+6+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+6+1], v[vgprValuA_X0_I0+8+6+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+16+6:vgprValuA_X0_I0+16+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+16+6+0], v[vgprValuA_X0_I0+16+6+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+6+1], v[vgprValuA_X0_I0+16+6+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuA_X0_I0+24+6:vgprValuA_X0_I0+24+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+24+6+0], v[vgprValuA_X0_I0+24+6+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+6+1], v[vgprValuA_X0_I0+24+6+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+0+6:vgprValuB_X0_I0+0+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+0], v[vgprValuB_X0_I0+0+6+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+1], v[vgprValuB_X0_I0+0+6+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+8+6:vgprValuB_X0_I0+8+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+8+6+0], v[vgprValuB_X0_I0+8+6+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+6+1], v[vgprValuB_X0_I0+8+6+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+16+6:vgprValuB_X0_I0+16+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+16+6+0], v[vgprValuB_X0_I0+16+6+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+6+1], v[vgprValuB_X0_I0+16+6+1], v203, s62 // shift if in this 64b group
v_lshlrev_b64 v[202:203], s60, v[vgprValuB_X0_I0+24+6:vgprValuB_X0_I0+24+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+24+6+0], v[vgprValuB_X0_I0+24+6+0], v202, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+6+1], v[vgprValuB_X0_I0+24+6+1], v203, s62 // shift if in this 64b group
v_cmp_lt_i32 s62, s61, 3                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+0], v[vgprValuA_X0_I0+0+6+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+1], v[vgprValuA_X0_I0+0+6+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+6+0], v[vgprValuA_X0_I0+8+6+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+8+6+1], v[vgprValuA_X0_I0+8+6+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+6+0], v[vgprValuA_X0_I0+16+6+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+16+6+1], v[vgprValuA_X0_I0+16+6+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+6+0], v[vgprValuA_X0_I0+24+6+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+24+6+1], v[vgprValuA_X0_I0+24+6+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+0], v[vgprValuB_X0_I0+0+6+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+1], v[vgprValuB_X0_I0+0+6+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+6+0], v[vgprValuB_X0_I0+8+6+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+8+6+1], v[vgprValuB_X0_I0+8+6+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+6+0], v[vgprValuB_X0_I0+16+6+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+16+6+1], v[vgprValuB_X0_I0+16+6+1], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+6+0], v[vgprValuB_X0_I0+24+6+0], 0, s62 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+24+6+1], v[vgprValuB_X0_I0+24+6+1], 0, s62 // shift if in this 64b group
s_nop 1
v_wmma_f32_16x16x16_f16 v[0:7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[0:7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x16_f16 v[8:15], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[8:15] // left value = v[8+0:15+0]
v_wmma_f32_16x16x16_f16 v[16:23], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[16:23] // left value = v[16+0:23+0]
v_wmma_f32_16x16x16_f16 v[24:31], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[24:31] // left value = v[24+0:31+0]
v_wmma_f32_16x16x16_f16 v[32:39], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[32:39] // left value = v[32+0:39+0]
v_wmma_f32_16x16x16_f16 v[40:47], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[40:47] // left value = v[40+0:47+0]
v_wmma_f32_16x16x16_f16 v[48:55], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[48:55] // left value = v[48+0:55+0]
v_wmma_f32_16x16x16_f16 v[56:63], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[56:63] // left value = v[56+0:63+0]
v_wmma_f32_16x16x16_f16 v[64:71], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[64:71] // left value = v[64+0:71+0]
v_wmma_f32_16x16x16_f16 v[72:79], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[72:79] // left value = v[72+0:79+0]
v_wmma_f32_16x16x16_f16 v[80:87], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[80:87] // left value = v[80+0:87+0]
v_wmma_f32_16x16x16_f16 v[88:95], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[88:95] // left value = v[88+0:95+0]
v_wmma_f32_16x16x16_f16 v[96:103], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[96:103] // left value = v[96+0:103+0]
v_wmma_f32_16x16x16_f16 v[104:111], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[104:111] // left value = v[104+0:111+0]
v_wmma_f32_16x16x16_f16 v[112:119], v[vgprValuA_X0_I0+16+0+0:vgprValuA_X0_I0+16+0+0+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[112:119] // left value = v[112+0:119+0]
v_wmma_f32_16x16x16_f16 v[120:127], v[vgprValuA_X0_I0+24+0+0:vgprValuA_X0_I0+24+0+0+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[120:127] // left value = v[120+0:127+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x10 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x10 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF

/* Tail: add MISC Vgpr [128...136) to pool */
label_Summation_End_UR8VN3A1SJCPC6PO:
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
.set sgprSrdC, 32
.set sgprSrdD, 24

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s42, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s41, s42, s[sgprStrideC1J]            // ScaleC s42 by Stride
s_mul_i32 s40, s42, s[sgprStrideC1J]               // ScaleC s42 by Stride
s_lshl_b64 s[40:41], s[40:41], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s40    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s41   // add hi to SRD
s_mul_hi_u32 s41, s42, s[sgprStrideD1J]            // ScaleD s42 by Stride
s_mul_i32 s40, s42, s[sgprStrideD1J]               // ScaleD s42 by Stride
s_lshl_b64 s[40:41], s[40:41], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s40    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s41   // add hi to SRD

s_mul_hi_u32 s41, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s40, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[40:41], s[40:41], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s40        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s41       // add hi to SRD
s_mul_hi_u32 s41, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s40, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[40:41], s[40:41], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s40        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s41       // add hi to SRD

s_and_b32 s11, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s11, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s41, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s40, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s39, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s39, s39, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s42, s39, s[sgprStrideC1J]            // Free1
s_mul_i32 s39, s39, s[sgprStrideC1J]               // Free1
s_add_u32 s40, s40, s39                            // Free1
s_addc_u32 s41, s41, s42                           // Free1
s_sub_u32 s39, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s39, s39, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s42, s39, s[sgprStrideCK]             // Free2
s_mul_i32 s39, s39, s[sgprStrideCK]                // Free2
s_add_u32 s40, s40, s39                            // Free2
s_addc_u32 s41, s41, s42                           // Free2
s_lshl_b64 s[40:41], s[40:41], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s40        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s41       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v132, 5, v[vgprSerial]               // 132 = Serial / 32
v_lshrrev_b32 v133, 1, v132                        // 133 = 132 / 2
v_mul_lo_u32 v129, 0x10, v133                      // wave coordination offset 1
v_and_b32 v133, 15, v[vgprSerial]                  // v133 = v[vgprSerial] % 16
v_add_lshl_u32 v129, v133, v129, 2                 // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v130, v129, s[sgprStrideC1J]          //  offset 1
v_mul_lo_u32 v131, v129, s[sgprStrideD1J]          //  offset 1
v_and_b32 v133, 1, v132                            // v133 = v132 % 2
v_mul_lo_u32 v133, 0x10, v133                      // wave coordination offset 0
v_and_b32 v128, 31, v[vgprSerial]                  // v128 = v[vgprSerial] % 32
v_lshrrev_b32 v128, 4, v128                        // 128 = 128 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v128, v133, v128, 2                 // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s8, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_nc_u32 v128, s8, v128                        // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 128, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_nc_u32 v129, s8, v129                        // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_3                         // branch if GSU == 1
label_GW_B0:
label_GW_B0_FD0:
s_and_b32 s40, 127, s[sgprSizeI]                   // s40 = s[sgprSizeI] % 128
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_Else            // jump if edges required
s_and_b32 s40, 127, s[sgprSizeJ]                   // s40 = s[sgprSizeJ] % 128
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_Then            // jump if edges required
label_GW_B0_FD0_VW4_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=28 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,2,0,0:vw4); (0,3,0,0:vw4); (0,4,0,0:vw4); (0,5,0,0:vw4); (0,6,0,0:vw4); (0,7,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,2,1,0:vw4); (0,3,1,0:vw4); (0,4,1,0:vw4); (0,5,1,0:vw4); (0,6,1,0:vw4); (0,7,1,0:vw4); (0,0,2,0:vw4); (0,1,2,0:vw4); (0,2,2,0:vw4); (0,3,2,0:vw4); (0,4,2,0:vw4); (0,5,2,0:vw4); (0,6,2,0:vw4); (0,7,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (0,2,3,0:vw4); (0,3,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_lshl_u32 v135, v131, v128, 0x2               // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 4, 2, 0), (0, 5, 2, 0), (0, 6, 2, 0), (0, 7, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0)] */
v_mov_b32 v[vgprValuC+140], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+185], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+186], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+187], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+188], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+189], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+190], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+191], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+192], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+193], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+194], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+195], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+196], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+197], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+198], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+199], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+204], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+205], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+206], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+207], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+208], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+209], v[vgprValuC+72]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+210], v[vgprValuC+80]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+211], v[vgprValuC+88]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+212], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+213], v[vgprValuC+73]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+214], v[vgprValuC+81]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+215], v[vgprValuC+89]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+216], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+217], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+218], v[vgprValuC+82]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+219], v[vgprValuC+90]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+220], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+221], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+222], v[vgprValuC+83]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+223], v[vgprValuC+91]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+224], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+225], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+226], v[vgprValuC+84]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+227], v[vgprValuC+92]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+228], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+229], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+230], v[vgprValuC+85]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+231], v[vgprValuC+93]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+232], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+233], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+234], v[vgprValuC+86]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+235], v[vgprValuC+94]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+236], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+237], v[vgprValuC+79]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+238], v[vgprValuC+87]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+239], v[vgprValuC+95]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+240], v[vgprValuC+96]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+241], v[vgprValuC+104]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+242], v[vgprValuC+112]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+243], v[vgprValuC+120]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+244], v[vgprValuC+97]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+245], v[vgprValuC+105]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+246], v[vgprValuC+113]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+247], v[vgprValuC+121]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+248], v[vgprValuC+98]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+249], v[vgprValuC+106]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+250], v[vgprValuC+114]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+251], v[vgprValuC+122]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+252], v[vgprValuC+99]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+253], v[vgprValuC+107]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+254], v[vgprValuC+115]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+255], v[vgprValuC+123]       // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b128 v[140:143], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[144:147], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b128 v[148:151], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b128 v[152:155], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
buffer_store_b128 v[156:159], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b128 v[160:163], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
buffer_store_b128 v[164:167], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
buffer_store_b128 v[168:171], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:224 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b128 v[172:175], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[176:179], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b128 v[180:183], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b128 v[184:187], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
buffer_store_b128 v[188:191], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b128 v[192:195], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
buffer_store_b128 v[196:199], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
buffer_store_b128 v[204:207], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:224 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b128 v[208:211], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[212:215], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b128 v[216:219], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b128 v[220:223], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
buffer_store_b128 v[224:227], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b128 v[228:231], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
buffer_store_b128 v[232:235], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
buffer_store_b128 v[236:239], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:224 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b128 v[240:243], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[244:247], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b128 v[248:251], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
buffer_store_b128 v[252:255], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,4,3,0:vw4); (0,5,3,0:vw4); (0,6,3,0:vw4); (0,7,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
/* (d1,vc1,d0,vc0)=(0,3,7,0) */

/* rC *= alpha batchElements=[(0, 4, 3, 0), (0, 5, 3, 0), (0, 6, 3, 0), (0, 7, 3, 0)] */
v_mov_b32 v[vgprValuC+140], v[vgprValuC+100]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+108]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+116]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+124]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+101]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+109]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+117]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+125]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+102]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+110]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+118]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+126]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+103]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+111]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+119]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+127]       // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b128 v[140:143], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128 // store D
buffer_store_b128 v[144:147], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:160 // store D
buffer_store_b128 v[148:151], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:192 // store D
buffer_store_b128 v[152:155], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:224 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_NonEdgeEnd:
label_GW_B0_FD0_VW4_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=22 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,2,0,0:vw4); (0,3,0,0:vw4); (0,4,0,0:vw4); (0,5,0,0:vw4); (0,6,0,0:vw4); (0,7,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,2,1,0:vw4); (0,3,1,0:vw4); (0,4,1,0:vw4); (0,5,1,0:vw4); (0,6,1,0:vw4); (0,7,1,0:vw4); (0,0,2,0:vw4); (0,1,2,0:vw4); (0,2,2,0:vw4); (0,3,2,0:vw4); (0,4,2,0:vw4); (0,5,2,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v135, v131, v128, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, v134, v135, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v202, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, v134, v202, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v228, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v228, v134, v228, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v230, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v230, v134, v230, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v232, v131, v128, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v234, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v234, v134, v234, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v236, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v236, v134, v236, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v238, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v240, v131, v128, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v240, v134, v240, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v242, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v242, v134, v242, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v244, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 4, 2, 0), (0, 5, 2, 0)] */
v_mov_b32 v[vgprValuC+136], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+185], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+186], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+187], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+188], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+189], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+190], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+191], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+192], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+193], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+194], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+195], v[vgprValuC+62]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+196], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+197], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+198], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+199], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+204], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+205], v[vgprValuC+72]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+206], v[vgprValuC+80]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+207], v[vgprValuC+88]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+208], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+209], v[vgprValuC+73]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+210], v[vgprValuC+81]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+211], v[vgprValuC+89]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+212], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+213], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+214], v[vgprValuC+82]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+215], v[vgprValuC+90]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+216], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+217], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+218], v[vgprValuC+83]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+219], v[vgprValuC+91]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+220], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+221], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+222], v[vgprValuC+84]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+223], v[vgprValuC+92]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+224], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+225], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+226], v[vgprValuC+85]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+227], v[vgprValuC+93]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b128 v[136:139], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[140:143], v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[144:147], v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[148:151], v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[152:155], v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[156:159], v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[160:163], v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[164:167], v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[168:171], v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[172:175], v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[176:179], v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[180:183], v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[184:187], v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[188:191], v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[192:195], v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[196:199], v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[204:207], v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[208:211], v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[212:215], v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[216:219], v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[220:223], v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[224:227], v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,6,2,0:vw4); (0,7,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (0,2,3,0:vw4); (0,3,3,0:vw4); (0,4,3,0:vw4); (0,5,3,0:vw4); (0,6,3,0:vw4); (0,7,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v135, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, v134, v135, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v176, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, v134, v176, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v177, v131, v128, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v177, v134, v177, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v178, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, v134, v178, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v179, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v179, v134, v179, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v180, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, v134, v180, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v181, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v181, v134, v181, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v182, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, v134, v182, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v183, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v183, v134, v183, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v184, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, v134, v184, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 6, 2, 0), (0, 7, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0), (0, 4, 3, 0), (0, 5, 3, 0), (0, 6, 3, 0), (0, 7, 3, 0)] */
v_mov_b32 v[vgprValuC+136], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+86]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+94]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+79]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+87]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+95]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+96]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+104]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+112]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+120]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+97]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+105]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+113]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+121]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+98]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+106]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+114]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+122]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+99]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+107]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+115]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+123]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+100]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+108]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+116]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+124]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+101]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+109]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+117]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+125]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+102]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+110]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+118]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+126]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+103]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+111]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+119]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+127]       // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b128 v[136:139], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[140:143], v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[144:147], v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[148:151], v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[152:155], v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[156:159], v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[160:163], v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[164:167], v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[168:171], v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b128 v[172:175], v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_Else:
label_GW_B0_FD0_VW1_Else:
label_GW_B0_FD0_VW1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=60 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,2,0,0:vw1); (0,2,0,1:vw1); (0,2,0,2:vw1); (0,2,0,3:vw1); (0,3,0,0:vw1); (0,3,0,1:vw1); (0,3,0,2:vw1); (0,3,0,3:vw1); (0,4,0,0:vw1); (0,4,0,1:vw1); (0,4,0,2:vw1); (0,4,0,3:vw1); (0,5,0,0:vw1); (0,5,0,1:vw1); (0,5,0,2:vw1); (0,5,0,3:vw1); (0,6,0,0:vw1); (0,6,0,1:vw1); (0,6,0,2:vw1); (0,6,0,3:vw1); (0,7,0,0:vw1); (0,7,0,1:vw1); (0,7,0,2:vw1); (0,7,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,2,1,0:vw1); (0,2,1,1:vw1); (0,2,1,2:vw1); (0,2,1,3:vw1); (0,3,1,0:vw1); (0,3,1,1:vw1); (0,3,1,2:vw1); (0,3,1,3:vw1); (0,4,1,0:vw1); (0,4,1,1:vw1); (0,4,1,2:vw1); (0,4,1,3:vw1); (0,5,1,0:vw1); (0,5,1,1:vw1); (0,5,1,2:vw1); (0,5,1,3:vw1); (0,6,1,0:vw1); (0,6,1,1:vw1); (0,6,1,2:vw1); (0,6,1,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v195, v131, v128, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v195, v134, v195, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v196, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v197, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v197, v134, v197, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v198, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v199, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, v134, v199, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v202, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, v134, v202, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v204, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v204, v134, v204, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v205, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v206, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v206, v134, v206, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v207, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v208, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, v134, v208, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v210, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v210, v134, v210, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v211, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v212, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v212, v134, v212, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v213, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v214, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, v134, v214, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v216, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v216, v134, v216, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v217, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v218, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v218, v134, v218, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v219, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v220, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v220, v134, v220, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v222, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v222, v134, v222, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v223, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v224, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v224, v134, v224, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v225, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v226, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, v134, v226, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v228, v131, v128, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v228, v134, v228, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v230, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v230, v134, v230, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v232, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v234, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v234, v134, v234, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v236, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v236, v134, v236, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v238, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v240, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v240, v134, v240, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v242, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v242, v134, v242, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v244, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v246, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v246, v134, v246, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v247, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v248, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v248, v134, v248, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v249, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v250, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, v134, v250, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v251, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v252, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v252, v134, v252, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v253, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v254, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v254, v134, v254, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v255, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 0, 3), (0, 3, 0, 0), (0, 3, 0, 1), (0, 3, 0, 2), (0, 3, 0, 3), (0, 4, 0, 0), (0, 4, 0, 1), (0, 4, 0, 2), (0, 4, 0, 3), (0, 5, 0, 0), (0, 5, 0, 1), (0, 5, 0, 2), (0, 5, 0, 3), (0, 6, 0, 0), (0, 6, 0, 1), (0, 6, 0, 2), (0, 6, 0, 3), (0, 7, 0, 0), (0, 7, 0, 1), (0, 7, 0, 2), (0, 7, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 1, 2), (0, 2, 1, 3), (0, 3, 1, 0), (0, 3, 1, 1), (0, 3, 1, 2), (0, 3, 1, 3), (0, 4, 1, 0), (0, 4, 1, 1), (0, 4, 1, 2), (0, 4, 1, 3), (0, 5, 1, 0), (0, 5, 1, 1), (0, 5, 1, 2), (0, 5, 1, 3), (0, 6, 1, 0), (0, 6, 1, 1), (0, 6, 1, 2), (0, 6, 1, 3)] */
v_mov_b32 v[vgprValuC+135], v[vgprValuC+0]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+8]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+1]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+9]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+2]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+10]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+3]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+11]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+4]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+12]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+5]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+13]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+32]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+40]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+48]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+56]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+33]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+41]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+49]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+57]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+34]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+42]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+50]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+58]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+35]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+43]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+51]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+59]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+36]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+44]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+185], v[vgprValuC+52]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+186], v[vgprValuC+60]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+187], v[vgprValuC+37]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+188], v[vgprValuC+45]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+189], v[vgprValuC+53]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+190], v[vgprValuC+61]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+191], v[vgprValuC+38]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+192], v[vgprValuC+46]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+193], v[vgprValuC+54]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+194], v[vgprValuC+62]        // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v135, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v145, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v149, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v151, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v153, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v155, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v157, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v159, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v161, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v163, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v165, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v166, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v167, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v169, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v171, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v172, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v173, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v174, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v175, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v176, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v177, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v178, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v179, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v180, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v181, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v182, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v183, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v184, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v185, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v186, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v187, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v188, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v189, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v190, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v191, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v192, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v193, v254, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v194, v255, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,7,1,0:vw1); (0,7,1,1:vw1); (0,7,1,2:vw1); (0,7,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,2,2,0:vw1); (0,2,2,1:vw1); (0,2,2,2:vw1); (0,2,2,3:vw1); (0,3,2,0:vw1); (0,3,2,1:vw1); (0,3,2,2:vw1); (0,3,2,3:vw1); (0,4,2,0:vw1); (0,4,2,1:vw1); (0,4,2,2:vw1); (0,4,2,3:vw1); (0,5,2,0:vw1); (0,5,2,1:vw1); (0,5,2,2:vw1); (0,5,2,3:vw1); (0,6,2,0:vw1); (0,6,2,1:vw1); (0,6,2,2:vw1); (0,6,2,3:vw1); (0,7,2,0:vw1); (0,7,2,1:vw1); (0,7,2,2:vw1); (0,7,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1); (0,2,3,0:vw1); (0,2,3,1:vw1); (0,2,3,2:vw1); (0,2,3,3:vw1); (0,3,3,0:vw1); (0,3,3,1:vw1); (0,3,3,2:vw1); (0,3,3,3:vw1); (0,4,3,0:vw1); (0,4,3,1:vw1); (0,4,3,2:vw1); (0,4,3,3:vw1); (0,5,3,0:vw1); (0,5,3,1:vw1); (0,5,3,2:vw1); (0,5,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v195, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v195, v134, v195, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v196, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v197, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v197, v134, v197, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v198, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v199, v131, v128, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, v134, v199, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v202, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, v134, v202, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v204, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v204, v134, v204, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v205, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v206, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v206, v134, v206, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v207, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v208, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, v134, v208, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v210, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v210, v134, v210, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v211, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v212, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v212, v134, v212, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v213, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v214, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, v134, v214, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v216, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v216, v134, v216, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v217, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v218, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v218, v134, v218, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v219, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v220, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v220, v134, v220, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v222, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v222, v134, v222, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v223, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v224, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v224, v134, v224, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v225, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v226, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, v134, v226, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v228, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v228, v134, v228, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v230, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v230, v134, v230, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v232, v131, v128, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v234, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v234, v134, v234, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v236, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v236, v134, v236, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v238, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v240, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v240, v134, v240, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v242, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v242, v134, v242, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v244, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v246, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v246, v134, v246, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v247, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v248, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v248, v134, v248, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v249, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v250, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, v134, v250, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v251, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v252, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v252, v134, v252, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v253, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v254, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v254, v134, v254, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v255, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 7, 1, 0), (0, 7, 1, 1), (0, 7, 1, 2), (0, 7, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 2, 2, 0), (0, 2, 2, 1), (0, 2, 2, 2), (0, 2, 2, 3), (0, 3, 2, 0), (0, 3, 2, 1), (0, 3, 2, 2), (0, 3, 2, 3), (0, 4, 2, 0), (0, 4, 2, 1), (0, 4, 2, 2), (0, 4, 2, 3), (0, 5, 2, 0), (0, 5, 2, 1), (0, 5, 2, 2), (0, 5, 2, 3), (0, 6, 2, 0), (0, 6, 2, 1), (0, 6, 2, 2), (0, 6, 2, 3), (0, 7, 2, 0), (0, 7, 2, 1), (0, 7, 2, 2), (0, 7, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3), (0, 2, 3, 0), (0, 2, 3, 1), (0, 2, 3, 2), (0, 2, 3, 3), (0, 3, 3, 0), (0, 3, 3, 1), (0, 3, 3, 2), (0, 3, 3, 3), (0, 4, 3, 0), (0, 4, 3, 1), (0, 4, 3, 2), (0, 4, 3, 3), (0, 5, 3, 0), (0, 5, 3, 1), (0, 5, 3, 2), (0, 5, 3, 3)] */
v_mov_b32 v[vgprValuC+135], v[vgprValuC+39]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+47]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+55]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+63]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+64]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+72]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+80]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+88]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+143], v[vgprValuC+65]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+73]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+145], v[vgprValuC+81]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+89]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+147], v[vgprValuC+66]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+148], v[vgprValuC+74]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+149], v[vgprValuC+82]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+150], v[vgprValuC+90]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+151], v[vgprValuC+67]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+152], v[vgprValuC+75]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+153], v[vgprValuC+83]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+154], v[vgprValuC+91]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+155], v[vgprValuC+68]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+156], v[vgprValuC+76]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+157], v[vgprValuC+84]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+158], v[vgprValuC+92]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+159], v[vgprValuC+69]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+160], v[vgprValuC+77]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+161], v[vgprValuC+85]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+162], v[vgprValuC+93]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+163], v[vgprValuC+70]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+164], v[vgprValuC+78]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+165], v[vgprValuC+86]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+166], v[vgprValuC+94]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+167], v[vgprValuC+71]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+168], v[vgprValuC+79]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+169], v[vgprValuC+87]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+170], v[vgprValuC+95]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+171], v[vgprValuC+96]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+172], v[vgprValuC+104]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+173], v[vgprValuC+112]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+174], v[vgprValuC+120]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+175], v[vgprValuC+97]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+176], v[vgprValuC+105]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+177], v[vgprValuC+113]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+178], v[vgprValuC+121]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+179], v[vgprValuC+98]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+180], v[vgprValuC+106]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+181], v[vgprValuC+114]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+182], v[vgprValuC+122]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+183], v[vgprValuC+99]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+184], v[vgprValuC+107]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+185], v[vgprValuC+115]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+186], v[vgprValuC+123]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+187], v[vgprValuC+100]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+188], v[vgprValuC+108]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+189], v[vgprValuC+116]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+190], v[vgprValuC+124]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+191], v[vgprValuC+101]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+192], v[vgprValuC+109]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+193], v[vgprValuC+117]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+194], v[vgprValuC+125]       // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v135, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v143, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v144, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v145, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v146, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v147, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v148, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v149, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v150, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v151, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v152, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v153, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v154, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v155, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v156, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v157, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v158, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v159, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v160, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v161, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v162, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v163, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v164, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v165, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v166, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v167, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v168, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v169, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v170, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v171, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v172, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v173, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v174, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v175, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v176, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v177, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v178, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v179, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v180, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v181, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v182, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v183, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v184, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v185, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v186, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v187, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v188, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v189, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v190, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v191, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v192, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v193, v254, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v194, v255, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,6,3,0:vw1); (0,6,3,1:vw1); (0,6,3,2:vw1); (0,6,3,3:vw1); (0,7,3,0:vw1); (0,7,3,1:vw1); (0,7,3,2:vw1); (0,7,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v143, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v143, v134, v143, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v144, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, v134, v144, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v145, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, v134, v145, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v146, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, v134, v146, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v147, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, v134, v147, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v148, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, v134, v148, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v149, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v149, v134, v149, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v150, v131, v132, 0x2               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, v134, v150, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 6, 3, 0), (0, 6, 3, 1), (0, 6, 3, 2), (0, 6, 3, 3), (0, 7, 3, 0), (0, 7, 3, 1), (0, 7, 3, 2), (0, 7, 3, 3)] */
v_mov_b32 v[vgprValuC+135], v[vgprValuC+102]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+110]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+137], v[vgprValuC+118]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+126]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+139], v[vgprValuC+103]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+111]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+141], v[vgprValuC+119]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+127]       // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v135, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v136, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v137, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v138, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v139, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v140, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v141, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v142, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[40:41]                               // addr of next instr
s_add_i32 s42, label_KernelEnd, 4                  // target branch offset
s_add_u32 s40, s40, s42                            // add target branch offset
s_addc_u32 s41, s41, 0                             // add high and carry
s_setpc_b64 s[40:41]                               // branch to label_KernelEnd
label_GSU_3:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1                         // Branch if Beta is not zero

label_GW_B0_1:
label_GW_B0_FD0_1:
s_and_b32 s40, 127, s[sgprSizeI]                   // s40 = s[sgprSizeI] % 128
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_Else_1          // jump if edges required
s_and_b32 s40, 127, s[sgprSizeJ]                   // s40 = s[sgprSizeJ] % 128
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_Then_1          // jump if edges required
label_GW_B0_FD0_VW4_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=28 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,2,0,0:vw4); (0,3,0,0:vw4); (0,4,0,0:vw4); (0,5,0,0:vw4); (0,6,0,0:vw4); (0,7,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,2,1,0:vw4); (0,3,1,0:vw4); (0,4,1,0:vw4); (0,5,1,0:vw4); (0,6,1,0:vw4); (0,7,1,0:vw4); (0,0,2,0:vw4); (0,1,2,0:vw4); (0,2,2,0:vw4); (0,3,2,0:vw4); (0,4,2,0:vw4); (0,5,2,0:vw4); (0,6,2,0:vw4); (0,7,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (0,2,3,0:vw4); (0,3,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_lshl_u32 v135, v131, v128, 0x1               // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 4, 2, 0), (0, 5, 2, 0), (0, 6, 2, 0), (0, 7, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0)] */
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+204], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+205], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+206], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+208], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+210], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+212], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+214], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+216], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+217], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+218], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+219], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+220], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+221], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+223], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+224], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+225], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+226], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+227], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+228], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+229], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+230], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+231], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+232], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+233], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+234], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+235], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+236], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+237], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+238], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+239], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+240], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+241], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+242], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+243], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+244], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+245], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+246], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+247], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+248], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+249], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+250], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+251], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+252], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+253], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+254], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+255], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v[vgprValuC+140], v[vgprValuC+140]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+141], v[vgprValuC+141]   // convert C to fp16
v_pack_b32_f16 v140, v[vgprValuC+140], v[vgprValuC+141] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+142], v[vgprValuC+142]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+143], v[vgprValuC+143]   // convert C to fp16
v_pack_b32_f16 v141, v[vgprValuC+142], v[vgprValuC+143] // Pack with neighbor
buffer_store_b64 v[140:141], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+144], v[vgprValuC+144]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+145], v[vgprValuC+145]   // convert C to fp16
v_pack_b32_f16 v144, v[vgprValuC+144], v[vgprValuC+145] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+146], v[vgprValuC+146]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+147], v[vgprValuC+147]   // convert C to fp16
v_pack_b32_f16 v145, v[vgprValuC+146], v[vgprValuC+147] // Pack with neighbor
buffer_store_b64 v[144:145], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
v_cvt_f16_f32 v[vgprValuC+148], v[vgprValuC+148]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+149], v[vgprValuC+149]   // convert C to fp16
v_pack_b32_f16 v148, v[vgprValuC+148], v[vgprValuC+149] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+150], v[vgprValuC+150]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+151], v[vgprValuC+151]   // convert C to fp16
v_pack_b32_f16 v149, v[vgprValuC+150], v[vgprValuC+151] // Pack with neighbor
buffer_store_b64 v[148:149], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cvt_f16_f32 v[vgprValuC+152], v[vgprValuC+152]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+153], v[vgprValuC+153]   // convert C to fp16
v_pack_b32_f16 v152, v[vgprValuC+152], v[vgprValuC+153] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+154], v[vgprValuC+154]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+155], v[vgprValuC+155]   // convert C to fp16
v_pack_b32_f16 v153, v[vgprValuC+154], v[vgprValuC+155] // Pack with neighbor
buffer_store_b64 v[152:153], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
v_cvt_f16_f32 v[vgprValuC+156], v[vgprValuC+156]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+157], v[vgprValuC+157]   // convert C to fp16
v_pack_b32_f16 v156, v[vgprValuC+156], v[vgprValuC+157] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+158], v[vgprValuC+158]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+159], v[vgprValuC+159]   // convert C to fp16
v_pack_b32_f16 v157, v[vgprValuC+158], v[vgprValuC+159] // Pack with neighbor
buffer_store_b64 v[156:157], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v[vgprValuC+160], v[vgprValuC+160]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+161], v[vgprValuC+161]   // convert C to fp16
v_pack_b32_f16 v160, v[vgprValuC+160], v[vgprValuC+161] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+162], v[vgprValuC+162]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+163], v[vgprValuC+163]   // convert C to fp16
v_pack_b32_f16 v161, v[vgprValuC+162], v[vgprValuC+163] // Pack with neighbor
buffer_store_b64 v[160:161], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D
v_cvt_f16_f32 v[vgprValuC+164], v[vgprValuC+164]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+165], v[vgprValuC+165]   // convert C to fp16
v_pack_b32_f16 v164, v[vgprValuC+164], v[vgprValuC+165] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+166], v[vgprValuC+166]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+167], v[vgprValuC+167]   // convert C to fp16
v_pack_b32_f16 v165, v[vgprValuC+166], v[vgprValuC+167] // Pack with neighbor
buffer_store_b64 v[164:165], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
v_cvt_f16_f32 v[vgprValuC+168], v[vgprValuC+168]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+169], v[vgprValuC+169]   // convert C to fp16
v_pack_b32_f16 v168, v[vgprValuC+168], v[vgprValuC+169] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+170], v[vgprValuC+170]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+171], v[vgprValuC+171]   // convert C to fp16
v_pack_b32_f16 v169, v[vgprValuC+170], v[vgprValuC+171] // Pack with neighbor
buffer_store_b64 v[168:169], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D
v_cvt_f16_f32 v[vgprValuC+172], v[vgprValuC+172]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+173], v[vgprValuC+173]   // convert C to fp16
v_pack_b32_f16 v172, v[vgprValuC+172], v[vgprValuC+173] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+174], v[vgprValuC+174]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+175], v[vgprValuC+175]   // convert C to fp16
v_pack_b32_f16 v173, v[vgprValuC+174], v[vgprValuC+175] // Pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b64 v[172:173], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+176], v[vgprValuC+176]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+177], v[vgprValuC+177]   // convert C to fp16
v_pack_b32_f16 v176, v[vgprValuC+176], v[vgprValuC+177] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+178], v[vgprValuC+178]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+179], v[vgprValuC+179]   // convert C to fp16
v_pack_b32_f16 v177, v[vgprValuC+178], v[vgprValuC+179] // Pack with neighbor
buffer_store_b64 v[176:177], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
v_cvt_f16_f32 v[vgprValuC+180], v[vgprValuC+180]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+181], v[vgprValuC+181]   // convert C to fp16
v_pack_b32_f16 v180, v[vgprValuC+180], v[vgprValuC+181] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+182], v[vgprValuC+182]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+183], v[vgprValuC+183]   // convert C to fp16
v_pack_b32_f16 v181, v[vgprValuC+182], v[vgprValuC+183] // Pack with neighbor
buffer_store_b64 v[180:181], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cvt_f16_f32 v[vgprValuC+184], v[vgprValuC+184]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+185], v[vgprValuC+185]   // convert C to fp16
v_pack_b32_f16 v184, v[vgprValuC+184], v[vgprValuC+185] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+186], v[vgprValuC+186]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+187], v[vgprValuC+187]   // convert C to fp16
v_pack_b32_f16 v185, v[vgprValuC+186], v[vgprValuC+187] // Pack with neighbor
buffer_store_b64 v[184:185], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
v_cvt_f16_f32 v[vgprValuC+188], v[vgprValuC+188]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+189], v[vgprValuC+189]   // convert C to fp16
v_pack_b32_f16 v188, v[vgprValuC+188], v[vgprValuC+189] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+190], v[vgprValuC+190]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+191], v[vgprValuC+191]   // convert C to fp16
v_pack_b32_f16 v189, v[vgprValuC+190], v[vgprValuC+191] // Pack with neighbor
buffer_store_b64 v[188:189], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v[vgprValuC+192], v[vgprValuC+192]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+193], v[vgprValuC+193]   // convert C to fp16
v_pack_b32_f16 v192, v[vgprValuC+192], v[vgprValuC+193] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+194], v[vgprValuC+194]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+195], v[vgprValuC+195]   // convert C to fp16
v_pack_b32_f16 v193, v[vgprValuC+194], v[vgprValuC+195] // Pack with neighbor
buffer_store_b64 v[192:193], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D
v_cvt_f16_f32 v[vgprValuC+196], v[vgprValuC+196]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+197], v[vgprValuC+197]   // convert C to fp16
v_pack_b32_f16 v196, v[vgprValuC+196], v[vgprValuC+197] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+198], v[vgprValuC+198]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+199], v[vgprValuC+199]   // convert C to fp16
v_pack_b32_f16 v197, v[vgprValuC+198], v[vgprValuC+199] // Pack with neighbor
buffer_store_b64 v[196:197], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
v_cvt_f16_f32 v[vgprValuC+204], v[vgprValuC+204]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+205], v[vgprValuC+205]   // convert C to fp16
v_pack_b32_f16 v204, v[vgprValuC+204], v[vgprValuC+205] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+206], v[vgprValuC+206]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+207], v[vgprValuC+207]   // convert C to fp16
v_pack_b32_f16 v205, v[vgprValuC+206], v[vgprValuC+207] // Pack with neighbor
buffer_store_b64 v[204:205], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D
v_cvt_f16_f32 v[vgprValuC+208], v[vgprValuC+208]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+209], v[vgprValuC+209]   // convert C to fp16
v_pack_b32_f16 v208, v[vgprValuC+208], v[vgprValuC+209] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+210], v[vgprValuC+210]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+211], v[vgprValuC+211]   // convert C to fp16
v_pack_b32_f16 v209, v[vgprValuC+210], v[vgprValuC+211] // Pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b64 v[208:209], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+212], v[vgprValuC+212]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+213], v[vgprValuC+213]   // convert C to fp16
v_pack_b32_f16 v212, v[vgprValuC+212], v[vgprValuC+213] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+214], v[vgprValuC+214]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+215], v[vgprValuC+215]   // convert C to fp16
v_pack_b32_f16 v213, v[vgprValuC+214], v[vgprValuC+215] // Pack with neighbor
buffer_store_b64 v[212:213], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
v_cvt_f16_f32 v[vgprValuC+216], v[vgprValuC+216]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+217], v[vgprValuC+217]   // convert C to fp16
v_pack_b32_f16 v216, v[vgprValuC+216], v[vgprValuC+217] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+218], v[vgprValuC+218]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+219], v[vgprValuC+219]   // convert C to fp16
v_pack_b32_f16 v217, v[vgprValuC+218], v[vgprValuC+219] // Pack with neighbor
buffer_store_b64 v[216:217], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cvt_f16_f32 v[vgprValuC+220], v[vgprValuC+220]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+221], v[vgprValuC+221]   // convert C to fp16
v_pack_b32_f16 v220, v[vgprValuC+220], v[vgprValuC+221] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+222], v[vgprValuC+222]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+223], v[vgprValuC+223]   // convert C to fp16
v_pack_b32_f16 v221, v[vgprValuC+222], v[vgprValuC+223] // Pack with neighbor
buffer_store_b64 v[220:221], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
v_cvt_f16_f32 v[vgprValuC+224], v[vgprValuC+224]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+225], v[vgprValuC+225]   // convert C to fp16
v_pack_b32_f16 v224, v[vgprValuC+224], v[vgprValuC+225] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+226], v[vgprValuC+226]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+227], v[vgprValuC+227]   // convert C to fp16
v_pack_b32_f16 v225, v[vgprValuC+226], v[vgprValuC+227] // Pack with neighbor
buffer_store_b64 v[224:225], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v[vgprValuC+228], v[vgprValuC+228]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+229], v[vgprValuC+229]   // convert C to fp16
v_pack_b32_f16 v228, v[vgprValuC+228], v[vgprValuC+229] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+230], v[vgprValuC+230]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+231], v[vgprValuC+231]   // convert C to fp16
v_pack_b32_f16 v229, v[vgprValuC+230], v[vgprValuC+231] // Pack with neighbor
buffer_store_b64 v[228:229], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D
v_cvt_f16_f32 v[vgprValuC+232], v[vgprValuC+232]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+233], v[vgprValuC+233]   // convert C to fp16
v_pack_b32_f16 v232, v[vgprValuC+232], v[vgprValuC+233] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+234], v[vgprValuC+234]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+235], v[vgprValuC+235]   // convert C to fp16
v_pack_b32_f16 v233, v[vgprValuC+234], v[vgprValuC+235] // Pack with neighbor
buffer_store_b64 v[232:233], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
v_cvt_f16_f32 v[vgprValuC+236], v[vgprValuC+236]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+237], v[vgprValuC+237]   // convert C to fp16
v_pack_b32_f16 v236, v[vgprValuC+236], v[vgprValuC+237] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+238], v[vgprValuC+238]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+239], v[vgprValuC+239]   // convert C to fp16
v_pack_b32_f16 v237, v[vgprValuC+238], v[vgprValuC+239] // Pack with neighbor
buffer_store_b64 v[236:237], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D
v_cvt_f16_f32 v[vgprValuC+240], v[vgprValuC+240]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+241], v[vgprValuC+241]   // convert C to fp16
v_pack_b32_f16 v240, v[vgprValuC+240], v[vgprValuC+241] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+242], v[vgprValuC+242]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+243], v[vgprValuC+243]   // convert C to fp16
v_pack_b32_f16 v241, v[vgprValuC+242], v[vgprValuC+243] // Pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b64 v[240:241], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+244], v[vgprValuC+244]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+245], v[vgprValuC+245]   // convert C to fp16
v_pack_b32_f16 v244, v[vgprValuC+244], v[vgprValuC+245] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+246], v[vgprValuC+246]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+247], v[vgprValuC+247]   // convert C to fp16
v_pack_b32_f16 v245, v[vgprValuC+246], v[vgprValuC+247] // Pack with neighbor
buffer_store_b64 v[244:245], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
v_cvt_f16_f32 v[vgprValuC+248], v[vgprValuC+248]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+249], v[vgprValuC+249]   // convert C to fp16
v_pack_b32_f16 v248, v[vgprValuC+248], v[vgprValuC+249] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+250], v[vgprValuC+250]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+251], v[vgprValuC+251]   // convert C to fp16
v_pack_b32_f16 v249, v[vgprValuC+250], v[vgprValuC+251] // Pack with neighbor
buffer_store_b64 v[248:249], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
v_cvt_f16_f32 v[vgprValuC+252], v[vgprValuC+252]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+253], v[vgprValuC+253]   // convert C to fp16
v_pack_b32_f16 v252, v[vgprValuC+252], v[vgprValuC+253] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+254], v[vgprValuC+254]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+255], v[vgprValuC+255]   // convert C to fp16
v_pack_b32_f16 v253, v[vgprValuC+254], v[vgprValuC+255] // Pack with neighbor
buffer_store_b64 v[252:253], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,4,3,0:vw4); (0,5,3,0:vw4); (0,6,3,0:vw4); (0,7,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
/* (d1,vc1,d0,vc0)=(0,3,7,0) */

/* rC *= alpha batchElements=[(0, 4, 3, 0), (0, 5, 3, 0), (0, 6, 3, 0), (0, 7, 3, 0)] */
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v[vgprValuC+140], v[vgprValuC+140]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+141], v[vgprValuC+141]   // convert C to fp16
v_pack_b32_f16 v140, v[vgprValuC+140], v[vgprValuC+141] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+142], v[vgprValuC+142]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+143], v[vgprValuC+143]   // convert C to fp16
v_pack_b32_f16 v141, v[vgprValuC+142], v[vgprValuC+143] // Pack with neighbor
buffer_store_b64 v[140:141], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D
v_cvt_f16_f32 v[vgprValuC+144], v[vgprValuC+144]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+145], v[vgprValuC+145]   // convert C to fp16
v_pack_b32_f16 v144, v[vgprValuC+144], v[vgprValuC+145] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+146], v[vgprValuC+146]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+147], v[vgprValuC+147]   // convert C to fp16
v_pack_b32_f16 v145, v[vgprValuC+146], v[vgprValuC+147] // Pack with neighbor
buffer_store_b64 v[144:145], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D
v_cvt_f16_f32 v[vgprValuC+148], v[vgprValuC+148]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+149], v[vgprValuC+149]   // convert C to fp16
v_pack_b32_f16 v148, v[vgprValuC+148], v[vgprValuC+149] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+150], v[vgprValuC+150]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+151], v[vgprValuC+151]   // convert C to fp16
v_pack_b32_f16 v149, v[vgprValuC+150], v[vgprValuC+151] // Pack with neighbor
buffer_store_b64 v[148:149], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D
v_cvt_f16_f32 v[vgprValuC+152], v[vgprValuC+152]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+153], v[vgprValuC+153]   // convert C to fp16
v_pack_b32_f16 v152, v[vgprValuC+152], v[vgprValuC+153] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+154], v[vgprValuC+154]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+155], v[vgprValuC+155]   // convert C to fp16
v_pack_b32_f16 v153, v[vgprValuC+154], v[vgprValuC+155] // Pack with neighbor
buffer_store_b64 v[152:153], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_NonEdgeEnd_1:
label_GW_B0_FD0_VW4_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=22 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,2,0,0:vw4); (0,3,0,0:vw4); (0,4,0,0:vw4); (0,5,0,0:vw4); (0,6,0,0:vw4); (0,7,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,2,1,0:vw4); (0,3,1,0:vw4); (0,4,1,0:vw4); (0,5,1,0:vw4); (0,6,1,0:vw4); (0,7,1,0:vw4); (0,0,2,0:vw4); (0,1,2,0:vw4); (0,2,2,0:vw4); (0,3,2,0:vw4); (0,4,2,0:vw4); (0,5,2,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v135, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, v134, v135, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v202, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, v134, v202, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v228, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v228, v134, v228, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v230, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v230, v134, v230, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v232, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v234, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v234, v134, v234, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v236, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v236, v134, v236, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v238, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v240, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v240, v134, v240, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v242, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v242, v134, v242, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v244, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 4, 2, 0), (0, 5, 2, 0)] */
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+204], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+205], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+206], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+208], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+210], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+212], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+214], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+216], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+217], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+218], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+219], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+220], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+221], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+222], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+223], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+224], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+225], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+226], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+227], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v[vgprValuC+136], v[vgprValuC+136]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+137], v[vgprValuC+137]   // convert C to fp16
v_pack_b32_f16 v136, v[vgprValuC+136], v[vgprValuC+137] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+138], v[vgprValuC+138]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+139], v[vgprValuC+139]   // convert C to fp16
v_pack_b32_f16 v137, v[vgprValuC+138], v[vgprValuC+139] // Pack with neighbor
buffer_store_b64 v[136:137], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+140], v[vgprValuC+140]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+141], v[vgprValuC+141]   // convert C to fp16
v_pack_b32_f16 v140, v[vgprValuC+140], v[vgprValuC+141] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+142], v[vgprValuC+142]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+143], v[vgprValuC+143]   // convert C to fp16
v_pack_b32_f16 v141, v[vgprValuC+142], v[vgprValuC+143] // Pack with neighbor
buffer_store_b64 v[140:141], v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+144], v[vgprValuC+144]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+145], v[vgprValuC+145]   // convert C to fp16
v_pack_b32_f16 v144, v[vgprValuC+144], v[vgprValuC+145] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+146], v[vgprValuC+146]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+147], v[vgprValuC+147]   // convert C to fp16
v_pack_b32_f16 v145, v[vgprValuC+146], v[vgprValuC+147] // Pack with neighbor
buffer_store_b64 v[144:145], v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+148], v[vgprValuC+148]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+149], v[vgprValuC+149]   // convert C to fp16
v_pack_b32_f16 v148, v[vgprValuC+148], v[vgprValuC+149] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+150], v[vgprValuC+150]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+151], v[vgprValuC+151]   // convert C to fp16
v_pack_b32_f16 v149, v[vgprValuC+150], v[vgprValuC+151] // Pack with neighbor
buffer_store_b64 v[148:149], v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+152], v[vgprValuC+152]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+153], v[vgprValuC+153]   // convert C to fp16
v_pack_b32_f16 v152, v[vgprValuC+152], v[vgprValuC+153] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+154], v[vgprValuC+154]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+155], v[vgprValuC+155]   // convert C to fp16
v_pack_b32_f16 v153, v[vgprValuC+154], v[vgprValuC+155] // Pack with neighbor
buffer_store_b64 v[152:153], v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+156], v[vgprValuC+156]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+157], v[vgprValuC+157]   // convert C to fp16
v_pack_b32_f16 v156, v[vgprValuC+156], v[vgprValuC+157] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+158], v[vgprValuC+158]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+159], v[vgprValuC+159]   // convert C to fp16
v_pack_b32_f16 v157, v[vgprValuC+158], v[vgprValuC+159] // Pack with neighbor
buffer_store_b64 v[156:157], v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+160], v[vgprValuC+160]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+161], v[vgprValuC+161]   // convert C to fp16
v_pack_b32_f16 v160, v[vgprValuC+160], v[vgprValuC+161] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+162], v[vgprValuC+162]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+163], v[vgprValuC+163]   // convert C to fp16
v_pack_b32_f16 v161, v[vgprValuC+162], v[vgprValuC+163] // Pack with neighbor
buffer_store_b64 v[160:161], v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+164], v[vgprValuC+164]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+165], v[vgprValuC+165]   // convert C to fp16
v_pack_b32_f16 v164, v[vgprValuC+164], v[vgprValuC+165] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+166], v[vgprValuC+166]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+167], v[vgprValuC+167]   // convert C to fp16
v_pack_b32_f16 v165, v[vgprValuC+166], v[vgprValuC+167] // Pack with neighbor
buffer_store_b64 v[164:165], v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+168], v[vgprValuC+168]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+169], v[vgprValuC+169]   // convert C to fp16
v_pack_b32_f16 v168, v[vgprValuC+168], v[vgprValuC+169] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+170], v[vgprValuC+170]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+171], v[vgprValuC+171]   // convert C to fp16
v_pack_b32_f16 v169, v[vgprValuC+170], v[vgprValuC+171] // Pack with neighbor
buffer_store_b64 v[168:169], v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+172], v[vgprValuC+172]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+173], v[vgprValuC+173]   // convert C to fp16
v_pack_b32_f16 v172, v[vgprValuC+172], v[vgprValuC+173] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+174], v[vgprValuC+174]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+175], v[vgprValuC+175]   // convert C to fp16
v_pack_b32_f16 v173, v[vgprValuC+174], v[vgprValuC+175] // Pack with neighbor
buffer_store_b64 v[172:173], v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+176], v[vgprValuC+176]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+177], v[vgprValuC+177]   // convert C to fp16
v_pack_b32_f16 v176, v[vgprValuC+176], v[vgprValuC+177] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+178], v[vgprValuC+178]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+179], v[vgprValuC+179]   // convert C to fp16
v_pack_b32_f16 v177, v[vgprValuC+178], v[vgprValuC+179] // Pack with neighbor
buffer_store_b64 v[176:177], v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+180], v[vgprValuC+180]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+181], v[vgprValuC+181]   // convert C to fp16
v_pack_b32_f16 v180, v[vgprValuC+180], v[vgprValuC+181] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+182], v[vgprValuC+182]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+183], v[vgprValuC+183]   // convert C to fp16
v_pack_b32_f16 v181, v[vgprValuC+182], v[vgprValuC+183] // Pack with neighbor
buffer_store_b64 v[180:181], v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+184], v[vgprValuC+184]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+185], v[vgprValuC+185]   // convert C to fp16
v_pack_b32_f16 v184, v[vgprValuC+184], v[vgprValuC+185] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+186], v[vgprValuC+186]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+187], v[vgprValuC+187]   // convert C to fp16
v_pack_b32_f16 v185, v[vgprValuC+186], v[vgprValuC+187] // Pack with neighbor
buffer_store_b64 v[184:185], v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+188], v[vgprValuC+188]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+189], v[vgprValuC+189]   // convert C to fp16
v_pack_b32_f16 v188, v[vgprValuC+188], v[vgprValuC+189] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+190], v[vgprValuC+190]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+191], v[vgprValuC+191]   // convert C to fp16
v_pack_b32_f16 v189, v[vgprValuC+190], v[vgprValuC+191] // Pack with neighbor
buffer_store_b64 v[188:189], v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+192], v[vgprValuC+192]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+193], v[vgprValuC+193]   // convert C to fp16
v_pack_b32_f16 v192, v[vgprValuC+192], v[vgprValuC+193] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+194], v[vgprValuC+194]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+195], v[vgprValuC+195]   // convert C to fp16
v_pack_b32_f16 v193, v[vgprValuC+194], v[vgprValuC+195] // Pack with neighbor
buffer_store_b64 v[192:193], v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+196], v[vgprValuC+196]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+197], v[vgprValuC+197]   // convert C to fp16
v_pack_b32_f16 v196, v[vgprValuC+196], v[vgprValuC+197] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+198], v[vgprValuC+198]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+199], v[vgprValuC+199]   // convert C to fp16
v_pack_b32_f16 v197, v[vgprValuC+198], v[vgprValuC+199] // Pack with neighbor
buffer_store_b64 v[196:197], v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+204], v[vgprValuC+204]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+205], v[vgprValuC+205]   // convert C to fp16
v_pack_b32_f16 v204, v[vgprValuC+204], v[vgprValuC+205] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+206], v[vgprValuC+206]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+207], v[vgprValuC+207]   // convert C to fp16
v_pack_b32_f16 v205, v[vgprValuC+206], v[vgprValuC+207] // Pack with neighbor
buffer_store_b64 v[204:205], v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+208], v[vgprValuC+208]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+209], v[vgprValuC+209]   // convert C to fp16
v_pack_b32_f16 v208, v[vgprValuC+208], v[vgprValuC+209] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+210], v[vgprValuC+210]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+211], v[vgprValuC+211]   // convert C to fp16
v_pack_b32_f16 v209, v[vgprValuC+210], v[vgprValuC+211] // Pack with neighbor
buffer_store_b64 v[208:209], v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+212], v[vgprValuC+212]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+213], v[vgprValuC+213]   // convert C to fp16
v_pack_b32_f16 v212, v[vgprValuC+212], v[vgprValuC+213] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+214], v[vgprValuC+214]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+215], v[vgprValuC+215]   // convert C to fp16
v_pack_b32_f16 v213, v[vgprValuC+214], v[vgprValuC+215] // Pack with neighbor
buffer_store_b64 v[212:213], v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+216], v[vgprValuC+216]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+217], v[vgprValuC+217]   // convert C to fp16
v_pack_b32_f16 v216, v[vgprValuC+216], v[vgprValuC+217] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+218], v[vgprValuC+218]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+219], v[vgprValuC+219]   // convert C to fp16
v_pack_b32_f16 v217, v[vgprValuC+218], v[vgprValuC+219] // Pack with neighbor
buffer_store_b64 v[216:217], v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+220], v[vgprValuC+220]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+221], v[vgprValuC+221]   // convert C to fp16
v_pack_b32_f16 v220, v[vgprValuC+220], v[vgprValuC+221] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+222], v[vgprValuC+222]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+223], v[vgprValuC+223]   // convert C to fp16
v_pack_b32_f16 v221, v[vgprValuC+222], v[vgprValuC+223] // Pack with neighbor
buffer_store_b64 v[220:221], v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+224], v[vgprValuC+224]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+225], v[vgprValuC+225]   // convert C to fp16
v_pack_b32_f16 v224, v[vgprValuC+224], v[vgprValuC+225] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+226], v[vgprValuC+226]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+227], v[vgprValuC+227]   // convert C to fp16
v_pack_b32_f16 v225, v[vgprValuC+226], v[vgprValuC+227] // Pack with neighbor
buffer_store_b64 v[224:225], v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,6,2,0:vw4); (0,7,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (0,2,3,0:vw4); (0,3,3,0:vw4); (0,4,3,0:vw4); (0,5,3,0:vw4); (0,6,3,0:vw4); (0,7,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v135, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, v134, v135, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v176, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, v134, v176, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v177, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v177, v134, v177, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v178, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, v134, v178, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v179, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v179, v134, v179, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v180, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, v134, v180, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v181, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v181, v134, v181, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v182, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, v134, v182, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v183, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v183, v134, v183, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v184, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, v134, v184, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 6, 2, 0), (0, 7, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0), (0, 4, 3, 0), (0, 5, 3, 0), (0, 6, 3, 0), (0, 7, 3, 0)] */
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v[vgprValuC+136], v[vgprValuC+136]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+137], v[vgprValuC+137]   // convert C to fp16
v_pack_b32_f16 v136, v[vgprValuC+136], v[vgprValuC+137] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+138], v[vgprValuC+138]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+139], v[vgprValuC+139]   // convert C to fp16
v_pack_b32_f16 v137, v[vgprValuC+138], v[vgprValuC+139] // Pack with neighbor
buffer_store_b64 v[136:137], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+140], v[vgprValuC+140]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+141], v[vgprValuC+141]   // convert C to fp16
v_pack_b32_f16 v140, v[vgprValuC+140], v[vgprValuC+141] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+142], v[vgprValuC+142]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+143], v[vgprValuC+143]   // convert C to fp16
v_pack_b32_f16 v141, v[vgprValuC+142], v[vgprValuC+143] // Pack with neighbor
buffer_store_b64 v[140:141], v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+144], v[vgprValuC+144]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+145], v[vgprValuC+145]   // convert C to fp16
v_pack_b32_f16 v144, v[vgprValuC+144], v[vgprValuC+145] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+146], v[vgprValuC+146]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+147], v[vgprValuC+147]   // convert C to fp16
v_pack_b32_f16 v145, v[vgprValuC+146], v[vgprValuC+147] // Pack with neighbor
buffer_store_b64 v[144:145], v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+148], v[vgprValuC+148]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+149], v[vgprValuC+149]   // convert C to fp16
v_pack_b32_f16 v148, v[vgprValuC+148], v[vgprValuC+149] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+150], v[vgprValuC+150]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+151], v[vgprValuC+151]   // convert C to fp16
v_pack_b32_f16 v149, v[vgprValuC+150], v[vgprValuC+151] // Pack with neighbor
buffer_store_b64 v[148:149], v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+152], v[vgprValuC+152]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+153], v[vgprValuC+153]   // convert C to fp16
v_pack_b32_f16 v152, v[vgprValuC+152], v[vgprValuC+153] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+154], v[vgprValuC+154]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+155], v[vgprValuC+155]   // convert C to fp16
v_pack_b32_f16 v153, v[vgprValuC+154], v[vgprValuC+155] // Pack with neighbor
buffer_store_b64 v[152:153], v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+156], v[vgprValuC+156]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+157], v[vgprValuC+157]   // convert C to fp16
v_pack_b32_f16 v156, v[vgprValuC+156], v[vgprValuC+157] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+158], v[vgprValuC+158]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+159], v[vgprValuC+159]   // convert C to fp16
v_pack_b32_f16 v157, v[vgprValuC+158], v[vgprValuC+159] // Pack with neighbor
buffer_store_b64 v[156:157], v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+160], v[vgprValuC+160]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+161], v[vgprValuC+161]   // convert C to fp16
v_pack_b32_f16 v160, v[vgprValuC+160], v[vgprValuC+161] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+162], v[vgprValuC+162]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+163], v[vgprValuC+163]   // convert C to fp16
v_pack_b32_f16 v161, v[vgprValuC+162], v[vgprValuC+163] // Pack with neighbor
buffer_store_b64 v[160:161], v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+164], v[vgprValuC+164]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+165], v[vgprValuC+165]   // convert C to fp16
v_pack_b32_f16 v164, v[vgprValuC+164], v[vgprValuC+165] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+166], v[vgprValuC+166]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+167], v[vgprValuC+167]   // convert C to fp16
v_pack_b32_f16 v165, v[vgprValuC+166], v[vgprValuC+167] // Pack with neighbor
buffer_store_b64 v[164:165], v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+168], v[vgprValuC+168]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+169], v[vgprValuC+169]   // convert C to fp16
v_pack_b32_f16 v168, v[vgprValuC+168], v[vgprValuC+169] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+170], v[vgprValuC+170]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+171], v[vgprValuC+171]   // convert C to fp16
v_pack_b32_f16 v169, v[vgprValuC+170], v[vgprValuC+171] // Pack with neighbor
buffer_store_b64 v[168:169], v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v[vgprValuC+172], v[vgprValuC+172]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+173], v[vgprValuC+173]   // convert C to fp16
v_pack_b32_f16 v172, v[vgprValuC+172], v[vgprValuC+173] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+174], v[vgprValuC+174]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+175], v[vgprValuC+175]   // convert C to fp16
v_pack_b32_f16 v173, v[vgprValuC+174], v[vgprValuC+175] // Pack with neighbor
buffer_store_b64 v[172:173], v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_Else_1:
label_GW_B0_FD0_VW1_Else_1:
label_GW_B0_FD0_VW1_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=60 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,2,0,0:vw1); (0,2,0,1:vw1); (0,2,0,2:vw1); (0,2,0,3:vw1); (0,3,0,0:vw1); (0,3,0,1:vw1); (0,3,0,2:vw1); (0,3,0,3:vw1); (0,4,0,0:vw1); (0,4,0,1:vw1); (0,4,0,2:vw1); (0,4,0,3:vw1); (0,5,0,0:vw1); (0,5,0,1:vw1); (0,5,0,2:vw1); (0,5,0,3:vw1); (0,6,0,0:vw1); (0,6,0,1:vw1); (0,6,0,2:vw1); (0,6,0,3:vw1); (0,7,0,0:vw1); (0,7,0,1:vw1); (0,7,0,2:vw1); (0,7,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1); (0,2,1,0:vw1); (0,2,1,1:vw1); (0,2,1,2:vw1); (0,2,1,3:vw1); (0,3,1,0:vw1); (0,3,1,1:vw1); (0,3,1,2:vw1); (0,3,1,3:vw1); (0,4,1,0:vw1); (0,4,1,1:vw1); (0,4,1,2:vw1); (0,4,1,3:vw1); (0,5,1,0:vw1); (0,5,1,1:vw1); (0,5,1,2:vw1); (0,5,1,3:vw1); (0,6,1,0:vw1); (0,6,1,1:vw1); (0,6,1,2:vw1); (0,6,1,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v195, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v195, v134, v195, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v196, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v197, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v197, v134, v197, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v198, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v199, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, v134, v199, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v202, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, v134, v202, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v204, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v204, v134, v204, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v205, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v206, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v206, v134, v206, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v207, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v208, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, v134, v208, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v210, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v210, v134, v210, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v211, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v212, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v212, v134, v212, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v213, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v214, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, v134, v214, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v216, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v216, v134, v216, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v217, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v218, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v218, v134, v218, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v219, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v220, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v220, v134, v220, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v222, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v222, v134, v222, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v223, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v224, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v224, v134, v224, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v225, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v226, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, v134, v226, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v228, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v228, v134, v228, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v230, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v230, v134, v230, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v232, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v234, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v234, v134, v234, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v236, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v236, v134, v236, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v238, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v240, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v240, v134, v240, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v242, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v242, v134, v242, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v244, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v246, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v246, v134, v246, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v247, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v248, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v248, v134, v248, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v249, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v250, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, v134, v250, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v251, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v252, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v252, v134, v252, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v253, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v254, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v254, v134, v254, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v255, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 0, 3), (0, 3, 0, 0), (0, 3, 0, 1), (0, 3, 0, 2), (0, 3, 0, 3), (0, 4, 0, 0), (0, 4, 0, 1), (0, 4, 0, 2), (0, 4, 0, 3), (0, 5, 0, 0), (0, 5, 0, 1), (0, 5, 0, 2), (0, 5, 0, 3), (0, 6, 0, 0), (0, 6, 0, 1), (0, 6, 0, 2), (0, 6, 0, 3), (0, 7, 0, 0), (0, 7, 0, 1), (0, 7, 0, 2), (0, 7, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 1, 2), (0, 2, 1, 3), (0, 3, 1, 0), (0, 3, 1, 1), (0, 3, 1, 2), (0, 3, 1, 3), (0, 4, 1, 0), (0, 4, 1, 1), (0, 4, 1, 2), (0, 4, 1, 3), (0, 5, 1, 0), (0, 5, 1, 1), (0, 5, 1, 2), (0, 5, 1, 3), (0, 6, 1, 0), (0, 6, 1, 1), (0, 6, 1, 2), (0, 6, 1, 3)] */
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
buffer_store_b16 v143, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
buffer_store_b16 v145, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
buffer_store_b16 v147, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
buffer_store_b16 v149, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
buffer_store_b16 v151, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
buffer_store_b16 v153, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
buffer_store_b16 v155, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
buffer_store_b16 v157, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
buffer_store_b16 v159, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
buffer_store_b16 v161, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v163, v[vgprValuC+163]               // convert C to fp16
buffer_store_b16 v163, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v164, v[vgprValuC+164]               // convert C to fp16
buffer_store_b16 v164, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v165, v[vgprValuC+165]               // convert C to fp16
buffer_store_b16 v165, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v166, v[vgprValuC+166]               // convert C to fp16
buffer_store_b16 v166, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v167, v[vgprValuC+167]               // convert C to fp16
buffer_store_b16 v167, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v168, v[vgprValuC+168]               // convert C to fp16
buffer_store_b16 v168, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v169, v[vgprValuC+169]               // convert C to fp16
buffer_store_b16 v169, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v170, v[vgprValuC+170]               // convert C to fp16
buffer_store_b16 v170, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v171, v[vgprValuC+171]               // convert C to fp16
buffer_store_b16 v171, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v172, v[vgprValuC+172]               // convert C to fp16
buffer_store_b16 v172, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v173, v[vgprValuC+173]               // convert C to fp16
buffer_store_b16 v173, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v174, v[vgprValuC+174]               // convert C to fp16
buffer_store_b16 v174, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v175, v[vgprValuC+175]               // convert C to fp16
buffer_store_b16 v175, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v176, v[vgprValuC+176]               // convert C to fp16
buffer_store_b16 v176, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v177, v[vgprValuC+177]               // convert C to fp16
buffer_store_b16 v177, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v178, v[vgprValuC+178]               // convert C to fp16
buffer_store_b16 v178, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v179, v[vgprValuC+179]               // convert C to fp16
buffer_store_b16 v179, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v180, v[vgprValuC+180]               // convert C to fp16
buffer_store_b16 v180, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v181, v[vgprValuC+181]               // convert C to fp16
buffer_store_b16 v181, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v182, v[vgprValuC+182]               // convert C to fp16
buffer_store_b16 v182, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v183, v[vgprValuC+183]               // convert C to fp16
buffer_store_b16 v183, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v184, v[vgprValuC+184]               // convert C to fp16
buffer_store_b16 v184, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v185, v[vgprValuC+185]               // convert C to fp16
buffer_store_b16 v185, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v186, v[vgprValuC+186]               // convert C to fp16
buffer_store_b16 v186, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v187, v[vgprValuC+187]               // convert C to fp16
buffer_store_b16 v187, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v188, v[vgprValuC+188]               // convert C to fp16
buffer_store_b16 v188, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v189, v[vgprValuC+189]               // convert C to fp16
buffer_store_b16 v189, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v190, v[vgprValuC+190]               // convert C to fp16
buffer_store_b16 v190, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v191, v[vgprValuC+191]               // convert C to fp16
buffer_store_b16 v191, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v192, v[vgprValuC+192]               // convert C to fp16
buffer_store_b16 v192, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v193, v[vgprValuC+193]               // convert C to fp16
buffer_store_b16 v193, v254, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v194, v[vgprValuC+194]               // convert C to fp16
buffer_store_b16 v194, v255, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,7,1,0:vw1); (0,7,1,1:vw1); (0,7,1,2:vw1); (0,7,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,2,2,0:vw1); (0,2,2,1:vw1); (0,2,2,2:vw1); (0,2,2,3:vw1); (0,3,2,0:vw1); (0,3,2,1:vw1); (0,3,2,2:vw1); (0,3,2,3:vw1); (0,4,2,0:vw1); (0,4,2,1:vw1); (0,4,2,2:vw1); (0,4,2,3:vw1); (0,5,2,0:vw1); (0,5,2,1:vw1); (0,5,2,2:vw1); (0,5,2,3:vw1); (0,6,2,0:vw1); (0,6,2,1:vw1); (0,6,2,2:vw1); (0,6,2,3:vw1); (0,7,2,0:vw1); (0,7,2,1:vw1); (0,7,2,2:vw1); (0,7,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1); (0,2,3,0:vw1); (0,2,3,1:vw1); (0,2,3,2:vw1); (0,2,3,3:vw1); (0,3,3,0:vw1); (0,3,3,1:vw1); (0,3,3,2:vw1); (0,3,3,3:vw1); (0,4,3,0:vw1); (0,4,3,1:vw1); (0,4,3,2:vw1); (0,4,3,3:vw1); (0,5,3,0:vw1); (0,5,3,1:vw1); (0,5,3,2:vw1); (0,5,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v195, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v195, v134, v195, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v196, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v197, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v197, v134, v197, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v198, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v199, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v199, v134, v199, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v202, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v202, v134, v202, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v204, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v204, v134, v204, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v205, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v206, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v206, v134, v206, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v207, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v208, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, v134, v208, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v210, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v210, v134, v210, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v211, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v212, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v212, v134, v212, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v213, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v214, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, v134, v214, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v216, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v216, v134, v216, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v217, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v218, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v218, v134, v218, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v219, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v220, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v220, v134, v220, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v222, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v222, v134, v222, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v223, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v224, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v224, v134, v224, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v225, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v226, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, v134, v226, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v228, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v228, v134, v228, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v230, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v230, v134, v230, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v232, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v234, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v234, v134, v234, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v236, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v236, v134, v236, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v238, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v240, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v240, v134, v240, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v242, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v242, v134, v242, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v244, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v246, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v246, v134, v246, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v247, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v248, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v248, v134, v248, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v249, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v250, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v250, v134, v250, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v251, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v252, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v252, v134, v252, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v253, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v254, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v254, v134, v254, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v255, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 7, 1, 0), (0, 7, 1, 1), (0, 7, 1, 2), (0, 7, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 2, 2, 0), (0, 2, 2, 1), (0, 2, 2, 2), (0, 2, 2, 3), (0, 3, 2, 0), (0, 3, 2, 1), (0, 3, 2, 2), (0, 3, 2, 3), (0, 4, 2, 0), (0, 4, 2, 1), (0, 4, 2, 2), (0, 4, 2, 3), (0, 5, 2, 0), (0, 5, 2, 1), (0, 5, 2, 2), (0, 5, 2, 3), (0, 6, 2, 0), (0, 6, 2, 1), (0, 6, 2, 2), (0, 6, 2, 3), (0, 7, 2, 0), (0, 7, 2, 1), (0, 7, 2, 2), (0, 7, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3), (0, 2, 3, 0), (0, 2, 3, 1), (0, 2, 3, 2), (0, 2, 3, 3), (0, 3, 3, 0), (0, 3, 3, 1), (0, 3, 3, 2), (0, 3, 3, 3), (0, 4, 3, 0), (0, 4, 3, 1), (0, 4, 3, 2), (0, 4, 3, 3), (0, 5, 3, 0), (0, 5, 3, 1), (0, 5, 3, 2), (0, 5, 3, 3)] */
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v195, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v197, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v199, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v202, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
buffer_store_b16 v143, v204, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
buffer_store_b16 v145, v206, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
buffer_store_b16 v147, v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
buffer_store_b16 v149, v210, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
buffer_store_b16 v151, v212, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
buffer_store_b16 v153, v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
buffer_store_b16 v155, v216, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
buffer_store_b16 v157, v218, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
buffer_store_b16 v159, v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
buffer_store_b16 v161, v222, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v163, v[vgprValuC+163]               // convert C to fp16
buffer_store_b16 v163, v224, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v164, v[vgprValuC+164]               // convert C to fp16
buffer_store_b16 v164, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v165, v[vgprValuC+165]               // convert C to fp16
buffer_store_b16 v165, v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v166, v[vgprValuC+166]               // convert C to fp16
buffer_store_b16 v166, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v167, v[vgprValuC+167]               // convert C to fp16
buffer_store_b16 v167, v228, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v168, v[vgprValuC+168]               // convert C to fp16
buffer_store_b16 v168, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v169, v[vgprValuC+169]               // convert C to fp16
buffer_store_b16 v169, v230, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v170, v[vgprValuC+170]               // convert C to fp16
buffer_store_b16 v170, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v171, v[vgprValuC+171]               // convert C to fp16
buffer_store_b16 v171, v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v172, v[vgprValuC+172]               // convert C to fp16
buffer_store_b16 v172, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v173, v[vgprValuC+173]               // convert C to fp16
buffer_store_b16 v173, v234, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v174, v[vgprValuC+174]               // convert C to fp16
buffer_store_b16 v174, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v175, v[vgprValuC+175]               // convert C to fp16
buffer_store_b16 v175, v236, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v176, v[vgprValuC+176]               // convert C to fp16
buffer_store_b16 v176, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v177, v[vgprValuC+177]               // convert C to fp16
buffer_store_b16 v177, v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v178, v[vgprValuC+178]               // convert C to fp16
buffer_store_b16 v178, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v179, v[vgprValuC+179]               // convert C to fp16
buffer_store_b16 v179, v240, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v180, v[vgprValuC+180]               // convert C to fp16
buffer_store_b16 v180, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v181, v[vgprValuC+181]               // convert C to fp16
buffer_store_b16 v181, v242, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v182, v[vgprValuC+182]               // convert C to fp16
buffer_store_b16 v182, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v183, v[vgprValuC+183]               // convert C to fp16
buffer_store_b16 v183, v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v184, v[vgprValuC+184]               // convert C to fp16
buffer_store_b16 v184, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v185, v[vgprValuC+185]               // convert C to fp16
buffer_store_b16 v185, v246, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v186, v[vgprValuC+186]               // convert C to fp16
buffer_store_b16 v186, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v187, v[vgprValuC+187]               // convert C to fp16
buffer_store_b16 v187, v248, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v188, v[vgprValuC+188]               // convert C to fp16
buffer_store_b16 v188, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v189, v[vgprValuC+189]               // convert C to fp16
buffer_store_b16 v189, v250, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v190, v[vgprValuC+190]               // convert C to fp16
buffer_store_b16 v190, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v191, v[vgprValuC+191]               // convert C to fp16
buffer_store_b16 v191, v252, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v192, v[vgprValuC+192]               // convert C to fp16
buffer_store_b16 v192, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v193, v[vgprValuC+193]               // convert C to fp16
buffer_store_b16 v193, v254, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v194, v[vgprValuC+194]               // convert C to fp16
buffer_store_b16 v194, v255, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,6,3,0:vw1); (0,6,3,1:vw1); (0,6,3,2:vw1); (0,6,3,3:vw1); (0,7,3,0:vw1); (0,7,3,1:vw1); (0,7,3,2:vw1); (0,7,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v143, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v143, v134, v143, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v144, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, v134, v144, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v145, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v145, v134, v145, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v146, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, v134, v146, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v147, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v147, v134, v147, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v148, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, v134, v148, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v149, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v149, v134, v149, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v150, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, v134, v150, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 6, 3, 0), (0, 6, 3, 1), (0, 6, 3, 2), (0, 6, 3, 3), (0, 7, 3, 0), (0, 7, 3, 1), (0, 7, 3, 2), (0, 7, 3, 3)] */
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1:
label_GW_B1_FD0:
s_and_b32 s40, 127, s[sgprSizeI]                   // s40 = s[sgprSizeI] % 128
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_Else            // jump if edges required
s_and_b32 s40, 127, s[sgprSizeJ]                   // s40 = s[sgprSizeJ] % 128
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_Then            // jump if edges required
label_GW_B1_FD0_VW4_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=18 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,2,0,0:vw4); (0,3,0,0:vw4); (0,4,0,0:vw4); (0,5,0,0:vw4); (0,6,0,0:vw4); (0,7,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,2,1,0:vw4); (0,3,1,0:vw4); (0,4,1,0:vw4); (0,5,1,0:vw4); (0,6,1,0:vw4); (0,7,1,0:vw4); (0,0,2,0:vw4); (0,1,2,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v136, v130, v128, 0x1               // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128
buffer_load_b64 v[138:139], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_b64 v[202:203], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:16 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_b64 v[216:217], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_b64 v[218:219], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:48 // load C
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
buffer_load_b64 v[220:221], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
buffer_load_b64 v[222:223], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:80 // load C
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
buffer_load_b64 v[224:225], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:96 // load C
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
buffer_load_b64 v[226:227], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:112 // load C
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_b64 v[228:229], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
buffer_load_b64 v[230:231], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:16 // load C
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
buffer_load_b64 v[232:233], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
buffer_load_b64 v[234:235], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:48 // load C
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
buffer_load_b64 v[236:237], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
buffer_load_b64 v[238:239], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:80 // load C
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
buffer_load_b64 v[240:241], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:96 // load C
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
buffer_load_b64 v[242:243], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:112 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_b64 v[244:245], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
buffer_load_b64 v[246:247], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:16 // load C
v_add_lshl_u32 v135, v131, v128, 0x1               // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=128, coord0Vgpr=128

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0), (0, 0, 2, 0), (0, 1, 2, 0)] */
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+204], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+205], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+206], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+207], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+208], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+209], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+210], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+211], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+212], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+213], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+214], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+215], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(17)                                // vlcnt(17) = 18 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v138, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v138, v[vgprValuC+141] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v139, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v139, v[vgprValuC+143] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+140], v[vgprValuC+140]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+141], v[vgprValuC+141]   // convert C to fp16
v_pack_b32_f16 v140, v[vgprValuC+140], v[vgprValuC+141] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+142], v[vgprValuC+142]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+143], v[vgprValuC+143]   // convert C to fp16
v_pack_b32_f16 v141, v[vgprValuC+142], v[vgprValuC+143] // Pack with neighbor
buffer_store_b64 v[140:141], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(16)                                // vlcnt(16) = 18 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v202, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v202, v[vgprValuC+145] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v203, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v203, v[vgprValuC+147] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+144], v[vgprValuC+144]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+145], v[vgprValuC+145]   // convert C to fp16
v_pack_b32_f16 v144, v[vgprValuC+144], v[vgprValuC+145] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+146], v[vgprValuC+146]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+147], v[vgprValuC+147]   // convert C to fp16
v_pack_b32_f16 v145, v[vgprValuC+146], v[vgprValuC+147] // Pack with neighbor
buffer_store_b64 v[144:145], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D

s_waitcnt vmcnt(15)                                // vlcnt(15) = 18 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v216, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v216, v[vgprValuC+149] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v217, v[vgprValuC+150] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v217, v[vgprValuC+151] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+148], v[vgprValuC+148]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+149], v[vgprValuC+149]   // convert C to fp16
v_pack_b32_f16 v148, v[vgprValuC+148], v[vgprValuC+149] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+150], v[vgprValuC+150]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+151], v[vgprValuC+151]   // convert C to fp16
v_pack_b32_f16 v149, v[vgprValuC+150], v[vgprValuC+151] // Pack with neighbor
buffer_store_b64 v[148:149], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(14)                                // vlcnt(14) = 18 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v218, v[vgprValuC+152] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v218, v[vgprValuC+153] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v219, v[vgprValuC+154] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v219, v[vgprValuC+155] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+152], v[vgprValuC+152]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+153], v[vgprValuC+153]   // convert C to fp16
v_pack_b32_f16 v152, v[vgprValuC+152], v[vgprValuC+153] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+154], v[vgprValuC+154]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+155], v[vgprValuC+155]   // convert C to fp16
v_pack_b32_f16 v153, v[vgprValuC+154], v[vgprValuC+155] // Pack with neighbor
buffer_store_b64 v[152:153], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D

s_waitcnt vmcnt(13)                                // vlcnt(13) = 18 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v220, v[vgprValuC+156] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v220, v[vgprValuC+157] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v221, v[vgprValuC+158] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v221, v[vgprValuC+159] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+156], v[vgprValuC+156]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+157], v[vgprValuC+157]   // convert C to fp16
v_pack_b32_f16 v156, v[vgprValuC+156], v[vgprValuC+157] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+158], v[vgprValuC+158]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+159], v[vgprValuC+159]   // convert C to fp16
v_pack_b32_f16 v157, v[vgprValuC+158], v[vgprValuC+159] // Pack with neighbor
buffer_store_b64 v[156:157], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 18 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v222, v[vgprValuC+160] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v222, v[vgprValuC+161] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v223, v[vgprValuC+162] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+163], s[sgprBeta], v223, v[vgprValuC+163] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+160], v[vgprValuC+160]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+161], v[vgprValuC+161]   // convert C to fp16
v_pack_b32_f16 v160, v[vgprValuC+160], v[vgprValuC+161] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+162], v[vgprValuC+162]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+163], v[vgprValuC+163]   // convert C to fp16
v_pack_b32_f16 v161, v[vgprValuC+162], v[vgprValuC+163] // Pack with neighbor
buffer_store_b64 v[160:161], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 18 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+164], s[sgprBeta], v224, v[vgprValuC+164] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+165], s[sgprBeta], v224, v[vgprValuC+165] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+166], s[sgprBeta], v225, v[vgprValuC+166] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+167], s[sgprBeta], v225, v[vgprValuC+167] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+164], v[vgprValuC+164]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+165], v[vgprValuC+165]   // convert C to fp16
v_pack_b32_f16 v164, v[vgprValuC+164], v[vgprValuC+165] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+166], v[vgprValuC+166]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+167], v[vgprValuC+167]   // convert C to fp16
v_pack_b32_f16 v165, v[vgprValuC+166], v[vgprValuC+167] // Pack with neighbor
buffer_store_b64 v[164:165], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 18 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+168], s[sgprBeta], v226, v[vgprValuC+168] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+169], s[sgprBeta], v226, v[vgprValuC+169] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+170], s[sgprBeta], v227, v[vgprValuC+170] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+171], s[sgprBeta], v227, v[vgprValuC+171] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+168], v[vgprValuC+168]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+169], v[vgprValuC+169]   // convert C to fp16
v_pack_b32_f16 v168, v[vgprValuC+168], v[vgprValuC+169] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+170], v[vgprValuC+170]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+171], v[vgprValuC+171]   // convert C to fp16
v_pack_b32_f16 v169, v[vgprValuC+170], v[vgprValuC+171] // Pack with neighbor
buffer_store_b64 v[168:169], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 18 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+172], s[sgprBeta], v228, v[vgprValuC+172] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+173], s[sgprBeta], v228, v[vgprValuC+173] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+174], s[sgprBeta], v229, v[vgprValuC+174] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+175], s[sgprBeta], v229, v[vgprValuC+175] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+172], v[vgprValuC+172]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+173], v[vgprValuC+173]   // convert C to fp16
v_pack_b32_f16 v172, v[vgprValuC+172], v[vgprValuC+173] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+174], v[vgprValuC+174]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+175], v[vgprValuC+175]   // convert C to fp16
v_pack_b32_f16 v173, v[vgprValuC+174], v[vgprValuC+175] // Pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b64 v[172:173], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 18 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+176], s[sgprBeta], v230, v[vgprValuC+176] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+177], s[sgprBeta], v230, v[vgprValuC+177] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+178], s[sgprBeta], v231, v[vgprValuC+178] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+179], s[sgprBeta], v231, v[vgprValuC+179] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+176], v[vgprValuC+176]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+177], v[vgprValuC+177]   // convert C to fp16
v_pack_b32_f16 v176, v[vgprValuC+176], v[vgprValuC+177] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+178], v[vgprValuC+178]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+179], v[vgprValuC+179]   // convert C to fp16
v_pack_b32_f16 v177, v[vgprValuC+178], v[vgprValuC+179] // Pack with neighbor
buffer_store_b64 v[176:177], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 18 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+180], s[sgprBeta], v232, v[vgprValuC+180] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+181], s[sgprBeta], v232, v[vgprValuC+181] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+182], s[sgprBeta], v233, v[vgprValuC+182] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+183], s[sgprBeta], v233, v[vgprValuC+183] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+180], v[vgprValuC+180]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+181], v[vgprValuC+181]   // convert C to fp16
v_pack_b32_f16 v180, v[vgprValuC+180], v[vgprValuC+181] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+182], v[vgprValuC+182]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+183], v[vgprValuC+183]   // convert C to fp16
v_pack_b32_f16 v181, v[vgprValuC+182], v[vgprValuC+183] // Pack with neighbor
buffer_store_b64 v[180:181], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 18 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+184], s[sgprBeta], v234, v[vgprValuC+184] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+185], s[sgprBeta], v234, v[vgprValuC+185] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+186], s[sgprBeta], v235, v[vgprValuC+186] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+187], s[sgprBeta], v235, v[vgprValuC+187] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+184], v[vgprValuC+184]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+185], v[vgprValuC+185]   // convert C to fp16
v_pack_b32_f16 v184, v[vgprValuC+184], v[vgprValuC+185] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+186], v[vgprValuC+186]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+187], v[vgprValuC+187]   // convert C to fp16
v_pack_b32_f16 v185, v[vgprValuC+186], v[vgprValuC+187] // Pack with neighbor
buffer_store_b64 v[184:185], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 18 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+188], s[sgprBeta], v236, v[vgprValuC+188] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+189], s[sgprBeta], v236, v[vgprValuC+189] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+190], s[sgprBeta], v237, v[vgprValuC+190] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+191], s[sgprBeta], v237, v[vgprValuC+191] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+188], v[vgprValuC+188]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+189], v[vgprValuC+189]   // convert C to fp16
v_pack_b32_f16 v188, v[vgprValuC+188], v[vgprValuC+189] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+190], v[vgprValuC+190]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+191], v[vgprValuC+191]   // convert C to fp16
v_pack_b32_f16 v189, v[vgprValuC+190], v[vgprValuC+191] // Pack with neighbor
buffer_store_b64 v[188:189], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 18 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+192], s[sgprBeta], v238, v[vgprValuC+192] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+193], s[sgprBeta], v238, v[vgprValuC+193] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+194], s[sgprBeta], v239, v[vgprValuC+194] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+195], s[sgprBeta], v239, v[vgprValuC+195] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+192], v[vgprValuC+192]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+193], v[vgprValuC+193]   // convert C to fp16
v_pack_b32_f16 v192, v[vgprValuC+192], v[vgprValuC+193] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+194], v[vgprValuC+194]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+195], v[vgprValuC+195]   // convert C to fp16
v_pack_b32_f16 v193, v[vgprValuC+194], v[vgprValuC+195] // Pack with neighbor
buffer_store_b64 v[192:193], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 18 - 15 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+196], s[sgprBeta], v240, v[vgprValuC+196] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+197], s[sgprBeta], v240, v[vgprValuC+197] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+198], s[sgprBeta], v241, v[vgprValuC+198] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+199], s[sgprBeta], v241, v[vgprValuC+199] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+196], v[vgprValuC+196]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+197], v[vgprValuC+197]   // convert C to fp16
v_pack_b32_f16 v196, v[vgprValuC+196], v[vgprValuC+197] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+198], v[vgprValuC+198]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+199], v[vgprValuC+199]   // convert C to fp16
v_pack_b32_f16 v197, v[vgprValuC+198], v[vgprValuC+199] // Pack with neighbor
buffer_store_b64 v[196:197], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 18 - 16 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+204], s[sgprBeta], v242, v[vgprValuC+204] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+205], s[sgprBeta], v242, v[vgprValuC+205] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+206], s[sgprBeta], v243, v[vgprValuC+206] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+207], s[sgprBeta], v243, v[vgprValuC+207] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+204], v[vgprValuC+204]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+205], v[vgprValuC+205]   // convert C to fp16
v_pack_b32_f16 v204, v[vgprValuC+204], v[vgprValuC+205] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+206], v[vgprValuC+206]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+207], v[vgprValuC+207]   // convert C to fp16
v_pack_b32_f16 v205, v[vgprValuC+206], v[vgprValuC+207] // Pack with neighbor
buffer_store_b64 v[204:205], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 18 - 17 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+208], s[sgprBeta], v244, v[vgprValuC+208] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+209], s[sgprBeta], v244, v[vgprValuC+209] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+210], s[sgprBeta], v245, v[vgprValuC+210] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+211], s[sgprBeta], v245, v[vgprValuC+211] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+208], v[vgprValuC+208]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+209], v[vgprValuC+209]   // convert C to fp16
v_pack_b32_f16 v208, v[vgprValuC+208], v[vgprValuC+209] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+210], v[vgprValuC+210]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+211], v[vgprValuC+211]   // convert C to fp16
v_pack_b32_f16 v209, v[vgprValuC+210], v[vgprValuC+211] // Pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b64 v[208:209], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 18 - 18 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+212], s[sgprBeta], v246, v[vgprValuC+212] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+213], s[sgprBeta], v246, v[vgprValuC+213] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+214], s[sgprBeta], v247, v[vgprValuC+214] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+215], s[sgprBeta], v247, v[vgprValuC+215] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+212], v[vgprValuC+212]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+213], v[vgprValuC+213]   // convert C to fp16
v_pack_b32_f16 v212, v[vgprValuC+212], v[vgprValuC+213] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+214], v[vgprValuC+214]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+215], v[vgprValuC+215]   // convert C to fp16
v_pack_b32_f16 v213, v[vgprValuC+214], v[vgprValuC+215] // Pack with neighbor
buffer_store_b64 v[212:213], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,2,2,0:vw4); (0,3,2,0:vw4); (0,4,2,0:vw4); (0,5,2,0:vw4); (0,6,2,0:vw4); (0,7,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (0,2,3,0:vw4); (0,3,3,0:vw4); (0,4,3,0:vw4); (0,5,3,0:vw4); (0,6,3,0:vw4); (0,7,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
buffer_load_b64 v[138:139], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
buffer_load_b64 v[196:197], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:48 // load C
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
buffer_load_b64 v[198:199], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
buffer_load_b64 v[202:203], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:80 // load C
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
buffer_load_b64 v[204:205], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:96 // load C
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
buffer_load_b64 v[206:207], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:112 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_b64 v[208:209], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
buffer_load_b64 v[210:211], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:16 // load C
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
buffer_load_b64 v[212:213], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:32 // load C
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
buffer_load_b64 v[214:215], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:48 // load C
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
buffer_load_b64 v[216:217], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64 // load C
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
buffer_load_b64 v[218:219], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:80 // load C
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
buffer_load_b64 v[220:221], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:96 // load C
/* (d1,vc1,d0,vc0)=(0,3,7,0) */
buffer_load_b64 v[222:223], v136, s[sgprSrdC:sgprSrdC+3], 0 offen offset:112 // load C

/* rC *= alpha batchElements=[(0, 2, 2, 0), (0, 3, 2, 0), (0, 4, 2, 0), (0, 5, 2, 0), (0, 6, 2, 0), (0, 7, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0), (0, 4, 3, 0), (0, 5, 3, 0), (0, 6, 3, 0), (0, 7, 3, 0)] */
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(13)                                // vlcnt(13) = 14 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v138, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v138, v[vgprValuC+141] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v139, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v139, v[vgprValuC+143] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+140], v[vgprValuC+140]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+141], v[vgprValuC+141]   // convert C to fp16
v_pack_b32_f16 v140, v[vgprValuC+140], v[vgprValuC+141] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+142], v[vgprValuC+142]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+143], v[vgprValuC+143]   // convert C to fp16
v_pack_b32_f16 v141, v[vgprValuC+142], v[vgprValuC+143] // Pack with neighbor
buffer_store_b64 v[140:141], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(12)                                // vlcnt(12) = 14 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v196, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v196, v[vgprValuC+145] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v197, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v197, v[vgprValuC+147] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+144], v[vgprValuC+144]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+145], v[vgprValuC+145]   // convert C to fp16
v_pack_b32_f16 v144, v[vgprValuC+144], v[vgprValuC+145] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+146], v[vgprValuC+146]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+147], v[vgprValuC+147]   // convert C to fp16
v_pack_b32_f16 v145, v[vgprValuC+146], v[vgprValuC+147] // Pack with neighbor
buffer_store_b64 v[144:145], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D

s_waitcnt vmcnt(11)                                // vlcnt(11) = 14 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v198, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v198, v[vgprValuC+149] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v199, v[vgprValuC+150] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v199, v[vgprValuC+151] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+148], v[vgprValuC+148]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+149], v[vgprValuC+149]   // convert C to fp16
v_pack_b32_f16 v148, v[vgprValuC+148], v[vgprValuC+149] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+150], v[vgprValuC+150]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+151], v[vgprValuC+151]   // convert C to fp16
v_pack_b32_f16 v149, v[vgprValuC+150], v[vgprValuC+151] // Pack with neighbor
buffer_store_b64 v[148:149], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(10)                                // vlcnt(10) = 14 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v202, v[vgprValuC+152] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v202, v[vgprValuC+153] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v203, v[vgprValuC+154] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v203, v[vgprValuC+155] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+152], v[vgprValuC+152]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+153], v[vgprValuC+153]   // convert C to fp16
v_pack_b32_f16 v152, v[vgprValuC+152], v[vgprValuC+153] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+154], v[vgprValuC+154]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+155], v[vgprValuC+155]   // convert C to fp16
v_pack_b32_f16 v153, v[vgprValuC+154], v[vgprValuC+155] // Pack with neighbor
buffer_store_b64 v[152:153], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 14 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v204, v[vgprValuC+156] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v204, v[vgprValuC+157] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v205, v[vgprValuC+158] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v205, v[vgprValuC+159] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+156], v[vgprValuC+156]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+157], v[vgprValuC+157]   // convert C to fp16
v_pack_b32_f16 v156, v[vgprValuC+156], v[vgprValuC+157] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+158], v[vgprValuC+158]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+159], v[vgprValuC+159]   // convert C to fp16
v_pack_b32_f16 v157, v[vgprValuC+158], v[vgprValuC+159] // Pack with neighbor
buffer_store_b64 v[156:157], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 14 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v206, v[vgprValuC+160] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v206, v[vgprValuC+161] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v207, v[vgprValuC+162] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+163], s[sgprBeta], v207, v[vgprValuC+163] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+160], v[vgprValuC+160]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+161], v[vgprValuC+161]   // convert C to fp16
v_pack_b32_f16 v160, v[vgprValuC+160], v[vgprValuC+161] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+162], v[vgprValuC+162]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+163], v[vgprValuC+163]   // convert C to fp16
v_pack_b32_f16 v161, v[vgprValuC+162], v[vgprValuC+163] // Pack with neighbor
buffer_store_b64 v[160:161], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 14 - 7 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+164], s[sgprBeta], v208, v[vgprValuC+164] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+165], s[sgprBeta], v208, v[vgprValuC+165] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+166], s[sgprBeta], v209, v[vgprValuC+166] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+167], s[sgprBeta], v209, v[vgprValuC+167] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+164], v[vgprValuC+164]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+165], v[vgprValuC+165]   // convert C to fp16
v_pack_b32_f16 v164, v[vgprValuC+164], v[vgprValuC+165] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+166], v[vgprValuC+166]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+167], v[vgprValuC+167]   // convert C to fp16
v_pack_b32_f16 v165, v[vgprValuC+166], v[vgprValuC+167] // Pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_b64 v[164:165], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 14 - 8 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+168], s[sgprBeta], v210, v[vgprValuC+168] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+169], s[sgprBeta], v210, v[vgprValuC+169] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+170], s[sgprBeta], v211, v[vgprValuC+170] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+171], s[sgprBeta], v211, v[vgprValuC+171] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+168], v[vgprValuC+168]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+169], v[vgprValuC+169]   // convert C to fp16
v_pack_b32_f16 v168, v[vgprValuC+168], v[vgprValuC+169] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+170], v[vgprValuC+170]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+171], v[vgprValuC+171]   // convert C to fp16
v_pack_b32_f16 v169, v[vgprValuC+170], v[vgprValuC+171] // Pack with neighbor
buffer_store_b64 v[168:169], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 14 - 9 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+172], s[sgprBeta], v212, v[vgprValuC+172] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+173], s[sgprBeta], v212, v[vgprValuC+173] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+174], s[sgprBeta], v213, v[vgprValuC+174] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+175], s[sgprBeta], v213, v[vgprValuC+175] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+172], v[vgprValuC+172]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+173], v[vgprValuC+173]   // convert C to fp16
v_pack_b32_f16 v172, v[vgprValuC+172], v[vgprValuC+173] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+174], v[vgprValuC+174]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+175], v[vgprValuC+175]   // convert C to fp16
v_pack_b32_f16 v173, v[vgprValuC+174], v[vgprValuC+175] // Pack with neighbor
buffer_store_b64 v[172:173], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 14 - 10 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+176], s[sgprBeta], v214, v[vgprValuC+176] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+177], s[sgprBeta], v214, v[vgprValuC+177] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+178], s[sgprBeta], v215, v[vgprValuC+178] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+179], s[sgprBeta], v215, v[vgprValuC+179] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+176], v[vgprValuC+176]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+177], v[vgprValuC+177]   // convert C to fp16
v_pack_b32_f16 v176, v[vgprValuC+176], v[vgprValuC+177] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+178], v[vgprValuC+178]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+179], v[vgprValuC+179]   // convert C to fp16
v_pack_b32_f16 v177, v[vgprValuC+178], v[vgprValuC+179] // Pack with neighbor
buffer_store_b64 v[176:177], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 14 - 11 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+180], s[sgprBeta], v216, v[vgprValuC+180] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+181], s[sgprBeta], v216, v[vgprValuC+181] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+182], s[sgprBeta], v217, v[vgprValuC+182] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+183], s[sgprBeta], v217, v[vgprValuC+183] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+180], v[vgprValuC+180]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+181], v[vgprValuC+181]   // convert C to fp16
v_pack_b32_f16 v180, v[vgprValuC+180], v[vgprValuC+181] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+182], v[vgprValuC+182]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+183], v[vgprValuC+183]   // convert C to fp16
v_pack_b32_f16 v181, v[vgprValuC+182], v[vgprValuC+183] // Pack with neighbor
buffer_store_b64 v[180:181], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 14 - 12 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+184], s[sgprBeta], v218, v[vgprValuC+184] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+185], s[sgprBeta], v218, v[vgprValuC+185] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+186], s[sgprBeta], v219, v[vgprValuC+186] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+187], s[sgprBeta], v219, v[vgprValuC+187] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+184], v[vgprValuC+184]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+185], v[vgprValuC+185]   // convert C to fp16
v_pack_b32_f16 v184, v[vgprValuC+184], v[vgprValuC+185] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+186], v[vgprValuC+186]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+187], v[vgprValuC+187]   // convert C to fp16
v_pack_b32_f16 v185, v[vgprValuC+186], v[vgprValuC+187] // Pack with neighbor
buffer_store_b64 v[184:185], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:80 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 14 - 13 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+188], s[sgprBeta], v220, v[vgprValuC+188] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+189], s[sgprBeta], v220, v[vgprValuC+189] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+190], s[sgprBeta], v221, v[vgprValuC+190] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+191], s[sgprBeta], v221, v[vgprValuC+191] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+188], v[vgprValuC+188]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+189], v[vgprValuC+189]   // convert C to fp16
v_pack_b32_f16 v188, v[vgprValuC+188], v[vgprValuC+189] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+190], v[vgprValuC+190]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+191], v[vgprValuC+191]   // convert C to fp16
v_pack_b32_f16 v189, v[vgprValuC+190], v[vgprValuC+191] // Pack with neighbor
buffer_store_b64 v[188:189], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:96 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 14 - 14 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+192], s[sgprBeta], v222, v[vgprValuC+192] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+193], s[sgprBeta], v222, v[vgprValuC+193] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+194], s[sgprBeta], v223, v[vgprValuC+194] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+195], s[sgprBeta], v223, v[vgprValuC+195] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+192], v[vgprValuC+192]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+193], v[vgprValuC+193]   // convert C to fp16
v_pack_b32_f16 v192, v[vgprValuC+192], v[vgprValuC+193] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+194], v[vgprValuC+194]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+195], v[vgprValuC+195]   // convert C to fp16
v_pack_b32_f16 v193, v[vgprValuC+194], v[vgprValuC+195] // Pack with neighbor
buffer_store_b64 v[192:193], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:112 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_NonEdgeEnd:
label_GW_B1_FD0_VW4_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,2,0,0:vw4); (0,3,0,0:vw4); (0,4,0,0:vw4); (0,5,0,0:vw4); (0,6,0,0:vw4); (0,7,0,0:vw4); (0,0,1,0:vw4); (0,1,1,0:vw4); (0,2,1,0:vw4); (0,3,1,0:vw4); (0,4,1,0:vw4); (0,5,1,0:vw4); (0,6,1,0:vw4); (0,7,1,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v135, v130, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, v134, v135, s42                // LDC clip if OOB. offset
buffer_load_b64 v[202:203], v135, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v135, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, v134, v135, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDC clip if OOB. offset
buffer_load_b64 v[204:205], v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v208, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, v134, v208, s42                // LDC clip if OOB. offset
buffer_load_b64 v[206:207], v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, v134, v208, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDC clip if OOB. offset
buffer_load_b64 v[210:211], v209, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v209, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v214, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, v134, v214, s42                // LDC clip if OOB. offset
buffer_load_b64 v[212:213], v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, v134, v214, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDC clip if OOB. offset
buffer_load_b64 v[216:217], v215, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v215, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v220, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v220, v134, v220, s42                // LDC clip if OOB. offset
buffer_load_b64 v[218:219], v220, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v220, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v220, v134, v220, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDC clip if OOB. offset
buffer_load_b64 v[222:223], v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v226, v130, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, v134, v226, s42                // LDC clip if OOB. offset
buffer_load_b64 v[224:225], v226, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v226, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, v134, v226, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDC clip if OOB. offset
buffer_load_b64 v[228:229], v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v232, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDC clip if OOB. offset
buffer_load_b64 v[230:231], v232, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v232, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDC clip if OOB. offset
buffer_load_b64 v[234:235], v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v238, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDC clip if OOB. offset
buffer_load_b64 v[236:237], v238, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v238, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDC clip if OOB. offset
buffer_load_b64 v[240:241], v239, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v239, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v244, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDC clip if OOB. offset
buffer_load_b64 v[242:243], v244, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v244, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDC clip if OOB. offset
buffer_load_b64 v[246:247], v245, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v245, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0), (0, 3, 1, 0), (0, 4, 1, 0), (0, 5, 1, 0), (0, 6, 1, 0), (0, 7, 1, 0)] */
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v202, v[vgprValuC+136] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v202, v[vgprValuC+137] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v203, v[vgprValuC+138] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v203, v[vgprValuC+139] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+136], v[vgprValuC+136]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+137], v[vgprValuC+137]   // convert C to fp16
v_pack_b32_f16 v136, v[vgprValuC+136], v[vgprValuC+137] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+138], v[vgprValuC+138]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+139], v[vgprValuC+139]   // convert C to fp16
v_pack_b32_f16 v137, v[vgprValuC+138], v[vgprValuC+139] // Pack with neighbor
buffer_store_b64 v[136:137], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v204, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v204, v[vgprValuC+141] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v205, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v205, v[vgprValuC+143] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+140], v[vgprValuC+140]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+141], v[vgprValuC+141]   // convert C to fp16
v_pack_b32_f16 v140, v[vgprValuC+140], v[vgprValuC+141] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+142], v[vgprValuC+142]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+143], v[vgprValuC+143]   // convert C to fp16
v_pack_b32_f16 v141, v[vgprValuC+142], v[vgprValuC+143] // Pack with neighbor
buffer_store_b64 v[140:141], v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v206, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v206, v[vgprValuC+145] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v207, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v207, v[vgprValuC+147] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+144], v[vgprValuC+144]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+145], v[vgprValuC+145]   // convert C to fp16
v_pack_b32_f16 v144, v[vgprValuC+144], v[vgprValuC+145] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+146], v[vgprValuC+146]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+147], v[vgprValuC+147]   // convert C to fp16
v_pack_b32_f16 v145, v[vgprValuC+146], v[vgprValuC+147] // Pack with neighbor
buffer_store_b64 v[144:145], v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v210, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v210, v[vgprValuC+149] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v211, v[vgprValuC+150] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v211, v[vgprValuC+151] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+148], v[vgprValuC+148]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+149], v[vgprValuC+149]   // convert C to fp16
v_pack_b32_f16 v148, v[vgprValuC+148], v[vgprValuC+149] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+150], v[vgprValuC+150]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+151], v[vgprValuC+151]   // convert C to fp16
v_pack_b32_f16 v149, v[vgprValuC+150], v[vgprValuC+151] // Pack with neighbor
buffer_store_b64 v[148:149], v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v212, v[vgprValuC+152] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v212, v[vgprValuC+153] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v213, v[vgprValuC+154] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v213, v[vgprValuC+155] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+152], v[vgprValuC+152]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+153], v[vgprValuC+153]   // convert C to fp16
v_pack_b32_f16 v152, v[vgprValuC+152], v[vgprValuC+153] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+154], v[vgprValuC+154]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+155], v[vgprValuC+155]   // convert C to fp16
v_pack_b32_f16 v153, v[vgprValuC+154], v[vgprValuC+155] // Pack with neighbor
buffer_store_b64 v[152:153], v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v216, v[vgprValuC+156] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v216, v[vgprValuC+157] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v217, v[vgprValuC+158] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v217, v[vgprValuC+159] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+156], v[vgprValuC+156]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+157], v[vgprValuC+157]   // convert C to fp16
v_pack_b32_f16 v156, v[vgprValuC+156], v[vgprValuC+157] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+158], v[vgprValuC+158]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+159], v[vgprValuC+159]   // convert C to fp16
v_pack_b32_f16 v157, v[vgprValuC+158], v[vgprValuC+159] // Pack with neighbor
buffer_store_b64 v[156:157], v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v218, v[vgprValuC+160] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v218, v[vgprValuC+161] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v219, v[vgprValuC+162] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+163], s[sgprBeta], v219, v[vgprValuC+163] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+160], v[vgprValuC+160]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+161], v[vgprValuC+161]   // convert C to fp16
v_pack_b32_f16 v160, v[vgprValuC+160], v[vgprValuC+161] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+162], v[vgprValuC+162]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+163], v[vgprValuC+163]   // convert C to fp16
v_pack_b32_f16 v161, v[vgprValuC+162], v[vgprValuC+163] // Pack with neighbor
buffer_store_b64 v[160:161], v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+164], s[sgprBeta], v222, v[vgprValuC+164] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+165], s[sgprBeta], v222, v[vgprValuC+165] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+166], s[sgprBeta], v223, v[vgprValuC+166] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+167], s[sgprBeta], v223, v[vgprValuC+167] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+164], v[vgprValuC+164]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+165], v[vgprValuC+165]   // convert C to fp16
v_pack_b32_f16 v164, v[vgprValuC+164], v[vgprValuC+165] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+166], v[vgprValuC+166]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+167], v[vgprValuC+167]   // convert C to fp16
v_pack_b32_f16 v165, v[vgprValuC+166], v[vgprValuC+167] // Pack with neighbor
buffer_store_b64 v[164:165], v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+168], s[sgprBeta], v224, v[vgprValuC+168] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+169], s[sgprBeta], v224, v[vgprValuC+169] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+170], s[sgprBeta], v225, v[vgprValuC+170] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+171], s[sgprBeta], v225, v[vgprValuC+171] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+168], v[vgprValuC+168]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+169], v[vgprValuC+169]   // convert C to fp16
v_pack_b32_f16 v168, v[vgprValuC+168], v[vgprValuC+169] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+170], v[vgprValuC+170]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+171], v[vgprValuC+171]   // convert C to fp16
v_pack_b32_f16 v169, v[vgprValuC+170], v[vgprValuC+171] // Pack with neighbor
buffer_store_b64 v[168:169], v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+172], s[sgprBeta], v228, v[vgprValuC+172] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+173], s[sgprBeta], v228, v[vgprValuC+173] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+174], s[sgprBeta], v229, v[vgprValuC+174] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+175], s[sgprBeta], v229, v[vgprValuC+175] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+172], v[vgprValuC+172]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+173], v[vgprValuC+173]   // convert C to fp16
v_pack_b32_f16 v172, v[vgprValuC+172], v[vgprValuC+173] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+174], v[vgprValuC+174]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+175], v[vgprValuC+175]   // convert C to fp16
v_pack_b32_f16 v173, v[vgprValuC+174], v[vgprValuC+175] // Pack with neighbor
buffer_store_b64 v[172:173], v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+176], s[sgprBeta], v230, v[vgprValuC+176] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+177], s[sgprBeta], v230, v[vgprValuC+177] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+178], s[sgprBeta], v231, v[vgprValuC+178] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+179], s[sgprBeta], v231, v[vgprValuC+179] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+176], v[vgprValuC+176]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+177], v[vgprValuC+177]   // convert C to fp16
v_pack_b32_f16 v176, v[vgprValuC+176], v[vgprValuC+177] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+178], v[vgprValuC+178]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+179], v[vgprValuC+179]   // convert C to fp16
v_pack_b32_f16 v177, v[vgprValuC+178], v[vgprValuC+179] // Pack with neighbor
buffer_store_b64 v[176:177], v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+180], s[sgprBeta], v234, v[vgprValuC+180] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+181], s[sgprBeta], v234, v[vgprValuC+181] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+182], s[sgprBeta], v235, v[vgprValuC+182] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+183], s[sgprBeta], v235, v[vgprValuC+183] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+180], v[vgprValuC+180]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+181], v[vgprValuC+181]   // convert C to fp16
v_pack_b32_f16 v180, v[vgprValuC+180], v[vgprValuC+181] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+182], v[vgprValuC+182]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+183], v[vgprValuC+183]   // convert C to fp16
v_pack_b32_f16 v181, v[vgprValuC+182], v[vgprValuC+183] // Pack with neighbor
buffer_store_b64 v[180:181], v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+184], s[sgprBeta], v236, v[vgprValuC+184] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+185], s[sgprBeta], v236, v[vgprValuC+185] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+186], s[sgprBeta], v237, v[vgprValuC+186] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+187], s[sgprBeta], v237, v[vgprValuC+187] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+184], v[vgprValuC+184]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+185], v[vgprValuC+185]   // convert C to fp16
v_pack_b32_f16 v184, v[vgprValuC+184], v[vgprValuC+185] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+186], v[vgprValuC+186]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+187], v[vgprValuC+187]   // convert C to fp16
v_pack_b32_f16 v185, v[vgprValuC+186], v[vgprValuC+187] // Pack with neighbor
buffer_store_b64 v[184:185], v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+188], s[sgprBeta], v240, v[vgprValuC+188] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+189], s[sgprBeta], v240, v[vgprValuC+189] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+190], s[sgprBeta], v241, v[vgprValuC+190] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+191], s[sgprBeta], v241, v[vgprValuC+191] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+188], v[vgprValuC+188]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+189], v[vgprValuC+189]   // convert C to fp16
v_pack_b32_f16 v188, v[vgprValuC+188], v[vgprValuC+189] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+190], v[vgprValuC+190]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+191], v[vgprValuC+191]   // convert C to fp16
v_pack_b32_f16 v189, v[vgprValuC+190], v[vgprValuC+191] // Pack with neighbor
buffer_store_b64 v[188:189], v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+192], s[sgprBeta], v242, v[vgprValuC+192] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+193], s[sgprBeta], v242, v[vgprValuC+193] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+194], s[sgprBeta], v243, v[vgprValuC+194] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+195], s[sgprBeta], v243, v[vgprValuC+195] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+192], v[vgprValuC+192]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+193], v[vgprValuC+193]   // convert C to fp16
v_pack_b32_f16 v192, v[vgprValuC+192], v[vgprValuC+193] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+194], v[vgprValuC+194]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+195], v[vgprValuC+195]   // convert C to fp16
v_pack_b32_f16 v193, v[vgprValuC+194], v[vgprValuC+195] // Pack with neighbor
buffer_store_b64 v[192:193], v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+196], s[sgprBeta], v246, v[vgprValuC+196] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+197], s[sgprBeta], v246, v[vgprValuC+197] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+198], s[sgprBeta], v247, v[vgprValuC+198] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+199], s[sgprBeta], v247, v[vgprValuC+199] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+196], v[vgprValuC+196]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+197], v[vgprValuC+197]   // convert C to fp16
v_pack_b32_f16 v196, v[vgprValuC+196], v[vgprValuC+197] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+198], v[vgprValuC+198]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+199], v[vgprValuC+199]   // convert C to fp16
v_pack_b32_f16 v197, v[vgprValuC+198], v[vgprValuC+199] // Pack with neighbor
buffer_store_b64 v[196:197], v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,2,0:vw4); (0,1,2,0:vw4); (0,2,2,0:vw4); (0,3,2,0:vw4); (0,4,2,0:vw4); (0,5,2,0:vw4); (0,6,2,0:vw4); (0,7,2,0:vw4); (0,0,3,0:vw4); (0,1,3,0:vw4); (0,2,3,0:vw4); (0,3,3,0:vw4); (0,4,3,0:vw4); (0,5,3,0:vw4); (0,6,3,0:vw4); (0,7,3,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v135, v130, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, v134, v135, s42                // LDC clip if OOB. offset
buffer_load_b64 v[202:203], v135, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v135, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v135, v134, v135, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDC clip if OOB. offset
buffer_load_b64 v[204:205], v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v208, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, v134, v208, s42                // LDC clip if OOB. offset
buffer_load_b64 v[206:207], v208, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v208, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v208, v134, v208, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDC clip if OOB. offset
buffer_load_b64 v[210:211], v209, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v209, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v214, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, v134, v214, s42                // LDC clip if OOB. offset
buffer_load_b64 v[212:213], v214, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v214, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v214, v134, v214, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDC clip if OOB. offset
buffer_load_b64 v[216:217], v215, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v215, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v220, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v220, v134, v220, s42                // LDC clip if OOB. offset
buffer_load_b64 v[218:219], v220, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v220, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v220, v134, v220, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDC clip if OOB. offset
buffer_load_b64 v[222:223], v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v226, v130, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, v134, v226, s42                // LDC clip if OOB. offset
buffer_load_b64 v[224:225], v226, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v226, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v226, v134, v226, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDC clip if OOB. offset
buffer_load_b64 v[228:229], v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v232, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDC clip if OOB. offset
buffer_load_b64 v[230:231], v232, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v232, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v232, v134, v232, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDC clip if OOB. offset
buffer_load_b64 v[234:235], v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v238, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDC clip if OOB. offset
buffer_load_b64 v[236:237], v238, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v238, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v238, v134, v238, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDC clip if OOB. offset
buffer_load_b64 v[240:241], v239, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v239, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v244, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDC clip if OOB. offset
buffer_load_b64 v[242:243], v244, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v244, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v244, v134, v244, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDC clip if OOB. offset
buffer_load_b64 v[246:247], v245, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v245, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 2, 0), (0, 1, 2, 0), (0, 2, 2, 0), (0, 3, 2, 0), (0, 4, 2, 0), (0, 5, 2, 0), (0, 6, 2, 0), (0, 7, 2, 0), (0, 0, 3, 0), (0, 1, 3, 0), (0, 2, 3, 0), (0, 3, 3, 0), (0, 4, 3, 0), (0, 5, 3, 0), (0, 6, 3, 0), (0, 7, 3, 0)] */
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+175], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+176], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+177], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+178], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+179], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+180], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+181], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+182], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+183], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+184], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+185], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+186], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+187], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+188], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+189], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+190], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+191], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+192], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+193], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+194], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+195], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+196], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+197], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+198], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+199], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v202, v[vgprValuC+136] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v202, v[vgprValuC+137] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v203, v[vgprValuC+138] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v203, v[vgprValuC+139] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+136], v[vgprValuC+136]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+137], v[vgprValuC+137]   // convert C to fp16
v_pack_b32_f16 v136, v[vgprValuC+136], v[vgprValuC+137] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+138], v[vgprValuC+138]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+139], v[vgprValuC+139]   // convert C to fp16
v_pack_b32_f16 v137, v[vgprValuC+138], v[vgprValuC+139] // Pack with neighbor
buffer_store_b64 v[136:137], v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v204, v[vgprValuC+140] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v204, v[vgprValuC+141] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v205, v[vgprValuC+142] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v205, v[vgprValuC+143] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+140], v[vgprValuC+140]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+141], v[vgprValuC+141]   // convert C to fp16
v_pack_b32_f16 v140, v[vgprValuC+140], v[vgprValuC+141] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+142], v[vgprValuC+142]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+143], v[vgprValuC+143]   // convert C to fp16
v_pack_b32_f16 v141, v[vgprValuC+142], v[vgprValuC+143] // Pack with neighbor
buffer_store_b64 v[140:141], v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v206, v[vgprValuC+144] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v206, v[vgprValuC+145] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v207, v[vgprValuC+146] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v207, v[vgprValuC+147] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+144], v[vgprValuC+144]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+145], v[vgprValuC+145]   // convert C to fp16
v_pack_b32_f16 v144, v[vgprValuC+144], v[vgprValuC+145] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+146], v[vgprValuC+146]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+147], v[vgprValuC+147]   // convert C to fp16
v_pack_b32_f16 v145, v[vgprValuC+146], v[vgprValuC+147] // Pack with neighbor
buffer_store_b64 v[144:145], v208, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v210, v[vgprValuC+148] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v210, v[vgprValuC+149] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v211, v[vgprValuC+150] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v211, v[vgprValuC+151] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+148], v[vgprValuC+148]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+149], v[vgprValuC+149]   // convert C to fp16
v_pack_b32_f16 v148, v[vgprValuC+148], v[vgprValuC+149] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+150], v[vgprValuC+150]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+151], v[vgprValuC+151]   // convert C to fp16
v_pack_b32_f16 v149, v[vgprValuC+150], v[vgprValuC+151] // Pack with neighbor
buffer_store_b64 v[148:149], v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v212, v[vgprValuC+152] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v212, v[vgprValuC+153] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v213, v[vgprValuC+154] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v213, v[vgprValuC+155] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+152], v[vgprValuC+152]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+153], v[vgprValuC+153]   // convert C to fp16
v_pack_b32_f16 v152, v[vgprValuC+152], v[vgprValuC+153] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+154], v[vgprValuC+154]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+155], v[vgprValuC+155]   // convert C to fp16
v_pack_b32_f16 v153, v[vgprValuC+154], v[vgprValuC+155] // Pack with neighbor
buffer_store_b64 v[152:153], v214, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v216, v[vgprValuC+156] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v216, v[vgprValuC+157] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v217, v[vgprValuC+158] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v217, v[vgprValuC+159] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+156], v[vgprValuC+156]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+157], v[vgprValuC+157]   // convert C to fp16
v_pack_b32_f16 v156, v[vgprValuC+156], v[vgprValuC+157] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+158], v[vgprValuC+158]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+159], v[vgprValuC+159]   // convert C to fp16
v_pack_b32_f16 v157, v[vgprValuC+158], v[vgprValuC+159] // Pack with neighbor
buffer_store_b64 v[156:157], v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v218, v[vgprValuC+160] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v218, v[vgprValuC+161] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v219, v[vgprValuC+162] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+163], s[sgprBeta], v219, v[vgprValuC+163] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+160], v[vgprValuC+160]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+161], v[vgprValuC+161]   // convert C to fp16
v_pack_b32_f16 v160, v[vgprValuC+160], v[vgprValuC+161] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+162], v[vgprValuC+162]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+163], v[vgprValuC+163]   // convert C to fp16
v_pack_b32_f16 v161, v[vgprValuC+162], v[vgprValuC+163] // Pack with neighbor
buffer_store_b64 v[160:161], v220, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+164], s[sgprBeta], v222, v[vgprValuC+164] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+165], s[sgprBeta], v222, v[vgprValuC+165] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+166], s[sgprBeta], v223, v[vgprValuC+166] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+167], s[sgprBeta], v223, v[vgprValuC+167] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+164], v[vgprValuC+164]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+165], v[vgprValuC+165]   // convert C to fp16
v_pack_b32_f16 v164, v[vgprValuC+164], v[vgprValuC+165] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+166], v[vgprValuC+166]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+167], v[vgprValuC+167]   // convert C to fp16
v_pack_b32_f16 v165, v[vgprValuC+166], v[vgprValuC+167] // Pack with neighbor
buffer_store_b64 v[164:165], v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+168], s[sgprBeta], v224, v[vgprValuC+168] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+169], s[sgprBeta], v224, v[vgprValuC+169] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+170], s[sgprBeta], v225, v[vgprValuC+170] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+171], s[sgprBeta], v225, v[vgprValuC+171] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+168], v[vgprValuC+168]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+169], v[vgprValuC+169]   // convert C to fp16
v_pack_b32_f16 v168, v[vgprValuC+168], v[vgprValuC+169] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+170], v[vgprValuC+170]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+171], v[vgprValuC+171]   // convert C to fp16
v_pack_b32_f16 v169, v[vgprValuC+170], v[vgprValuC+171] // Pack with neighbor
buffer_store_b64 v[168:169], v226, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+172], s[sgprBeta], v228, v[vgprValuC+172] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+173], s[sgprBeta], v228, v[vgprValuC+173] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+174], s[sgprBeta], v229, v[vgprValuC+174] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+175], s[sgprBeta], v229, v[vgprValuC+175] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+172], v[vgprValuC+172]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+173], v[vgprValuC+173]   // convert C to fp16
v_pack_b32_f16 v172, v[vgprValuC+172], v[vgprValuC+173] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+174], v[vgprValuC+174]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+175], v[vgprValuC+175]   // convert C to fp16
v_pack_b32_f16 v173, v[vgprValuC+174], v[vgprValuC+175] // Pack with neighbor
buffer_store_b64 v[172:173], v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+176], s[sgprBeta], v230, v[vgprValuC+176] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+177], s[sgprBeta], v230, v[vgprValuC+177] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+178], s[sgprBeta], v231, v[vgprValuC+178] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+179], s[sgprBeta], v231, v[vgprValuC+179] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+176], v[vgprValuC+176]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+177], v[vgprValuC+177]   // convert C to fp16
v_pack_b32_f16 v176, v[vgprValuC+176], v[vgprValuC+177] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+178], v[vgprValuC+178]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+179], v[vgprValuC+179]   // convert C to fp16
v_pack_b32_f16 v177, v[vgprValuC+178], v[vgprValuC+179] // Pack with neighbor
buffer_store_b64 v[176:177], v232, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+180], s[sgprBeta], v234, v[vgprValuC+180] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+181], s[sgprBeta], v234, v[vgprValuC+181] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+182], s[sgprBeta], v235, v[vgprValuC+182] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+183], s[sgprBeta], v235, v[vgprValuC+183] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+180], v[vgprValuC+180]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+181], v[vgprValuC+181]   // convert C to fp16
v_pack_b32_f16 v180, v[vgprValuC+180], v[vgprValuC+181] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+182], v[vgprValuC+182]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+183], v[vgprValuC+183]   // convert C to fp16
v_pack_b32_f16 v181, v[vgprValuC+182], v[vgprValuC+183] // Pack with neighbor
buffer_store_b64 v[180:181], v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+184], s[sgprBeta], v236, v[vgprValuC+184] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+185], s[sgprBeta], v236, v[vgprValuC+185] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+186], s[sgprBeta], v237, v[vgprValuC+186] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+187], s[sgprBeta], v237, v[vgprValuC+187] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+184], v[vgprValuC+184]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+185], v[vgprValuC+185]   // convert C to fp16
v_pack_b32_f16 v184, v[vgprValuC+184], v[vgprValuC+185] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+186], v[vgprValuC+186]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+187], v[vgprValuC+187]   // convert C to fp16
v_pack_b32_f16 v185, v[vgprValuC+186], v[vgprValuC+187] // Pack with neighbor
buffer_store_b64 v[184:185], v238, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+188], s[sgprBeta], v240, v[vgprValuC+188] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+189], s[sgprBeta], v240, v[vgprValuC+189] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+190], s[sgprBeta], v241, v[vgprValuC+190] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+191], s[sgprBeta], v241, v[vgprValuC+191] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+188], v[vgprValuC+188]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+189], v[vgprValuC+189]   // convert C to fp16
v_pack_b32_f16 v188, v[vgprValuC+188], v[vgprValuC+189] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+190], v[vgprValuC+190]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+191], v[vgprValuC+191]   // convert C to fp16
v_pack_b32_f16 v189, v[vgprValuC+190], v[vgprValuC+191] // Pack with neighbor
buffer_store_b64 v[188:189], v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+192], s[sgprBeta], v242, v[vgprValuC+192] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+193], s[sgprBeta], v242, v[vgprValuC+193] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+194], s[sgprBeta], v243, v[vgprValuC+194] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+195], s[sgprBeta], v243, v[vgprValuC+195] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+192], v[vgprValuC+192]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+193], v[vgprValuC+193]   // convert C to fp16
v_pack_b32_f16 v192, v[vgprValuC+192], v[vgprValuC+193] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+194], v[vgprValuC+194]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+195], v[vgprValuC+195]   // convert C to fp16
v_pack_b32_f16 v193, v[vgprValuC+194], v[vgprValuC+195] // Pack with neighbor
buffer_store_b64 v[192:193], v244, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+196], s[sgprBeta], v246, v[vgprValuC+196] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+197], s[sgprBeta], v246, v[vgprValuC+197] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+198], s[sgprBeta], v247, v[vgprValuC+198] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+199], s[sgprBeta], v247, v[vgprValuC+199] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v[vgprValuC+196], v[vgprValuC+196]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+197], v[vgprValuC+197]   // convert C to fp16
v_pack_b32_f16 v196, v[vgprValuC+196], v[vgprValuC+197] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+198], v[vgprValuC+198]   // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+199], v[vgprValuC+199]   // convert C to fp16
v_pack_b32_f16 v197, v[vgprValuC+198], v[vgprValuC+199] // Pack with neighbor
buffer_store_b64 v[196:197], v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_Else:
label_GW_B1_FD0_VW1_Else:
label_GW_B1_FD0_VW1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=40 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1); (0,2,0,0:vw1); (0,2,0,1:vw1); (0,2,0,2:vw1); (0,2,0,3:vw1); (0,3,0,0:vw1); (0,3,0,1:vw1); (0,3,0,2:vw1); (0,3,0,3:vw1); (0,4,0,0:vw1); (0,4,0,1:vw1); (0,4,0,2:vw1); (0,4,0,3:vw1); (0,5,0,0:vw1); (0,5,0,1:vw1); (0,5,0,2:vw1); (0,5,0,3:vw1); (0,6,0,0:vw1); (0,6,0,1:vw1); (0,6,0,2:vw1); (0,6,0,3:vw1); (0,7,0,0:vw1); (0,7,0,1:vw1); (0,7,0,2:vw1); (0,7,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,1,1,0:vw1); (0,1,1,1:vw1); (0,1,1,2:vw1); (0,1,1,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v176, v130, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, v134, v176, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, v134, v176, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v178, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, v134, v178, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, v134, v178, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v180, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, v134, v180, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, v134, v180, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v182, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, v134, v182, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, v134, v182, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v184, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, v134, v184, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, v134, v184, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v186, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v186, v134, v186, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v186, v134, v186, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v188, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v188, v134, v188, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v188, v134, v188, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v190, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, v134, v190, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, v134, v190, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v192, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v192, v134, v192, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v192, v134, v192, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v194, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v194, v134, v194, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v194, v134, v194, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v196, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v198, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v202, v203, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v203, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v205, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v204, v205, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v205, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v207, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v206, v207, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v207, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v208, v209, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v209, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v211, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v210, v211, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v211, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v213, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v212, v213, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v213, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v214, v215, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v215, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v217, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v216, v217, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v217, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v219, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v218, v219, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v219, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v220, v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v223, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v222, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v225, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v226, v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v228, v229, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v229, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v230, v231, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v231, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v232, v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v234, v235, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v235, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v236, v237, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v237, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v238, v239, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v239, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v130, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v240, v241, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v241, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v242, v243, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v243, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v244, v245, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v245, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v247, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v246, v247, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v247, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v249, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v248, v249, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v249, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v251, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v250, v251, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v251, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v253, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v252, v253, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v253, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v255, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v254, v255, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v255, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 0, 3), (0, 3, 0, 0), (0, 3, 0, 1), (0, 3, 0, 2), (0, 3, 0, 3), (0, 4, 0, 0), (0, 4, 0, 1), (0, 4, 0, 2), (0, 4, 0, 3), (0, 5, 0, 0), (0, 5, 0, 1), (0, 5, 0, 2), (0, 5, 0, 3), (0, 6, 0, 0), (0, 6, 0, 1), (0, 6, 0, 2), (0, 6, 0, 3), (0, 7, 0, 0), (0, 7, 0, 1), (0, 7, 0, 2), (0, 7, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2), (0, 1, 1, 3)] */
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+16] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+24] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+17] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+25] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+18] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+26] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+19] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+27] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+12] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+20] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+28] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+13] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+21] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+29] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+14] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+22] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+30] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+15] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+23] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+31] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+32] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+40] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+48] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+56] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+33] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+41] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+49] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+57] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v175, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v177, v[vgprValuC+136] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v179, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v181, v[vgprValuC+138] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v183, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v185, v[vgprValuC+140] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v187, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v189, v[vgprValuC+142] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v191, v[vgprValuC+143] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
buffer_store_b16 v143, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v193, v[vgprValuC+144] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v195, v[vgprValuC+145] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
buffer_store_b16 v145, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v197, v[vgprValuC+146] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v199, v[vgprValuC+147] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
buffer_store_b16 v147, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v202, v[vgprValuC+148] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v204, v[vgprValuC+149] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
buffer_store_b16 v149, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v206, v[vgprValuC+150] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v208, v[vgprValuC+151] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
buffer_store_b16 v151, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v210, v[vgprValuC+152] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v212, v[vgprValuC+153] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
buffer_store_b16 v153, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v214, v[vgprValuC+154] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v216, v[vgprValuC+155] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
buffer_store_b16 v155, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v218, v[vgprValuC+156] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v220, v[vgprValuC+157] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
buffer_store_b16 v157, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v222, v[vgprValuC+158] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v224, v[vgprValuC+159] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
buffer_store_b16 v159, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v226, v[vgprValuC+160] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v228, v[vgprValuC+161] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
buffer_store_b16 v161, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v230, v[vgprValuC+162] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+163], s[sgprBeta], v232, v[vgprValuC+163] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v163, v[vgprValuC+163]               // convert C to fp16
buffer_store_b16 v163, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+164], s[sgprBeta], v234, v[vgprValuC+164] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v164, v[vgprValuC+164]               // convert C to fp16
buffer_store_b16 v164, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+165], s[sgprBeta], v236, v[vgprValuC+165] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v165, v[vgprValuC+165]               // convert C to fp16
buffer_store_b16 v165, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+166], s[sgprBeta], v238, v[vgprValuC+166] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v166, v[vgprValuC+166]               // convert C to fp16
buffer_store_b16 v166, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+167], s[sgprBeta], v240, v[vgprValuC+167] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v167, v[vgprValuC+167]               // convert C to fp16
buffer_store_b16 v167, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+168], s[sgprBeta], v242, v[vgprValuC+168] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v168, v[vgprValuC+168]               // convert C to fp16
buffer_store_b16 v168, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+169], s[sgprBeta], v244, v[vgprValuC+169] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v169, v[vgprValuC+169]               // convert C to fp16
buffer_store_b16 v169, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+170], s[sgprBeta], v246, v[vgprValuC+170] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v170, v[vgprValuC+170]               // convert C to fp16
buffer_store_b16 v170, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+171], s[sgprBeta], v248, v[vgprValuC+171] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v171, v[vgprValuC+171]               // convert C to fp16
buffer_store_b16 v171, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+172], s[sgprBeta], v250, v[vgprValuC+172] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v172, v[vgprValuC+172]               // convert C to fp16
buffer_store_b16 v172, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+173], s[sgprBeta], v252, v[vgprValuC+173] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v173, v[vgprValuC+173]               // convert C to fp16
buffer_store_b16 v173, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+174], s[sgprBeta], v254, v[vgprValuC+174] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v174, v[vgprValuC+174]               // convert C to fp16
buffer_store_b16 v174, v255, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,2,1,0:vw1); (0,2,1,1:vw1); (0,2,1,2:vw1); (0,2,1,3:vw1); (0,3,1,0:vw1); (0,3,1,1:vw1); (0,3,1,2:vw1); (0,3,1,3:vw1); (0,4,1,0:vw1); (0,4,1,1:vw1); (0,4,1,2:vw1); (0,4,1,3:vw1); (0,5,1,0:vw1); (0,5,1,1:vw1); (0,5,1,2:vw1); (0,5,1,3:vw1); (0,6,1,0:vw1); (0,6,1,1:vw1); (0,6,1,2:vw1); (0,6,1,3:vw1); (0,7,1,0:vw1); (0,7,1,1:vw1); (0,7,1,2:vw1); (0,7,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,1,2,0:vw1); (0,1,2,1:vw1); (0,1,2,2:vw1); (0,1,2,3:vw1); (0,2,2,0:vw1); (0,2,2,1:vw1); (0,2,2,2:vw1); (0,2,2,3:vw1); (0,3,2,0:vw1); (0,3,2,1:vw1); (0,3,2,2:vw1); (0,3,2,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,1,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v176, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, v134, v176, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, v134, v176, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v178, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, v134, v178, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, v134, v178, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v180, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, v134, v180, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, v134, v180, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v182, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, v134, v182, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, v134, v182, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v184, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, v134, v184, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, v134, v184, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v186, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v186, v134, v186, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v186, v134, v186, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v188, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v188, v134, v188, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v188, v134, v188, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v190, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, v134, v190, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, v134, v190, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v192, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v192, v134, v192, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v192, v134, v192, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v194, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v194, v134, v194, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v194, v134, v194, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v196, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v198, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v202, v203, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v203, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v205, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v204, v205, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v205, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v207, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v206, v207, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v207, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v208, v209, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v209, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v211, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v210, v211, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v211, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v213, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v212, v213, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v213, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v214, v215, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v215, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v217, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v216, v217, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v217, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v219, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v218, v219, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v219, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v220, v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v223, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v222, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v225, v130, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v226, v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v228, v229, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v229, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v230, v231, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v231, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v232, v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v234, v235, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v235, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v236, v237, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v237, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v238, v239, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v239, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v240, v241, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v241, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v242, v243, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v243, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v244, v245, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v245, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v247, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v246, v247, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v247, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v249, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v248, v249, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v249, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v251, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v250, v251, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v251, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v253, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v252, v253, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v253, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v255, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v254, v255, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v255, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 1, 2), (0, 2, 1, 3), (0, 3, 1, 0), (0, 3, 1, 1), (0, 3, 1, 2), (0, 3, 1, 3), (0, 4, 1, 0), (0, 4, 1, 1), (0, 4, 1, 2), (0, 4, 1, 3), (0, 5, 1, 0), (0, 5, 1, 1), (0, 5, 1, 2), (0, 5, 1, 3), (0, 6, 1, 0), (0, 6, 1, 1), (0, 6, 1, 2), (0, 6, 1, 3), (0, 7, 1, 0), (0, 7, 1, 1), (0, 7, 1, 2), (0, 7, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 1, 2, 0), (0, 1, 2, 1), (0, 1, 2, 2), (0, 1, 2, 3), (0, 2, 2, 0), (0, 2, 2, 1), (0, 2, 2, 2), (0, 2, 2, 3), (0, 3, 2, 0), (0, 3, 2, 1), (0, 3, 2, 2), (0, 3, 2, 3)] */
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+34] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+42] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+50] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+58] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+35] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+43] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+51] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+59] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+36] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+44] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+52] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+60] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+37] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+45] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+53] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+61] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+38] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+46] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+54] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+62] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+39] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+47] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+55] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+63] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+64] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+72] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+80] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+88] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+65] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+73] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+81] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+89] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+66] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+74] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+82] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+90] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+67] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+75] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+83] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+91] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v175, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v177, v[vgprValuC+136] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v179, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v181, v[vgprValuC+138] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v183, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v185, v[vgprValuC+140] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v187, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v189, v[vgprValuC+142] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v191, v[vgprValuC+143] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
buffer_store_b16 v143, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v193, v[vgprValuC+144] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v195, v[vgprValuC+145] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
buffer_store_b16 v145, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v197, v[vgprValuC+146] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v199, v[vgprValuC+147] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
buffer_store_b16 v147, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v202, v[vgprValuC+148] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v204, v[vgprValuC+149] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
buffer_store_b16 v149, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v206, v[vgprValuC+150] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v208, v[vgprValuC+151] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
buffer_store_b16 v151, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v210, v[vgprValuC+152] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v212, v[vgprValuC+153] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
buffer_store_b16 v153, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v214, v[vgprValuC+154] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v216, v[vgprValuC+155] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
buffer_store_b16 v155, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v218, v[vgprValuC+156] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v220, v[vgprValuC+157] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
buffer_store_b16 v157, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v222, v[vgprValuC+158] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v224, v[vgprValuC+159] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
buffer_store_b16 v159, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v226, v[vgprValuC+160] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v228, v[vgprValuC+161] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
buffer_store_b16 v161, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v230, v[vgprValuC+162] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+163], s[sgprBeta], v232, v[vgprValuC+163] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v163, v[vgprValuC+163]               // convert C to fp16
buffer_store_b16 v163, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+164], s[sgprBeta], v234, v[vgprValuC+164] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v164, v[vgprValuC+164]               // convert C to fp16
buffer_store_b16 v164, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+165], s[sgprBeta], v236, v[vgprValuC+165] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v165, v[vgprValuC+165]               // convert C to fp16
buffer_store_b16 v165, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+166], s[sgprBeta], v238, v[vgprValuC+166] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v166, v[vgprValuC+166]               // convert C to fp16
buffer_store_b16 v166, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+167], s[sgprBeta], v240, v[vgprValuC+167] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v167, v[vgprValuC+167]               // convert C to fp16
buffer_store_b16 v167, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+168], s[sgprBeta], v242, v[vgprValuC+168] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v168, v[vgprValuC+168]               // convert C to fp16
buffer_store_b16 v168, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+169], s[sgprBeta], v244, v[vgprValuC+169] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v169, v[vgprValuC+169]               // convert C to fp16
buffer_store_b16 v169, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+170], s[sgprBeta], v246, v[vgprValuC+170] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v170, v[vgprValuC+170]               // convert C to fp16
buffer_store_b16 v170, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+171], s[sgprBeta], v248, v[vgprValuC+171] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v171, v[vgprValuC+171]               // convert C to fp16
buffer_store_b16 v171, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+172], s[sgprBeta], v250, v[vgprValuC+172] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v172, v[vgprValuC+172]               // convert C to fp16
buffer_store_b16 v172, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+173], s[sgprBeta], v252, v[vgprValuC+173] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v173, v[vgprValuC+173]               // convert C to fp16
buffer_store_b16 v173, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+174], s[sgprBeta], v254, v[vgprValuC+174] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v174, v[vgprValuC+174]               // convert C to fp16
buffer_store_b16 v174, v255, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,4,2,0:vw1); (0,4,2,1:vw1); (0,4,2,2:vw1); (0,4,2,3:vw1); (0,5,2,0:vw1); (0,5,2,1:vw1); (0,5,2,2:vw1); (0,5,2,3:vw1); (0,6,2,0:vw1); (0,6,2,1:vw1); (0,6,2,2:vw1); (0,6,2,3:vw1); (0,7,2,0:vw1); (0,7,2,1:vw1); (0,7,2,2:vw1); (0,7,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,1,3,0:vw1); (0,1,3,1:vw1); (0,1,3,2:vw1); (0,1,3,3:vw1); (0,2,3,0:vw1); (0,2,3,1:vw1); (0,2,3,2:vw1); (0,2,3,3:vw1); (0,3,3,0:vw1); (0,3,3,1:vw1); (0,3,3,2:vw1); (0,3,3,3:vw1); (0,4,3,0:vw1); (0,4,3,1:vw1); (0,4,3,2:vw1); (0,4,3,3:vw1); (0,5,3,0:vw1); (0,5,3,1:vw1); (0,5,3,2:vw1); (0,5,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,2,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v176, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, v134, v176, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v175, v176, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v176, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v176, v134, v176, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v178, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, v134, v178, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v177, v178, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v178, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v178, v134, v178, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v180, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, v134, v180, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v179, v180, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v180, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v180, v134, v180, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v182, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, v134, v182, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v181, v182, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v182, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v182, v134, v182, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v184, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, v134, v184, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v183, v184, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v184, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v184, v134, v184, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v186, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v186, v134, v186, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v185, v186, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v186, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v186, v134, v186, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v188, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v188, v134, v188, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v187, v188, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v188, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v188, v134, v188, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v190, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, v134, v190, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v189, v190, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v190, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v190, v134, v190, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v192, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v192, v134, v192, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v191, v192, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v192, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v192, v134, v192, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v194, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v194, v134, v194, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v193, v194, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v194, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v194, v134, v194, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v196, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v195, v196, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v196, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v196, v134, v196, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v198, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v197, v198, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v198, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v198, v134, v198, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v200, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v199, v200, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v200, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v200, v134, v200, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v203, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v202, v203, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v203, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v203, v134, v203, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v205, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v204, v205, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v205, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v205, v134, v205, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v207, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v206, v207, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v207, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v207, v134, v207, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v129, vcc_lo, v129, 1                 // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v130, v130, s[sgprStrideC1J]          // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v131, v131, s[sgprStrideD1J]          // Move coutRowPtrD to next row
v_cmp_lt_u32 s40, v128, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v209, v130, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v208, v209, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v209, v131, v128, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v209, v134, v209, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v132, vcc_lo, v128, 1                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v211, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v210, v211, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v211, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v211, v134, v211, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v132, vcc_lo, v128, 2                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v213, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v212, v213, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v213, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v213, v134, v213, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v132, vcc_lo, v128, 3                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v215, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v214, v215, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v215, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v215, v134, v215, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,0) */
v_add_co_u32 v132, vcc_lo, v128, 8                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v217, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v216, v217, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v217, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v217, v134, v217, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,1) */
v_add_co_u32 v132, vcc_lo, v128, 9                 // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v219, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v218, v219, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v219, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v219, v134, v219, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,2) */
v_add_co_u32 v132, vcc_lo, v128, 10                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v221, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v220, v221, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v221, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v221, v134, v221, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,1,3) */
v_add_co_u32 v132, vcc_lo, v128, 11                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v223, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v222, v223, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v223, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v223, v134, v223, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,0) */
v_add_co_u32 v132, vcc_lo, v128, 16                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v225, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v224, v225, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v225, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v225, v134, v225, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,1) */
v_add_co_u32 v132, vcc_lo, v128, 17                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v227, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v226, v227, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v227, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v227, v134, v227, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,2) */
v_add_co_u32 v132, vcc_lo, v128, 18                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v229, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v228, v229, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v229, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v229, v134, v229, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,2,3) */
v_add_co_u32 v132, vcc_lo, v128, 19                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v231, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v230, v231, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v231, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v231, v134, v231, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,0) */
v_add_co_u32 v132, vcc_lo, v128, 24                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v233, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v232, v233, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v233, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v233, v134, v233, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,1) */
v_add_co_u32 v132, vcc_lo, v128, 25                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v235, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v234, v235, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v235, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v235, v134, v235, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,2) */
v_add_co_u32 v132, vcc_lo, v128, 26                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v237, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v236, v237, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v237, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v237, v134, v237, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,3,3) */
v_add_co_u32 v132, vcc_lo, v128, 27                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v239, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v238, v239, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v239, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v239, v134, v239, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,0) */
v_add_co_u32 v132, vcc_lo, v128, 32                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v241, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v240, v241, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v241, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v241, v134, v241, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,1) */
v_add_co_u32 v132, vcc_lo, v128, 33                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v243, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v242, v243, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v243, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v243, v134, v243, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,2) */
v_add_co_u32 v132, vcc_lo, v128, 34                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v245, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v244, v245, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v245, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v245, v134, v245, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,4,3) */
v_add_co_u32 v132, vcc_lo, v128, 35                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v247, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v246, v247, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v247, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v247, v134, v247, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,0) */
v_add_co_u32 v132, vcc_lo, v128, 40                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v249, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v248, v249, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v249, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v249, v134, v249, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,1) */
v_add_co_u32 v132, vcc_lo, v128, 41                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v251, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v250, v251, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v251, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v251, v134, v251, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,2) */
v_add_co_u32 v132, vcc_lo, v128, 42                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v253, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v252, v253, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v253, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v253, v134, v253, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,5,3) */
v_add_co_u32 v132, vcc_lo, v128, 43                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v255, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v254, v255, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v255, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v255, v134, v255, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 4, 2, 0), (0, 4, 2, 1), (0, 4, 2, 2), (0, 4, 2, 3), (0, 5, 2, 0), (0, 5, 2, 1), (0, 5, 2, 2), (0, 5, 2, 3), (0, 6, 2, 0), (0, 6, 2, 1), (0, 6, 2, 2), (0, 6, 2, 3), (0, 7, 2, 0), (0, 7, 2, 1), (0, 7, 2, 2), (0, 7, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 1, 3, 0), (0, 1, 3, 1), (0, 1, 3, 2), (0, 1, 3, 3), (0, 2, 3, 0), (0, 2, 3, 1), (0, 2, 3, 2), (0, 2, 3, 3), (0, 3, 3, 0), (0, 3, 3, 1), (0, 3, 3, 2), (0, 3, 3, 3), (0, 4, 3, 0), (0, 4, 3, 1), (0, 4, 3, 2), (0, 4, 3, 3), (0, 5, 3, 0), (0, 5, 3, 1), (0, 5, 3, 2), (0, 5, 3, 3)] */
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+68] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+76] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+84] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+92] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+69] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+77] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+85] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+93] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+143], s[sgprAlpha], v[vgprValuC+70] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+144], s[sgprAlpha], v[vgprValuC+78] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+145], s[sgprAlpha], v[vgprValuC+86] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+146], s[sgprAlpha], v[vgprValuC+94] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+147], s[sgprAlpha], v[vgprValuC+71] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+148], s[sgprAlpha], v[vgprValuC+79] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+149], s[sgprAlpha], v[vgprValuC+87] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+150], s[sgprAlpha], v[vgprValuC+95] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+151], s[sgprAlpha], v[vgprValuC+96] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+152], s[sgprAlpha], v[vgprValuC+104] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+153], s[sgprAlpha], v[vgprValuC+112] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+154], s[sgprAlpha], v[vgprValuC+120] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+155], s[sgprAlpha], v[vgprValuC+97] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+156], s[sgprAlpha], v[vgprValuC+105] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+157], s[sgprAlpha], v[vgprValuC+113] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+158], s[sgprAlpha], v[vgprValuC+121] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+159], s[sgprAlpha], v[vgprValuC+98] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+160], s[sgprAlpha], v[vgprValuC+106] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+161], s[sgprAlpha], v[vgprValuC+114] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+162], s[sgprAlpha], v[vgprValuC+122] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+163], s[sgprAlpha], v[vgprValuC+99] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+164], s[sgprAlpha], v[vgprValuC+107] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+165], s[sgprAlpha], v[vgprValuC+115] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+166], s[sgprAlpha], v[vgprValuC+123] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+167], s[sgprAlpha], v[vgprValuC+100] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+168], s[sgprAlpha], v[vgprValuC+108] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+169], s[sgprAlpha], v[vgprValuC+116] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+170], s[sgprAlpha], v[vgprValuC+124] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+171], s[sgprAlpha], v[vgprValuC+101] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+172], s[sgprAlpha], v[vgprValuC+109] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+173], s[sgprAlpha], v[vgprValuC+117] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+174], s[sgprAlpha], v[vgprValuC+125] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v175, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v176, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v177, v[vgprValuC+136] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v178, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v179, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v180, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v181, v[vgprValuC+138] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v182, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v183, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v184, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v185, v[vgprValuC+140] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v186, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v187, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v188, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v189, v[vgprValuC+142] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v190, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+143], s[sgprBeta], v191, v[vgprValuC+143] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v143, v[vgprValuC+143]               // convert C to fp16
buffer_store_b16 v143, v192, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+144], s[sgprBeta], v193, v[vgprValuC+144] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v144, v[vgprValuC+144]               // convert C to fp16
buffer_store_b16 v144, v194, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+145], s[sgprBeta], v195, v[vgprValuC+145] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v145, v[vgprValuC+145]               // convert C to fp16
buffer_store_b16 v145, v196, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+146], s[sgprBeta], v197, v[vgprValuC+146] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v146, v[vgprValuC+146]               // convert C to fp16
buffer_store_b16 v146, v198, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+147], s[sgprBeta], v199, v[vgprValuC+147] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v147, v[vgprValuC+147]               // convert C to fp16
buffer_store_b16 v147, v200, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+148], s[sgprBeta], v202, v[vgprValuC+148] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v148, v[vgprValuC+148]               // convert C to fp16
buffer_store_b16 v148, v203, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+149], s[sgprBeta], v204, v[vgprValuC+149] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v149, v[vgprValuC+149]               // convert C to fp16
buffer_store_b16 v149, v205, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+150], s[sgprBeta], v206, v[vgprValuC+150] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v150, v[vgprValuC+150]               // convert C to fp16
buffer_store_b16 v150, v207, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+151], s[sgprBeta], v208, v[vgprValuC+151] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v151, v[vgprValuC+151]               // convert C to fp16
buffer_store_b16 v151, v209, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+152], s[sgprBeta], v210, v[vgprValuC+152] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v152, v[vgprValuC+152]               // convert C to fp16
buffer_store_b16 v152, v211, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+153], s[sgprBeta], v212, v[vgprValuC+153] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v153, v[vgprValuC+153]               // convert C to fp16
buffer_store_b16 v153, v213, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+154], s[sgprBeta], v214, v[vgprValuC+154] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v154, v[vgprValuC+154]               // convert C to fp16
buffer_store_b16 v154, v215, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+155], s[sgprBeta], v216, v[vgprValuC+155] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v155, v[vgprValuC+155]               // convert C to fp16
buffer_store_b16 v155, v217, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+156], s[sgprBeta], v218, v[vgprValuC+156] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v156, v[vgprValuC+156]               // convert C to fp16
buffer_store_b16 v156, v219, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+157], s[sgprBeta], v220, v[vgprValuC+157] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v157, v[vgprValuC+157]               // convert C to fp16
buffer_store_b16 v157, v221, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+158], s[sgprBeta], v222, v[vgprValuC+158] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v158, v[vgprValuC+158]               // convert C to fp16
buffer_store_b16 v158, v223, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+159], s[sgprBeta], v224, v[vgprValuC+159] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v159, v[vgprValuC+159]               // convert C to fp16
buffer_store_b16 v159, v225, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+160], s[sgprBeta], v226, v[vgprValuC+160] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v160, v[vgprValuC+160]               // convert C to fp16
buffer_store_b16 v160, v227, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+161], s[sgprBeta], v228, v[vgprValuC+161] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v161, v[vgprValuC+161]               // convert C to fp16
buffer_store_b16 v161, v229, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+162], s[sgprBeta], v230, v[vgprValuC+162] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v162, v[vgprValuC+162]               // convert C to fp16
buffer_store_b16 v162, v231, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+163], s[sgprBeta], v232, v[vgprValuC+163] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v163, v[vgprValuC+163]               // convert C to fp16
buffer_store_b16 v163, v233, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+164], s[sgprBeta], v234, v[vgprValuC+164] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v164, v[vgprValuC+164]               // convert C to fp16
buffer_store_b16 v164, v235, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+165], s[sgprBeta], v236, v[vgprValuC+165] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v165, v[vgprValuC+165]               // convert C to fp16
buffer_store_b16 v165, v237, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+166], s[sgprBeta], v238, v[vgprValuC+166] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v166, v[vgprValuC+166]               // convert C to fp16
buffer_store_b16 v166, v239, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+167], s[sgprBeta], v240, v[vgprValuC+167] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v167, v[vgprValuC+167]               // convert C to fp16
buffer_store_b16 v167, v241, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+168], s[sgprBeta], v242, v[vgprValuC+168] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v168, v[vgprValuC+168]               // convert C to fp16
buffer_store_b16 v168, v243, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+169], s[sgprBeta], v244, v[vgprValuC+169] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v169, v[vgprValuC+169]               // convert C to fp16
buffer_store_b16 v169, v245, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+170], s[sgprBeta], v246, v[vgprValuC+170] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v170, v[vgprValuC+170]               // convert C to fp16
buffer_store_b16 v170, v247, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+171], s[sgprBeta], v248, v[vgprValuC+171] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v171, v[vgprValuC+171]               // convert C to fp16
buffer_store_b16 v171, v249, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+172], s[sgprBeta], v250, v[vgprValuC+172] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v172, v[vgprValuC+172]               // convert C to fp16
buffer_store_b16 v172, v251, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+173], s[sgprBeta], v252, v[vgprValuC+173] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v173, v[vgprValuC+173]               // convert C to fp16
buffer_store_b16 v173, v253, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+174], s[sgprBeta], v254, v[vgprValuC+174] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v174, v[vgprValuC+174]               // convert C to fp16
buffer_store_b16 v174, v255, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,6,3,0:vw1); (0,6,3,1:vw1); (0,6,3,2:vw1); (0,6,3,3:vw1); (0,7,3,0:vw1); (0,7,3,1:vw1); (0,7,3,2:vw1); (0,7,3,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v134, BufferOOB
/* (d1,vc1,d0,vc0)=(0,3,6,0) */
v_add_co_u32 v132, vcc_lo, v128, 48                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v144, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, v134, v144, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v143, v144, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v144, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v144, v134, v144, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,1) */
v_add_co_u32 v132, vcc_lo, v128, 49                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v146, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, v134, v146, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v145, v146, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v146, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v146, v134, v146, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,2) */
v_add_co_u32 v132, vcc_lo, v128, 50                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v148, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, v134, v148, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v147, v148, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v148, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v148, v134, v148, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,6,3) */
v_add_co_u32 v132, vcc_lo, v128, 51                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v150, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, v134, v150, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v149, v150, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v150, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v150, v134, v150, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,0) */
v_add_co_u32 v132, vcc_lo, v128, 56                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v152, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v152, v134, v152, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v151, v152, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v152, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v152, v134, v152, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,1) */
v_add_co_u32 v132, vcc_lo, v128, 57                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v154, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, v134, v154, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v153, v154, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v154, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v154, v134, v154, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,2) */
v_add_co_u32 v132, vcc_lo, v128, 58                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v156, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v156, v134, v156, s42                // LDC clip if OOB. offset
buffer_load_d16_b16 v155, v156, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v156, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v156, v134, v156, s42                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,7,3) */
v_add_co_u32 v132, vcc_lo, v128, 59                // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v132, s[sgprSizeI]               // coord0 < size0
v_cmp_lt_u32 s42, v129, s[sgprSizeJ]               // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v158, v130, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, v134, v158, s42                // LDC clip if OOB. offset
buffer_load_d16_hi_b16 v157, v158, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v158, v131, v132, 0x1               // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v158, v134, v158, s42                // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 6, 3, 0), (0, 6, 3, 1), (0, 6, 3, 2), (0, 6, 3, 3), (0, 7, 3, 0), (0, 7, 3, 1), (0, 7, 3, 2), (0, 7, 3, 3)] */
v_mul_f32 v[vgprValuC+135], s[sgprAlpha], v[vgprValuC+102] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+136], s[sgprAlpha], v[vgprValuC+110] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+137], s[sgprAlpha], v[vgprValuC+118] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+138], s[sgprAlpha], v[vgprValuC+126] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+139], s[sgprAlpha], v[vgprValuC+103] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+140], s[sgprAlpha], v[vgprValuC+111] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+141], s[sgprAlpha], v[vgprValuC+119] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+142], s[sgprAlpha], v[vgprValuC+127] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+135], s[sgprBeta], v143, v[vgprValuC+135] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v135, v[vgprValuC+135]               // convert C to fp16
buffer_store_b16 v135, v144, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+136], s[sgprBeta], v145, v[vgprValuC+136] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v136, v[vgprValuC+136]               // convert C to fp16
buffer_store_b16 v136, v146, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+137], s[sgprBeta], v147, v[vgprValuC+137] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v137, v[vgprValuC+137]               // convert C to fp16
buffer_store_b16 v137, v148, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+138], s[sgprBeta], v149, v[vgprValuC+138] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v138, v[vgprValuC+138]               // convert C to fp16
buffer_store_b16 v138, v150, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+139], s[sgprBeta], v151, v[vgprValuC+139] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v139, v[vgprValuC+139]               // convert C to fp16
buffer_store_b16 v139, v152, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+140], s[sgprBeta], v153, v[vgprValuC+140] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v140, v[vgprValuC+140]               // convert C to fp16
buffer_store_b16 v140, v154, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+141], s[sgprBeta], v155, v[vgprValuC+141] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v141, v[vgprValuC+141]               // convert C to fp16
buffer_store_b16 v141, v156, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+142], s[sgprBeta], v157, v[vgprValuC+142] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v142, v[vgprValuC+142]               // convert C to fp16
buffer_store_b16 v142, v158, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
