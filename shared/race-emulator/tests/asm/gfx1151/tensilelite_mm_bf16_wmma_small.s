
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA2_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA256_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1
.globl Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA2_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA256_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1
.p2align 8
.type Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA2_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA256_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA2_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA256_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 48 // vgprs
  .amdhsa_next_free_sgpr 62 // sgprs
  .amdhsa_group_segment_fixed_size 1638 // lds bytes
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
/* Num SGPR   =62 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 1 */
/* SubGroup= 2 x 16 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=2, GlobalReadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
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
  - .name: Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA2_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA256_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1
    .symbol: 'Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA2_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA256_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd'
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
        .value_type:      bf16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      bf16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      bf16
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
    .group_segment_fixed_size:   1638
    .kernarg_segment_align:      8
    .kernarg_segment_size:       104
    .max_flat_workgroup_size:    32
    .private_segment_fixed_size: 0
    .sgpr_count:                 62
    .sgpr_spill_count:           0
    .vgpr_count:                 48
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA2_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA256_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1:
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
/* ValuC range: [0-8), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 16
.set vgprLocalWriteAddrA, 13
.set vgprLocalWriteAddrB, 14
.set vgprGlobalReadOffsetA, 8
.set vgprGlobalReadOffsetB, 12
.set vgprLocalReadAddrA, 41
.set vgprLocalReadAddrB, 15
.set vgprSerial, 42

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuA_X0_I0_D0_PACK, vgprBase+8
.set vgprValuB_X0_I0_BASE, vgprBase+17
.set vgprG2LA_BASE, vgprBase+0
.set vgprG2LB_BASE, vgprBase+17
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+0
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
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 16
.set MT1, 16
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
.set SrdShiftLeftA, 2
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
.macro GLOBAL_OFFSET_A vgprAddr:req, vgprOffset0I:req, vgprOffsetL:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]   // add prepad for pointer shift
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
s_mov_b32 m0, 0x666                                // LDS clamp at 1638 bytes
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
v_cvt_f64_u32 v[16:17], s49                        // s45 = s[sgprWorkGroup0] / s49
v_rcp_f64 v[16:17], v[16:17]                       // s45 = s[sgprWorkGroup0] / s49
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup0]          // s45 = s[sgprWorkGroup0] / s49
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s45 = s[sgprWorkGroup0] / s49
v_cvt_u32_f64 v16, v[16:17]                        // s45 = s[sgprWorkGroup0] / s49
v_mul_lo_u32 v17, v16, s49                         // s45 = s[sgprWorkGroup0] / s49
v_sub_nc_u32 v18, s[sgprWorkGroup0], v17           // s45 = s[sgprWorkGroup0] / s49
v_cmp_ge_u32 vcc_lo, v18, s49                      // s45 = s[sgprWorkGroup0] / s49
s_mov_b32 exec_lo vcc_lo                           // s45 = s[sgprWorkGroup0] / s49
v_add_nc_u32 v16, v16, 1                           // s45 = s[sgprWorkGroup0] / s49
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v17, v16, s49                         // s45 = s[sgprWorkGroup0] / s49
v_sub_nc_u32 v18, s[sgprWorkGroup0], v17           // s45 = s[sgprWorkGroup0] / s49
v_readfirstlane_b32 s45, v16                       // quotient
v_readfirstlane_b32 s46, v18                       // remainder
s_mul_i32 s45, s45, s49
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s46, s46, s48
/* temp0 = temp0 + temp1 */
s_add_u32 s45, s45, s46
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[16:17], s49                        // s46 = s42 / s49
v_rcp_f64 v[16:17], v[16:17]                       // s46 = s42 / s49
v_cvt_f64_u32 v[18:19], s42                        // s46 = s42 / s49
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s46 = s42 / s49
v_cvt_u32_f64 v16, v[16:17]                        // s46 = s42 / s49
v_mul_lo_u32 v17, v16, s49                         // s46 = s42 / s49
v_sub_nc_u32 v18, s42, v17                         // s46 = s42 / s49
v_cmp_ge_u32 vcc_lo, v18, s49                      // s46 = s42 / s49
s_mov_b32 exec_lo vcc_lo                           // s46 = s42 / s49
v_add_nc_u32 v16, v16, 1                           // s46 = s42 / s49
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s46, v16                       // quotient
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
/* init: add vgpr [16...57) to pool */
/* init: add vgpr [0...8) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 4, v1                            // 1. N offset: nOffset = nIdx * nStride(16)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 0, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s41, 256                                 // LSU offset: stride = lsuStride(16)*(MT0(16) + PAD0(0))
v_mul_lo_u32 v2, s41, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v2, v0, 0x1  // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 0, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s41, 16                                  // LSU offset: stride = lsuStride(16) when umlds==True
v_mul_lo_u32 v0, s41, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 128

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x280, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 3, v[vgprSerial]                 // 1 = Serial / 8
v_and_b32 v0, 7, v[vgprSerial]                     // 0 = Serial % 8
/* tile *= glvw */
v_lshlrev_b32 v0, 1, v0                            // v0 = v0 * 2
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
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v4     // lwAL**(MTA + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 5, v[vgprLocalWriteAddrA] // padding 32 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2     // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrB]        // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 5, v[vgprLocalWriteAddrB] // padding 32 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x280, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=640
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
v_mov_b32 v18, MT0                                 // set MT0 into sgpr
v_mov_b32 v17, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v16, v18                             // v16 = ceil(v17 / v18)
v_rcp_iflag_f32 v16, v16                           // v16 = ceil(v17 / v18)
v_cvt_f32_u32 v19, v17                             // v16 = ceil(v17 / v18)
v_mul_f32 v16, v16, v19                            // v16 = ceil(v17 / v18)
v_cvt_u32_f32 v16, v16                             // v16 = ceil(v17 / v18)
v_mul_u32_u24 v19, v16, v18                        // v16 = ceil(v17 / v18)
v_sub_nc_u32 v19, v17, v19                         // v16 = ceil(v17 / v18)
v_cmp_ne_u32 vcc_lo, v19, 0                        // v16 = ceil(v17 / v18)
v_add_co_ci_u32 v16, vcc_lo, v16, 0, vcc_lo        // ceil
v_mov_b32 v18, MT1                                 // set MT1 into sgpr
v_mov_b32 v17, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v16     // set back to numWorkGroup0
v_cvt_f32_u32 v16, v18                             // v16 = ceil(v17 / v18)
v_rcp_iflag_f32 v16, v16                           // v16 = ceil(v17 / v18)
v_cvt_f32_u32 v19, v17                             // v16 = ceil(v17 / v18)
v_mul_f32 v16, v16, v19                            // v16 = ceil(v17 / v18)
v_cvt_u32_f32 v16, v16                             // v16 = ceil(v17 / v18)
v_mul_u32_u24 v19, v16, v18                        // v16 = ceil(v17 / v18)
v_sub_nc_u32 v19, v17, v19                         // v16 = ceil(v17 / v18)
v_cmp_ne_u32 vcc_lo, v19, 0                        // v16 = ceil(v17 / v18)
v_add_co_ci_u32 v16, vcc_lo, v16, 0, vcc_lo        // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v16     // set back to numWorkGroup1

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
s_lshr_b32 s44, s16, 4                             // s44 = s16 / 16
s_and_b32 s42, 15, s16                             // s42 = s16 % 16
s_addc_u32 s44, s44, 0
s_lshr_b32 s45, s17, 4                             // s45 = s17 / 16
s_and_b32 s42, 15, s17                             // s42 = s17 % 16
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
s_lshr_b32 s44, s16, 4                             // s44 = s16 / 16
s_and_b32 s42, 15, s16                             // s42 = s16 % 16
s_addc_u32 s44, s44, 0
s_lshr_b32 s45, s17, 4                             // s45 = s17 / 16
s_and_b32 s42, 15, s17                             // s42 = s17 % 16
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
/* init: add vgpr [16...57) to pool */
/* init: add vgpr [0...8) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 4, v1                            // 1. N offset: nOffset = nIdx * nStride(16)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 0, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s41, 256                                 // LSU offset: stride = lsuStride(16)*(MT0(16) + PAD0(0))
v_mul_lo_u32 v2, s41, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v2, v0, 0x1  // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 256
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 0, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s41, 16                                  // LSU offset: stride = lsuStride(16) when umlds==True
v_mul_lo_u32 v0, s41, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 128

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x280, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 3, v[vgprSerial]                 // 1 = Serial / 8
v_and_b32 v0, 7, v[vgprSerial]                     // 0 = Serial % 8
/* tile *= glvw */
v_lshlrev_b32 v0, 1, v0                            // v0 = v0 * 2
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
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v4     // lwAL**(MTA + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 32 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 5, v[vgprLocalWriteAddrA] // padding 32 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2     // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrB]        // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 5, v[vgprLocalWriteAddrB] // padding 32 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x280, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=640
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args
v_mov_b32 v18, MT0                                 // set MT0 into sgpr
v_mov_b32 v17, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v16, v18                             // v16 = ceil(v17 / v18)
v_rcp_iflag_f32 v16, v16                           // v16 = ceil(v17 / v18)
v_cvt_f32_u32 v19, v17                             // v16 = ceil(v17 / v18)
v_mul_f32 v16, v16, v19                            // v16 = ceil(v17 / v18)
v_cvt_u32_f32 v16, v16                             // v16 = ceil(v17 / v18)
v_mul_u32_u24 v19, v16, v18                        // v16 = ceil(v17 / v18)
v_sub_nc_u32 v19, v17, v19                         // v16 = ceil(v17 / v18)
v_cmp_ne_u32 vcc_lo, v19, 0                        // v16 = ceil(v17 / v18)
v_add_co_ci_u32 v16, vcc_lo, v16, 0, vcc_lo        // ceil
v_mov_b32 v18, MT1                                 // set MT1 into sgpr
v_mov_b32 v17, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v16     // set back to numWorkGroup0
v_cvt_f32_u32 v16, v18                             // v16 = ceil(v17 / v18)
v_rcp_iflag_f32 v16, v16                           // v16 = ceil(v17 / v18)
v_cvt_f32_u32 v19, v17                             // v16 = ceil(v17 / v18)
v_mul_f32 v16, v16, v19                            // v16 = ceil(v17 / v18)
v_cvt_u32_f32 v16, v16                             // v16 = ceil(v17 / v18)
v_mul_u32_u24 v19, v16, v18                        // v16 = ceil(v17 / v18)
v_sub_nc_u32 v19, v17, v19                         // v16 = ceil(v17 / v18)
v_cmp_ne_u32 vcc_lo, v19, 0                        // v16 = ceil(v17 / v18)
v_add_co_ci_u32 v16, vcc_lo, v16, 0, vcc_lo        // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v16     // set back to numWorkGroup1

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
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 4  // pre-pad to make room for possible pointer shift
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
v_cvt_f64_u32 v[16:17], s40                        // s41 = s[sgprWorkGroup0] / s40
v_rcp_f64 v[16:17], v[16:17]                       // s41 = s[sgprWorkGroup0] / s40
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup0]          // s41 = s[sgprWorkGroup0] / s40
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s41 = s[sgprWorkGroup0] / s40
v_cvt_u32_f64 v16, v[16:17]                        // s41 = s[sgprWorkGroup0] / s40
v_mul_lo_u32 v17, v16, s40                         // s41 = s[sgprWorkGroup0] / s40
v_sub_nc_u32 v18, s[sgprWorkGroup0], v17           // s41 = s[sgprWorkGroup0] / s40
v_cmp_ge_u32 vcc_lo, v18, s40                      // s41 = s[sgprWorkGroup0] / s40
s_mov_b32 exec_lo vcc_lo                           // s41 = s[sgprWorkGroup0] / s40
v_add_nc_u32 v16, v16, 1                           // s41 = s[sgprWorkGroup0] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s41, v16                       // quotient
s_mul_i32 s44, s41, s40                            // quotient * non-magic divisor
s_sub_u32 s44, s[sgprWorkGroup0], s44              // WorkGroup0=remainder
s_mul_i32 s44, s44, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s44, s44, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[16:17], s40                        // s42 = s[sgprNumWorkGroups0] / s40
v_rcp_f64 v[16:17], v[16:17]                       // s42 = s[sgprNumWorkGroups0] / s40
v_cvt_f64_u32 v[18:19], s[sgprNumWorkGroups0]      // s42 = s[sgprNumWorkGroups0] / s40
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s42 = s[sgprNumWorkGroups0] / s40
v_cvt_u32_f64 v16, v[16:17]                        // s42 = s[sgprNumWorkGroups0] / s40
v_mul_lo_u32 v17, v16, s40                         // s42 = s[sgprNumWorkGroups0] / s40
v_sub_nc_u32 v18, s[sgprNumWorkGroups0], v17       // s42 = s[sgprNumWorkGroups0] / s40
v_cmp_ge_u32 vcc_lo, v18, s40                      // s42 = s[sgprNumWorkGroups0] / s40
s_mov_b32 exec_lo vcc_lo                           // s42 = s[sgprNumWorkGroups0] / s40
v_add_nc_u32 v16, v16, 1                           // s42 = s[sgprNumWorkGroups0] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s42, v16                       // quotient
s_mul_i32 s43, s40, s42                            // quotient * non-magic divisor
s_sub_u32 s43, s[sgprNumWorkGroups0], s43          // NumWorkGroups0=remainder
s_cmp_eq_u32 s43, 0                                // remainder == 0 ?
s_cmov_b32 s43, s40                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s41, s42                              // blockId >= numFullBlocks ?
s_cselect_b32 s42, s43, s40
v_cvt_f64_u32 v[16:17], s42                        // s[sgprWorkGroup1] = s44 / s42
v_rcp_f64 v[16:17], v[16:17]                       // s[sgprWorkGroup1] = s44 / s42
v_cvt_f64_u32 v[18:19], s44                        // s[sgprWorkGroup1] = s44 / s42
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s[sgprWorkGroup1] = s44 / s42
v_cvt_u32_f64 v16, v[16:17]                        // s[sgprWorkGroup1] = s44 / s42
v_mul_lo_u32 v17, v16, s42                         // s[sgprWorkGroup1] = s44 / s42
v_sub_nc_u32 v18, s44, v17                         // s[sgprWorkGroup1] = s44 / s42
v_cmp_ge_u32 vcc_lo, v18, s42                      // s[sgprWorkGroup1] = s44 / s42
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s44 / s42
v_add_nc_u32 v16, v16, 1                           // s[sgprWorkGroup1] = s44 / s42
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v17, v16, s42                         // s[sgprWorkGroup1] = s44 / s42
v_sub_nc_u32 v18, s44, v17                         // s[sgprWorkGroup1] = s44 / s42
v_readfirstlane_b32 s[sgprWorkGroup1], v16         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v18         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s42 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s44, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s41, s41, s40                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s41 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s40, s40                                 // WGM
v_cvt_f64_u32 v[16:17], s40                        // s41 = s[sgprWorkGroup1] / s40
v_rcp_f64 v[16:17], v[16:17]                       // s41 = s[sgprWorkGroup1] / s40
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup1]          // s41 = s[sgprWorkGroup1] / s40
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s41 = s[sgprWorkGroup1] / s40
v_cvt_u32_f64 v16, v[16:17]                        // s41 = s[sgprWorkGroup1] / s40
v_mul_lo_u32 v17, v16, s40                         // s41 = s[sgprWorkGroup1] / s40
v_sub_nc_u32 v18, s[sgprWorkGroup1], v17           // s41 = s[sgprWorkGroup1] / s40
v_cmp_ge_u32 vcc_lo, v18, s40                      // s41 = s[sgprWorkGroup1] / s40
s_mov_b32 exec_lo vcc_lo                           // s41 = s[sgprWorkGroup1] / s40
v_add_nc_u32 v16, v16, 1                           // s41 = s[sgprWorkGroup1] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s41, v16                       // quotient
s_mul_i32 s44, s41, s40                            // quotient * non-magic divisor
s_sub_u32 s44, s[sgprWorkGroup1], s44              // WorkGroup1=remainder
s_mul_i32 s44, s44, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s44, s44, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[16:17], s40                        // s42 = s[sgprNumWorkGroups1] / s40
v_rcp_f64 v[16:17], v[16:17]                       // s42 = s[sgprNumWorkGroups1] / s40
v_cvt_f64_u32 v[18:19], s[sgprNumWorkGroups1]      // s42 = s[sgprNumWorkGroups1] / s40
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s42 = s[sgprNumWorkGroups1] / s40
v_cvt_u32_f64 v16, v[16:17]                        // s42 = s[sgprNumWorkGroups1] / s40
v_mul_lo_u32 v17, v16, s40                         // s42 = s[sgprNumWorkGroups1] / s40
v_sub_nc_u32 v18, s[sgprNumWorkGroups1], v17       // s42 = s[sgprNumWorkGroups1] / s40
v_cmp_ge_u32 vcc_lo, v18, s40                      // s42 = s[sgprNumWorkGroups1] / s40
s_mov_b32 exec_lo vcc_lo                           // s42 = s[sgprNumWorkGroups1] / s40
v_add_nc_u32 v16, v16, 1                           // s42 = s[sgprNumWorkGroups1] / s40
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s42, v16                       // quotient
s_mul_i32 s43, s40, s42                            // quotient * non-magic divisor
s_sub_u32 s43, s[sgprNumWorkGroups1], s43          // NumWorkGroups1=remainder
s_cmp_eq_u32 s43, 0                                // remainder == 0 ?
s_cmov_b32 s43, s40                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s41, s42                              // blockId >= numFullBlocks ?
s_cselect_b32 s42, s43, s40
v_cvt_f64_u32 v[16:17], s42                        // s[sgprWorkGroup0] = s44 / s42
v_rcp_f64 v[16:17], v[16:17]                       // s[sgprWorkGroup0] = s44 / s42
v_cvt_f64_u32 v[18:19], s44                        // s[sgprWorkGroup0] = s44 / s42
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s[sgprWorkGroup0] = s44 / s42
v_cvt_u32_f64 v16, v[16:17]                        // s[sgprWorkGroup0] = s44 / s42
v_mul_lo_u32 v17, v16, s42                         // s[sgprWorkGroup0] = s44 / s42
v_sub_nc_u32 v18, s44, v17                         // s[sgprWorkGroup0] = s44 / s42
v_cmp_ge_u32 vcc_lo, v18, s42                      // s[sgprWorkGroup0] = s44 / s42
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup0] = s44 / s42
v_add_nc_u32 v16, v16, 1                           // s[sgprWorkGroup0] = s44 / s42
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v17, v16, s42                         // s[sgprWorkGroup0] = s44 / s42
v_sub_nc_u32 v18, s44, v17                         // s[sgprWorkGroup0] = s44 / s42
v_readfirstlane_b32 s[sgprWorkGroup0], v16         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v18         // remainder
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
v_mov_b32 v6, v0                                   // groA0I_0

/* global read addresses: tile offsets b */
v_mov_b32 v7, v2                                   // groB1J_0

/* global read addresses: unroll offsets a */
v_mov_b32 v16, v1                                  // groAL_0
v_add_co_u32 v17, vcc_lo, 4, v16                   // groAL_1 + LSPA
v_add_co_u32 v18, vcc_lo, 4, v17                   // groAL_2 + LSPA
v_add_co_u32 v19, vcc_lo, 4, v18                   // groAL_3 + LSPA

/* global read addresses: unroll offsets b */
v_mov_b32 v20, v3                                  // groBL_0

/* global read addresses: shift a */
s_mul_i32 s56, s[sgprWorkGroup0], 16               // WorkGroup[01] * MT
s_sub_u32 s56, s[sgprSizeI], s56                   // edge = Size0I - WG*MT
s_sub_u32 s56, s56, 2                              // edge -= margin(2)
v_mov_b32 v21, s56                                 // edge vgpr = Size0I- WG*MT - margin(2)
v_min_i32 v6, v21, v6                              // offset = (offset < edge) ? offset(v6) : edge(v21)

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s59, s[sgprWorkGroup0], 16            // WorkGroup[01] * MT
s_mul_i32 s58, s[sgprWorkGroup0], 16               // WorkGroup[01] * MT
s_and_b32 s56, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s57, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s56, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 4 // s[LoopCounterL] = s[sgprSizesSum] / 16
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v21, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v21, v21                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v22, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v21, v21, v22                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v21, v21                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v22, v21, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v22, s[sgprLoopCounterL], v22         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v22, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v21, 1, v21                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v22, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v22, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v21, v21, 1                           // quotient - 1
v_mul_u32_u24 v22, v21, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v22, s[sgprLoopCounterL], v22         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v21       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v22        // remainder
s_mul_i32 s57, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s56, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s57, s57, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s56, s56, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s56, s56, s57                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s57, s56, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s56, s56, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_mul_hi_u32 s57, s56, s[sgprStrideAL]             // tlu=1, scaled unroll-offset by stride
s_mul_i32 s56, s56, s[sgprStrideAL]                // tlu=1, scaled unroll-offset by stride
s_add_u32 s58, s58, s56                            // accum GsuOffset term to tilestart
s_addc_u32 s59, s59, s57                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s56, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s57, constStrideA0I, s56              // stride x (size-1)
s_mul_i32 s56, constStrideA0I, s56                 // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s56 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s57 // sum tensor size
s_sub_u32 s56, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s57, s[sgprStrideAL], s56             // stride x (size-1)
s_mul_i32 s56, s[sgprStrideAL], s56                // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s56 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s57 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s58 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s59 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 4 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s57, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s56, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s58, s58, s56                            // accum wg term to tilestart
s_addc_u32 s59, s59, s57                           // accum wg term to tilestart
s_lshl_b64 s[58:59], s[58:59], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s58    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s59   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s59, s[sgprWorkGroup1], 16            // WorkGroup[01] * MT
s_mul_i32 s58, s[sgprWorkGroup1], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s59, s58, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s58, s58, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s56, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s57, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s56, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 4 // s[LoopCounterL] = s[sgprSizesSum] / 16
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v21, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v21, v21                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v22, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v21, v21, v22                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v21, v21                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v22, v21, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v22, s[sgprLoopCounterL], v22         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v22, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v21, 1, v21                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v22, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v22, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v21, v21, 1                           // quotient - 1
v_mul_u32_u24 v22, v21, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v22, s[sgprLoopCounterL], v22         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v21       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v22        // remainder
s_mul_i32 s57, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s56, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s57, s57, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s56, s56, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s56, s56, s57                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s57, s56, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s56, s56, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_B_End:
s_add_u32 s58, s58, s56                            // accum GsuOffset term to tilestart
s_addc_u32 s59, s59, s57                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s56, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s57, constStrideBL, s56               // stride x (size-1)
s_mul_i32 s56, constStrideBL, s56                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s56 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s57 // sum tensor size
s_sub_u32 s56, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s57, s[sgprStrideB1J], s56            // stride x (size-1)
s_mul_i32 s56, s[sgprStrideB1J], s56               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s56 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s57 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s58 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s59 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s57, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s56, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s58, s58, s56                            // accum wg term to tilestart
s_addc_u32 s59, s59, s57                           // accum wg term to tilestart
s_lshl_b64 s[58:59], s[58:59], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s58    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s59   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  6, 16, 21 // gROA_0_0_0_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+1,  6, 17, 21 // gROA_0_0_1_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+2,  6, 18, 21 // gROA_0_0_2_0
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+3,  6, 19, 21 // gROA_0_0_3_0
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0, 20,  7, 16 // gROB_0_0_0_0
/* ============================================================= */

/* global read addresses: increments a */
s_and_b32 s59, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s59, s59, DepthU*BpeAGR                  // GSU*DepthU*Bpe
s_and_b32 s58, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cmov_b32 s59, DepthU*BpeAGR                      // DepthU*Bpe if GSUC = 1
s_mul_i32 s[sgprGlobalReadIncsA+0], s59, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_and_b32 s59, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s59, s59, DepthU*BpeBGR                  // GSU*DepthU*Bpe
s_and_b32 s58, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeBGR, s59 // incrB (unrollIdx)
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...8) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [16...41) from pool */
v_mov_b32 v[vgprValuC+0], 0                        // initC
v_mov_b32 v[vgprValuC+1], 0                        // initC
v_mov_b32 v[vgprValuC+2], 0                        // initC
v_mov_b32 v[vgprValuC+3], 0                        // initC
v_mov_b32 v[vgprValuC+4], 0                        // initC
v_mov_b32 v[vgprValuC+5], 0                        // initC
v_mov_b32 v[vgprValuC+6], 0                        // initC
v_mov_b32 v[vgprValuC+7], 0                        // initC
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_and_b32 s58, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s58, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v43, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v43, v43                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v44, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v43, v43, v44                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v43, v43                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v44, v43, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v44, s[sgprLoopCounterL], v44         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v44, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v43, 1, v43                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v44, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v44, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v43, v43, 1                           // quotient - 1
v_mul_u32_u24 v44, v43, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v44, s[sgprLoopCounterL], v44         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v43       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v44        // remainder
s_add_u32 s58, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s58                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s60, s[sgprStaggerU], 0x1f00
s_lshr_b32 s60, s60, 0x8
s_and_b32 s61, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s58, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s59, s58, s60                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s59           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s58, s58, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s59, s58, 1                              // staggerU mask
s_cmp_ge_u32 s58, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s59, 0          // set Mask
s_cmp_eq_u32 s61, 0x0
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s58, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s61, 0x2000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s58, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s61, 0x4000
s_cbranch_scc0 label_StaggerUMapping_3
s_mov_b32 s58, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s61, 0x6000
s_cbranch_scc0 label_StaggerUMapping_4
s_mul_i32 s59, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s58, s58, s59
s_add_u32 s58, s58, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s61, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s58, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s58 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s60 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s59, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s58, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s58        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s59       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s58 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s59 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s59, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s58, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s58        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s59       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s58 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s59 // limit -= inc)
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
buffer_load_b32 v[vgprG2LA+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b32 v[vgprG2LA+1], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b32 v[vgprG2LA+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s58, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s59, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s58        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s59       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s58 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s59 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s58, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s59, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s58        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s59       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s58 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s59 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt vmcnt(0)                                 // 5wait for global read
// Skip barrier: NumThreads=32PGR=0, prior iter done reading lds

/* local write a */
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:128 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 128
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:288 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 288
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:416 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 416

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
s_waitcnt lgkmcnt(0)                               // 2prefetch wait for local write
// Skip barrier: NumThreads=32

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:0  */
/*  mfmaIndex:0  */
ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:160 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:224 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:288 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=8 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=9 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=10 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=11 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:416 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=12 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=13 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:480 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=14 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=15 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
v_wmma_f32_16x16x16_bf16 v[0:7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[0:7] // left value = v[0+0:7+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=16 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=2 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/* Tail: add ValuA/B vgpr buffer [16...41) to pool */

/* Tail: add address/G2L vgpr [41...41) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (4,4,0) */
.set vgprG2LA_BASE, 16
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 20
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_and_b32 s58, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s59, s[sgprSizesSum], 4                 // s59 = s[sgprSizesSum] / 16
s_and_b32 s60, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v24, s60                             // s58 = s59 / s60
v_rcp_iflag_f32 v24, v24                           // s58 = s59 / s60
v_cvt_f32_u32 v25, s59                             // s58 = s59 / s60
v_mul_f32 v24, v24, v25                            // s58 = s59 / s60
v_cvt_u32_f32 v24, v24                             // s58 = s59 / s60
v_mul_u32_u24 v25, v24, s60                        // s58 = s59 / s60
v_sub_nc_u32 v25, s59, v25                         // s58 = s59 / s60
v_cmp_eq_u32 vcc_lo, v25, s60                      // s58 = s59 / s60
s_mov_b32 exec_lo vcc_lo                           // s58 = s59 / s60
v_add_nc_u32 v24, 1, v24                           // s58 = s59 / s60
v_mov_b32 v25, 0                                   // s[sgprGSUSumIdx+1] = s59 % s60
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v25, s60                      // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v24, v24, 1                           // quotient - 1
v_mul_u32_u24 v25, v24, s60                        // re-calculate remainder
v_sub_nc_u32 v25, s59, v25                         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s58, v24                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v25        // remainder
s_sub_u32 s59, s60, 1                              // GSU-1
s_cmp_eq_u32 s58, 0                                // quotient == 0
s_cselect_b32 s58, s[sgprGSUSumIdx+1], s59         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s58                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s58, 2, s[sgprStaggerUIter]
s_cmp_ge_i32 s58, 0
s_cbranch_scc0 label_Negative_T8JHFHKM7BO5OHXW
s_mul_hi_u32 s59, s58, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s58, s58, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_YSQ29IP70005TTFS
label_Negative_T8JHFHKM7BO5OHXW:
s_abs_i32 s58, s58
s_mul_hi_u32 s59, s58, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s58, s58, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s58, s58, 0xffffffff
s_xor_b32 s59, s59, 0xffffffff
s_add_u32 s58, s58, 0x1
s_addc_u32 s59, s59, 0
label_MultiplyDone_YSQ29IP70005TTFS:
s_sub_u32 s58, s58, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s59, s59, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s58        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s59       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s58 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s59 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s58, 2, s[sgprStaggerUIter]
s_cmp_ge_i32 s58, 0
s_cbranch_scc0 label_Negative_S4FDBQ587JJL6NOU
s_mul_hi_u32 s59, s58, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s58, s58, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_L43KTIIJOOEN7J6P
label_Negative_S4FDBQ587JJL6NOU:
s_abs_i32 s58, s58
s_mul_hi_u32 s59, s58, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s58, s58, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s58, s58, 0xffffffff
s_xor_b32 s59, s59, 0xffffffff
s_add_u32 s58, s58, 0x1
s_addc_u32 s59, s59, 0
label_MultiplyDone_L43KTIIJOOEN7J6P:
s_sub_u32 s58, s58, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s59, s59, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s58        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s59       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s58 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s59 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* Tail global read A */
/* g2l=0, load component 0 */
buffer_load_d16_b16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_d16_hi_b16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=1, load component 0 */
buffer_load_d16_b16 v[vgprG2LA+1+0], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=1, load component 1 */
buffer_load_d16_hi_b16 v[vgprG2LA+1+0], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=2, load component 0 */
buffer_load_d16_b16 v[vgprG2LA+2+0], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=2, load component 1 */
buffer_load_d16_hi_b16 v[vgprG2LA+2+0], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=3, load component 0 */
buffer_load_d16_b16 v[vgprG2LA+3+0], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=3, load component 1 */
buffer_load_d16_hi_b16 v[vgprG2LA+3+0], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value

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
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip barrier: NumThreads=32

/* local write a */
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:128 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 128
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:288 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 288
ds_store_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:416 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 416

/* local write b */
ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip barrier: NumThreads=32
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 16
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X0_I0_D0_PACK, 24
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuB_X0_I0_BASE, 32
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:160 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:224 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:288 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=8 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+4], v[vgprLocalReadAddrA] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=9 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:352 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=10 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+5], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=11 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:416 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=12 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+6], v[vgprLocalReadAddrA] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=13 oIdx=0 buffer=0 iui=0
ds_load_u16 v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:480 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=14 oIdx=0 buffer=0 iui=0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+7], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=15 oIdx=0 buffer=0 iui=0

/* local read b */
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0

/* local read inc a */
/* Adding additional 64 pad since cumulative inc has reached 256 */
s_mov_b32 s58, 576                                 // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc_lo, s58, v[vgprLocalReadAddrA+0] // lrA += 512 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s58, 32                                  // inc
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, s58, v[vgprLocalReadAddrB+0] // lrB += 32 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
s_sub_i32 s58, s[sgprLoopCounterL], 1              // calculate 64bit groups index
s_lshr_b32 s59, s58, 2                             // calculate 64bit groups index
s_and_b32 s58, s58, 3                              // calculate shift value
s_sub_i32 s58, 3, s58                              // calculate shift value
s_lshl_b32 s58, s58, 4                             // calculate shift value
v_cmp_eq_i32 s60, s59, 0                           // handle this 64bit group: part 1
v_lshlrev_b64 v[44:45], s58, v[vgprValuA_X0_I0+0+0:vgprValuA_X0_I0+0+0+1] // shfit for ValuA[0:1]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0], v[vgprValuA_X0_I0+0+0+0], v44, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+1], v[vgprValuA_X0_I0+0+0+1], v45, s60 // shift if in this 64b group
v_lshlrev_b64 v[44:45], s58, v[vgprValuB_X0_I0+0+0:vgprValuB_X0_I0+0+0+1] // shfit for ValuB[0:1]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0], v[vgprValuB_X0_I0+0+0+0], v44, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+1], v[vgprValuB_X0_I0+0+0+1], v45, s60 // shift if in this 64b group
v_cmp_eq_i32 s60, s59, 1                           // handle this 64bit group: part 1
v_lshlrev_b64 v[44:45], s58, v[vgprValuA_X0_I0+0+2:vgprValuA_X0_I0+0+2+1] // shfit for ValuA[2:3]
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+0], v[vgprValuA_X0_I0+0+2+0], v44, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+1], v[vgprValuA_X0_I0+0+2+1], v45, s60 // shift if in this 64b group
v_lshlrev_b64 v[44:45], s58, v[vgprValuB_X0_I0+0+2:vgprValuB_X0_I0+0+2+1] // shfit for ValuB[2:3]
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+0], v[vgprValuB_X0_I0+0+2+0], v44, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+1], v[vgprValuB_X0_I0+0+2+1], v45, s60 // shift if in this 64b group
v_cmp_lt_i32 s60, s59, 1                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+0], v[vgprValuA_X0_I0+0+2+0], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+2+1], v[vgprValuA_X0_I0+0+2+1], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+0], v[vgprValuB_X0_I0+0+2+0], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+2+1], v[vgprValuB_X0_I0+0+2+1], 0, s60 // shift if in this 64b group
v_cmp_eq_i32 s60, s59, 2                           // handle this 64bit group: part 1
v_lshlrev_b64 v[44:45], s58, v[vgprValuA_X0_I0+0+4:vgprValuA_X0_I0+0+4+1] // shfit for ValuA[4:5]
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+0], v[vgprValuA_X0_I0+0+4+0], v44, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+1], v[vgprValuA_X0_I0+0+4+1], v45, s60 // shift if in this 64b group
v_lshlrev_b64 v[44:45], s58, v[vgprValuB_X0_I0+0+4:vgprValuB_X0_I0+0+4+1] // shfit for ValuB[4:5]
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+0], v[vgprValuB_X0_I0+0+4+0], v44, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+1], v[vgprValuB_X0_I0+0+4+1], v45, s60 // shift if in this 64b group
v_cmp_lt_i32 s60, s59, 2                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+0], v[vgprValuA_X0_I0+0+4+0], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+4+1], v[vgprValuA_X0_I0+0+4+1], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+0], v[vgprValuB_X0_I0+0+4+0], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+4+1], v[vgprValuB_X0_I0+0+4+1], 0, s60 // shift if in this 64b group
v_cmp_eq_i32 s60, s59, 3                           // handle this 64bit group: part 1
v_lshlrev_b64 v[44:45], s58, v[vgprValuA_X0_I0+0+6:vgprValuA_X0_I0+0+6+1] // shfit for ValuA[6:7]
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+0], v[vgprValuA_X0_I0+0+6+0], v44, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+1], v[vgprValuA_X0_I0+0+6+1], v45, s60 // shift if in this 64b group
v_lshlrev_b64 v[44:45], s58, v[vgprValuB_X0_I0+0+6:vgprValuB_X0_I0+0+6+1] // shfit for ValuB[6:7]
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+0], v[vgprValuB_X0_I0+0+6+0], v44, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+1], v[vgprValuB_X0_I0+0+6+1], v45, s60 // shift if in this 64b group
v_cmp_lt_i32 s60, s59, 3                           // handle this 64bit group: part 2
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+0], v[vgprValuA_X0_I0+0+6+0], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuA_X0_I0+0+6+1], v[vgprValuA_X0_I0+0+6+1], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+0], v[vgprValuB_X0_I0+0+6+0], 0, s60 // shift if in this 64b group
v_cndmask_b32 v[vgprValuB_X0_I0+0+6+1], v[vgprValuB_X0_I0+0+6+1], 0, s60 // shift if in this 64b group
s_nop 1
v_wmma_f32_16x16x16_bf16 v[0:7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[0:7] // left value = v[0+0:7+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x10 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x10 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_X0_I0_D0_PACK, UNDEF
.set vgprValuA_X0_I0_D1, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF

/* Tail: add MISC Vgpr [8...16) to pool */

/* Tail: add LocalReadAddrA Vgpr [41...42) to pool */
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

/* shift vector components d0 */
v_mov_b32 v11, s[sgprWorkGroup0]
v_mul_i32_i24 v11, -0x10, v11                      // wg*MT
v_add_co_u32 v11, vcc_lo, s[sgprSizesFree+0], v11  // wgMT = Size - wg*MT
v_mov_b32 v12, 0x10                                // MT
v_cmp_lt_u32 s8, v11, v12                          // wgMT < MT
v_cndmask_b32 v11, v12, v11, s8                    // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v13, 5, v[vgprSerial]                // 13 = Serial / 32
v_and_b32 v13, 0, v13                              // v13 = v13 % 1
v_lshrrev_b32 v14, 4, v11                          // 14 = 11 / 16
v_and_b32 v14, 0, v14                              // v14 = v14 % 1
v_cmp_eq_u32 s8, v14, v13                          // wave_id == block_belong_to_wave?
v_cndmask_b32 v11, v12, v11, s8                    // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(1)) */
v_lshrrev_b32 v12, 1, v11                          // 12 = 11 / 2
v_lshlrev_b32 v14, 3, v13                          // v14 = v13 * 8
v_sub_nc_u32 v12, v12, v14

/* gbReg: glvw block id */
v_lshrrev_b32 v14, 1, v11                          // 14 = 11 / 2

/* tgbReg: glvw block id */
v_lshrrev_b32 v15, 4, v[vgprSerial]                // 15 = Serial / 16
v_and_b32 v15, 1, v15                              // v15 = v15 % 2
                                                   // v15 = v15 * 1 (multiplier is 1, do nothing)
v_lshrrev_b32 v15, 1, v15                          // 15 = 15 / 2
v_lshlrev_b32 v13, 3, v13                          // v13 = v13 * 8
v_add_co_u32 v15, vcc_lo, v13, v15                 // tgbReg = (tid_coal * continOut) / GLVW
v_sub_nc_u32 v14, v14, v15

/* vwReg: glvw in which vw block? */
v_and_b32 v13, 0, v11                              // permute register between threads
v_lshrrev_b32 v13, 1, v13                          // permute register between threads

/* rReg : reminder of M_size % GlobalReadVectorWidth */
v_and_b32 v15, 1, v11                              // v15 = v11 % 2
v_cmp_eq_u32 vcc_lo, v15, 0x1                      // wgMT%VW == 1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1 // branch to shift d0 r=1

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0

/******************************************/
/* shift d0 r=1                           */
/******************************************/
label_ShiftVectorComponents0_GLVW1:
v_cmp_eq_u32 vcc_lo, v12, 0x0

/* branch to shift d0 r1 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0
v_cmp_eq_u32 vcc_lo, v12, 0x1

/* branch to shift d0 r1 mb1 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM1
v_cmp_eq_u32 vcc_lo, v12, 0x2

/* branch to shift d0 r1 mb2 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM2
v_cmp_eq_u32 vcc_lo, v12, 0x3

/* branch to shift d0 r1 mb3 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM3
v_cmp_eq_u32 vcc_lo, v12, 0x4

/* branch to shift d0 r1 mb4 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM4
v_cmp_eq_u32 vcc_lo, v12, 0x5

/* branch to shift d0 r1 mb5 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM5
v_cmp_eq_u32 vcc_lo, v12, 0x6

/* branch to shift d0 r1 mb6 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM6
v_cmp_eq_u32 vcc_lo, v12, 0x7

/* branch to shift d0 r1 mb7 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM7

/******************************************/
/* shift d0 r=1 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0:  /// r1 mb0
v_cmp_eq_u32 vcc_lo, v13, 0x0

/* branch to shift d0 r1 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0_VW0

/******************************************/
/* shift d0 r=1 mb=1                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM1:  /// r1 mb1
v_cmp_eq_u32 vcc_lo, v13, 0x0

/* branch to shift d0 r1 mb1 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM1_VW0

/******************************************/
/* shift d0 r=1 mb=2                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM2:  /// r1 mb2
v_cmp_eq_u32 vcc_lo, v13, 0x0

/* branch to shift d0 r1 mb2 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM2_VW0

/******************************************/
/* shift d0 r=1 mb=3                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM3:  /// r1 mb3
v_cmp_eq_u32 vcc_lo, v13, 0x0

/* branch to shift d0 r1 mb3 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM3_VW0

/******************************************/
/* shift d0 r=1 mb=4                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM4:  /// r1 mb4
v_cmp_eq_u32 vcc_lo, v13, 0x0

/* branch to shift d0 r1 mb4 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM4_VW0

/******************************************/
/* shift d0 r=1 mb=5                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM5:  /// r1 mb5
v_cmp_eq_u32 vcc_lo, v13, 0x0

/* branch to shift d0 r1 mb5 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM5_VW0

/******************************************/
/* shift d0 r=1 mb=6                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM6:  /// r1 mb6
v_cmp_eq_u32 vcc_lo, v13, 0x0

/* branch to shift d0 r1 mb6 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM6_VW0

/******************************************/
/* shift d0 r=1 mb=7                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM7:  /// r1 mb7
v_cmp_eq_u32 vcc_lo, v13, 0x0

/* branch to shift d0 r1 mb7 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM7_VW0

/******************************************/
/* shift d0 r=1 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0_VW0:  /// r1 mb0 vw0
s_mov_b32 s8, 0
v_cmp_eq_u32 s8, v14, s8                           // is thread in edge glvw region
s_mov_b32 exec_lo s8                               // is thread in edge glvw region
v_and_b32 v8, 31, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v8, 2, v8                            // permute register between threads
v_mov_b32 v15, v0                                  // glvw 1 mb 0 tt1 0 r 0
ds_bpermute_b32 v15, v8, v15 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
s_waitcnt_vscnt null 0                             // (Wait all)
v_mov_b32 v0, v15
s_mov_b32 s8, 0xFFFFFFFF                           // to restore all threads active
s_or_saveexec_b32 vcc_lo, s8                       // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=1 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM1_VW0:  /// r1 mb1 vw0
s_mov_b32 s8, 1
v_cmp_eq_u32 s8, v14, s8                           // is thread in edge glvw region
s_mov_b32 exec_lo s8                               // is thread in edge glvw region
v_and_b32 v8, 31, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v8, 2, v8                            // permute register between threads
v_mov_b32 v15, v1                                  // glvw 1 mb 1 tt1 0 r 0
ds_bpermute_b32 v15, v8, v15 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
s_waitcnt_vscnt null 0                             // (Wait all)
v_mov_b32 v1, v15
s_mov_b32 s8, 0xFFFFFFFF                           // to restore all threads active
s_or_saveexec_b32 vcc_lo, s8                       // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=2 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM2_VW0:  /// r1 mb2 vw0
s_mov_b32 s8, 2
v_cmp_eq_u32 s8, v14, s8                           // is thread in edge glvw region
s_mov_b32 exec_lo s8                               // is thread in edge glvw region
v_and_b32 v8, 31, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v8, 2, v8                            // permute register between threads
v_mov_b32 v15, v2                                  // glvw 1 mb 2 tt1 0 r 0
ds_bpermute_b32 v15, v8, v15 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
s_waitcnt_vscnt null 0                             // (Wait all)
v_mov_b32 v2, v15
s_mov_b32 s8, 0xFFFFFFFF                           // to restore all threads active
s_or_saveexec_b32 vcc_lo, s8                       // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=3 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM3_VW0:  /// r1 mb3 vw0
s_mov_b32 s8, 3
v_cmp_eq_u32 s8, v14, s8                           // is thread in edge glvw region
s_mov_b32 exec_lo s8                               // is thread in edge glvw region
v_and_b32 v8, 31, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v8, 2, v8                            // permute register between threads
v_mov_b32 v15, v3                                  // glvw 1 mb 3 tt1 0 r 0
ds_bpermute_b32 v15, v8, v15 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
s_waitcnt_vscnt null 0                             // (Wait all)
v_mov_b32 v3, v15
s_mov_b32 s8, 0xFFFFFFFF                           // to restore all threads active
s_or_saveexec_b32 vcc_lo, s8                       // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=4 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM4_VW0:  /// r1 mb4 vw0
s_mov_b32 s8, 4
v_cmp_eq_u32 s8, v14, s8                           // is thread in edge glvw region
s_mov_b32 exec_lo s8                               // is thread in edge glvw region
v_and_b32 v8, 31, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v8, 2, v8                            // permute register between threads
v_mov_b32 v15, v4                                  // glvw 1 mb 4 tt1 0 r 0
ds_bpermute_b32 v15, v8, v15 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
s_waitcnt_vscnt null 0                             // (Wait all)
v_mov_b32 v4, v15
s_mov_b32 s8, 0xFFFFFFFF                           // to restore all threads active
s_or_saveexec_b32 vcc_lo, s8                       // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=5 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM5_VW0:  /// r1 mb5 vw0
s_mov_b32 s8, 5
v_cmp_eq_u32 s8, v14, s8                           // is thread in edge glvw region
s_mov_b32 exec_lo s8                               // is thread in edge glvw region
v_and_b32 v8, 31, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v8, 2, v8                            // permute register between threads
v_mov_b32 v15, v5                                  // glvw 1 mb 5 tt1 0 r 0
ds_bpermute_b32 v15, v8, v15 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
s_waitcnt_vscnt null 0                             // (Wait all)
v_mov_b32 v5, v15
s_mov_b32 s8, 0xFFFFFFFF                           // to restore all threads active
s_or_saveexec_b32 vcc_lo, s8                       // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=6 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM6_VW0:  /// r1 mb6 vw0
s_mov_b32 s8, 6
v_cmp_eq_u32 s8, v14, s8                           // is thread in edge glvw region
s_mov_b32 exec_lo s8                               // is thread in edge glvw region
v_and_b32 v8, 31, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v8, 2, v8                            // permute register between threads
v_mov_b32 v15, v6                                  // glvw 1 mb 6 tt1 0 r 0
ds_bpermute_b32 v15, v8, v15 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
s_waitcnt_vscnt null 0                             // (Wait all)
v_mov_b32 v6, v15
s_mov_b32 s8, 0xFFFFFFFF                           // to restore all threads active
s_or_saveexec_b32 vcc_lo, s8                       // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=1 mb=7 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM7_VW0:  /// r1 mb7 vw0
s_mov_b32 s8, 7
v_cmp_eq_u32 s8, v14, s8                           // is thread in edge glvw region
s_mov_b32 exec_lo s8                               // is thread in edge glvw region
v_and_b32 v8, 31, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v8, 2, v8                            // permute register between threads
v_mov_b32 v15, v7                                  // glvw 1 mb 7 tt1 0 r 0
ds_bpermute_b32 v15, v8, v15 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
s_waitcnt_vscnt null 0                             // (Wait all)
v_mov_b32 v7, v15
s_mov_b32 s8, 0xFFFFFFFF                           // to restore all threads active
s_or_saveexec_b32 vcc_lo, s8                       // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0

label_ShiftVectorComponents0_GLVW0:  /// end shift0

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v12, 5, v[vgprSerial]                // 12 = Serial / 32
v_lshrrev_b32 v13, 0, v12                          // 13 = 12 / 1
v_mul_lo_u32 v9, 0x10, v13                         // wave coordination offset 1
v_and_b32 v13, 15, v[vgprSerial]                   // v13 = v[vgprSerial] % 16
v_add_lshl_u32 v9, v13, v9, 0                      // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v10, v9, s[sgprStrideC1J]             //  offset 1
v_mul_lo_u32 v11, v9, s[sgprStrideD1J]             //  offset 1
v_and_b32 v13, 0, v12                              // v13 = v12 % 1
v_mul_lo_u32 v13, 0x10, v13                        // wave coordination offset 0
v_and_b32 v8, 31, v[vgprSerial]                    // v8 = v[vgprSerial] % 32
v_lshrrev_b32 v8, 4, v8                            // 8 = 8 / 16
                                                   // thread0 * continuous_output (multiplier is 1, do nothing)
v_add_lshl_u32 v8, v13, v8, 0                      // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s8, 16, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v8, s8, v8                            // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 16, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v9, s8, v9                            // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_3                         // branch if GSU == 1
label_GW_B0:
label_GW_B0_FD0:
s_and_b32 s40, 15, s[sgprSizeI]                    // s40 = s[sgprSizeI] % 16
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Else            // jump if edges required
s_and_b32 s40, 15, s[sgprSizeJ]                    // s40 = s[sgprSizeJ] % 16
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Then            // jump if edges required
label_GW_B0_FD0_VW1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=26 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
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
v_add_lshl_u32 v19, v11, v8, 0x2                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mov_b32 v[vgprValuC+21], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+7]          // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v21, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v22, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D
buffer_store_b32 v23, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
buffer_store_b32 v24, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D
buffer_store_b32 v25, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:32 // store D
buffer_store_b32 v26, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:40 // store D
buffer_store_b32 v27, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:48 // store D
buffer_store_b32 v28, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:56 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_NonEdgeEnd:
label_GW_B0_FD0_VW1_Else:
label_GW_B0_FD0_VW1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=12 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v27, v11, v8, 0x2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v14, v27, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v12, vcc_lo, v8, 2                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v28, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v14, v28, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v12, vcc_lo, v8, 4                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v29, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v29, v14, v29, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v12, vcc_lo, v8, 6                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v30, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v14, v30, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v12, vcc_lo, v8, 8                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v31, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v14, v31, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v12, vcc_lo, v8, 10                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v32, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v14, v32, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v12, vcc_lo, v8, 12                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v33, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v33, v14, v33, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v12, vcc_lo, v8, 14                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v34, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v14, v34, s42                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mov_b32 v[vgprValuC+19], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+7]          // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_b32 v19, v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v20, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v21, v29, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v22, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v23, v31, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v24, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v25, v33, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_b32 v26, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
s_and_b32 s40, 15, s[sgprSizeI]                    // s40 = s[sgprSizeI] % 16
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Else_1          // jump if edges required
s_and_b32 s40, 15, s[sgprSizeJ]                    // s40 = s[sgprSizeJ] % 16
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_Then_1          // jump if edges required
label_GW_B0_FD0_VW1_NonEdge_1:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=26 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
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
v_add_lshl_u32 v19, v11, v8, 0x1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s8, v[vgprValuC+21], v[vgprValuC+21]   // check Nan
v_bfe_u32 v15, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+21], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v15, v17, s8
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_b16 v21, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+22], v[vgprValuC+22]   // check Nan
v_bfe_u32 v15, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+22], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v15, v17, s8
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_b16 v22, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:4 // store D
v_cmp_u_f32 s8, v[vgprValuC+23], v[vgprValuC+23]   // check Nan
v_bfe_u32 v15, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+23], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v15, v17, s8
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_b16 v23, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D
v_cmp_u_f32 s8, v[vgprValuC+24], v[vgprValuC+24]   // check Nan
v_bfe_u32 v15, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+24], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v15, v17, s8
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_b16 v24, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:12 // store D
v_cmp_u_f32 s8, v[vgprValuC+25], v[vgprValuC+25]   // check Nan
v_bfe_u32 v15, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+25], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v15, v17, s8
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_b16 v25, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D
v_cmp_u_f32 s8, v[vgprValuC+26], v[vgprValuC+26]   // check Nan
v_bfe_u32 v15, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+26], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v15, v17, s8
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_b16 v26, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:20 // store D
v_cmp_u_f32 s8, v[vgprValuC+27], v[vgprValuC+27]   // check Nan
v_bfe_u32 v15, v[vgprValuC+27], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+27], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+27], v15, v17, s8
v_lshrrev_b32 v27, 16, v[vgprValuC+27]             // convert C to bf16
buffer_store_b16 v27, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D
v_cmp_u_f32 s8, v[vgprValuC+28], v[vgprValuC+28]   // check Nan
v_bfe_u32 v15, v[vgprValuC+28], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+28], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+28], v15, v17, s8
v_lshrrev_b32 v28, 16, v[vgprValuC+28]             // convert C to bf16
buffer_store_b16 v28, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:28 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW1_NonEdgeEnd_1:
label_GW_B0_FD0_VW1_Else_1:
label_GW_B0_FD0_VW1_Then_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=12 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v27, v11, v8, 0x1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v14, v27, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v12, vcc_lo, v8, 2                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v28, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v14, v28, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v12, vcc_lo, v8, 4                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v29, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v29, v14, v29, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v12, vcc_lo, v8, 6                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v30, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v14, v30, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v12, vcc_lo, v8, 8                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v31, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v14, v31, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v12, vcc_lo, v8, 10                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v32, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v14, v32, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v12, vcc_lo, v8, 12                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v33, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v33, v14, v33, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v12, vcc_lo, v8, 14                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v34, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v14, v34, s42                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s40, v[vgprValuC+19], v[vgprValuC+19]  // check Nan
v_bfe_u32 v15, v[vgprValuC+19], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+19], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+19], v15, v17, s40
v_lshrrev_b32 v19, 16, v[vgprValuC+19]             // convert C to bf16
buffer_store_b16 v19, v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s40, v[vgprValuC+20], v[vgprValuC+20]  // check Nan
v_bfe_u32 v15, v[vgprValuC+20], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+20], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+20], v15, v17, s40
v_lshrrev_b32 v20, 16, v[vgprValuC+20]             // convert C to bf16
buffer_store_b16 v20, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s40, v[vgprValuC+21], v[vgprValuC+21]  // check Nan
v_bfe_u32 v15, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+21], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v15, v17, s40
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_b16 v21, v29, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s40, v[vgprValuC+22], v[vgprValuC+22]  // check Nan
v_bfe_u32 v15, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+22], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v15, v17, s40
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_b16 v22, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s40, v[vgprValuC+23], v[vgprValuC+23]  // check Nan
v_bfe_u32 v15, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+23], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v15, v17, s40
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_b16 v23, v31, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s40, v[vgprValuC+24], v[vgprValuC+24]  // check Nan
v_bfe_u32 v15, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+24], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v15, v17, s40
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_b16 v24, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s40, v[vgprValuC+25], v[vgprValuC+25]  // check Nan
v_bfe_u32 v15, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+25], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v15, v17, s40
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_b16 v25, v33, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s40, v[vgprValuC+26], v[vgprValuC+26]  // check Nan
v_bfe_u32 v15, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+26], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v15, v17, s40
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_b16 v26, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1:
label_GW_B1_FD0:
s_and_b32 s40, 15, s[sgprSizeI]                    // s40 = s[sgprSizeI] % 16
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_Else            // jump if edges required
s_and_b32 s40, 15, s[sgprSizeJ]                    // s40 = s[sgprSizeJ] % 16
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rem
s_cmpk_gt_u32 s40, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_Then            // jump if edges required
label_GW_B1_FD0_VW1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=12 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v20, v10, v8, 0x1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8
buffer_load_d16_b16 v29, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v30, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:4 // load C
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
buffer_load_d16_b16 v31, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:8 // load C
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
buffer_load_d16_b16 v32, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:12 // load C
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
buffer_load_d16_b16 v33, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:16 // load C
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
buffer_load_d16_b16 v34, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:20 // load C
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
buffer_load_d16_b16 v35, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:24 // load C
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
buffer_load_d16_b16 v36, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:28 // load C
v_add_lshl_u32 v19, v11, v8, 0x1                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 8 - 1 (beta) (interleaved)
v_lshlrev_b32 v12, 16, v29                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+21], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+21], v[vgprValuC+21]   // check Nan
v_bfe_u32 v15, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+21], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v15, v17, s8
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_b16 v21, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 8 - 2 (beta) (interleaved)
v_lshlrev_b32 v12, 16, v30                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+22], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+22], v[vgprValuC+22]   // check Nan
v_bfe_u32 v15, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+22], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v15, v17, s8
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_b16 v22, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:4 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 8 - 3 (beta) (interleaved)
v_lshlrev_b32 v12, 16, v31                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+23], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+23], v[vgprValuC+23]   // check Nan
v_bfe_u32 v15, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+23], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v15, v17, s8
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_b16 v23, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:8 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 8 - 4 (beta) (interleaved)
v_lshlrev_b32 v12, 16, v32                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+24], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+24], v[vgprValuC+24]   // check Nan
v_bfe_u32 v15, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+24], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v15, v17, s8
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_b16 v24, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:12 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 8 - 5 (beta) (interleaved)
v_lshlrev_b32 v12, 16, v33                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+25], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+25], v[vgprValuC+25]   // check Nan
v_bfe_u32 v15, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+25], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v15, v17, s8
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_b16 v25, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:16 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 8 - 6 (beta) (interleaved)
v_lshlrev_b32 v12, 16, v34                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+26], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+26], v[vgprValuC+26]   // check Nan
v_bfe_u32 v15, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+26], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v15, v17, s8
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_b16 v26, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:20 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 8 - 7 (beta) (interleaved)
v_lshlrev_b32 v12, 16, v35                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+27], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+27], v[vgprValuC+27]   // check Nan
v_bfe_u32 v15, v[vgprValuC+27], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+27], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+27], v15, v17, s8
v_lshrrev_b32 v27, 16, v[vgprValuC+27]             // convert C to bf16
buffer_store_b16 v27, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:24 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 8 - 8 (beta) (interleaved)
v_lshlrev_b32 v12, 16, v36                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+28], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+28], v[vgprValuC+28]   // check Nan
v_bfe_u32 v15, v[vgprValuC+28], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+28], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+28], v15, v17, s8
v_lshrrev_b32 v28, 16, v[vgprValuC+28]             // convert C to bf16
buffer_store_b16 v28, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:28 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW1_NonEdgeEnd:
label_GW_B1_FD0_VW1_Else:
label_GW_B1_FD0_VW1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=8 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (0,2,0,0:vw1); (0,3,0,0:vw1); (0,4,0,0:vw1); (0,5,0,0:vw1); (0,6,0,0:vw1); (0,7,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v28, v10, v8, 0x1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v14, v28, s42                   // LDC clip if OOB. offset
buffer_load_d16_b16 v27, v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v28, v11, v8, 0x1                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v14, v28, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v12, vcc_lo, v8, 2                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v30, v10, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v14, v30, s42                   // LDC clip if OOB. offset
buffer_load_d16_b16 v29, v30, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v30, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v14, v30, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
v_add_co_u32 v12, vcc_lo, v8, 4                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v32, v10, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v14, v32, s42                   // LDC clip if OOB. offset
buffer_load_d16_b16 v31, v32, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v32, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v14, v32, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
v_add_co_u32 v12, vcc_lo, v8, 6                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v34, v10, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v14, v34, s42                   // LDC clip if OOB. offset
buffer_load_d16_b16 v33, v34, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v34, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v14, v34, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
v_add_co_u32 v12, vcc_lo, v8, 8                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v36, v10, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v14, v36, s42                   // LDC clip if OOB. offset
buffer_load_d16_b16 v35, v36, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v36, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v14, v36, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
v_add_co_u32 v12, vcc_lo, v8, 10                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v38, v10, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v14, v38, s42                   // LDC clip if OOB. offset
buffer_load_d16_b16 v37, v38, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v38, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v14, v38, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
v_add_co_u32 v12, vcc_lo, v8, 12                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v40, v10, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v14, v40, s42                   // LDC clip if OOB. offset
buffer_load_d16_b16 v39, v40, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v40, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v14, v40, s42                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
v_add_co_u32 v12, vcc_lo, v8, 14                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v43, v10, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v14, v43, s42                   // LDC clip if OOB. offset
buffer_load_d16_b16 v41, v43, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v43, v11, v12, 0x1                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v14, v43, s42                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v12, 16, v27                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+19], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s40, v[vgprValuC+19], v[vgprValuC+19]  // check Nan
v_bfe_u32 v15, v[vgprValuC+19], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+19], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+19], v15, v17, s40
v_lshrrev_b32 v19, 16, v[vgprValuC+19]             // convert C to bf16
buffer_store_b16 v19, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v12, 16, v29                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+20], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s40, v[vgprValuC+20], v[vgprValuC+20]  // check Nan
v_bfe_u32 v15, v[vgprValuC+20], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+20], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+20], v15, v17, s40
v_lshrrev_b32 v20, 16, v[vgprValuC+20]             // convert C to bf16
buffer_store_b16 v20, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v12, 16, v31                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+21], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s40, v[vgprValuC+21], v[vgprValuC+21]  // check Nan
v_bfe_u32 v15, v[vgprValuC+21], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+21], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+21], v15, v17, s40
v_lshrrev_b32 v21, 16, v[vgprValuC+21]             // convert C to bf16
buffer_store_b16 v21, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v12, 16, v33                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+22], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s40, v[vgprValuC+22], v[vgprValuC+22]  // check Nan
v_bfe_u32 v15, v[vgprValuC+22], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+22], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+22], v15, v17, s40
v_lshrrev_b32 v22, 16, v[vgprValuC+22]             // convert C to bf16
buffer_store_b16 v22, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v12, 16, v35                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+23], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s40, v[vgprValuC+23], v[vgprValuC+23]  // check Nan
v_bfe_u32 v15, v[vgprValuC+23], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+23], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+23], v15, v17, s40
v_lshrrev_b32 v23, 16, v[vgprValuC+23]             // convert C to bf16
buffer_store_b16 v23, v36, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v12, 16, v37                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+24], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s40, v[vgprValuC+24], v[vgprValuC+24]  // check Nan
v_bfe_u32 v15, v[vgprValuC+24], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+24], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+24], v15, v17, s40
v_lshrrev_b32 v24, 16, v[vgprValuC+24]             // convert C to bf16
buffer_store_b16 v24, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v12, 16, v39                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+25], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s40, v[vgprValuC+25], v[vgprValuC+25]  // check Nan
v_bfe_u32 v15, v[vgprValuC+25], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+25], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+25], v15, v17, s40
v_lshrrev_b32 v25, 16, v[vgprValuC+25]             // convert C to bf16
buffer_store_b16 v25, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v12, 16, v41                         // cvt bf16 to fp32.
v_fmac_f32 v[vgprValuC+26], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s40, v[vgprValuC+26], v[vgprValuC+26]  // check Nan
v_bfe_u32 v15, v[vgprValuC+26], 16, 1              // Non-Nan case: store lsb of bf16
v_add3_u32 v15, v[vgprValuC+26], v15, v18          // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+26], v15, v17, s40
v_lshrrev_b32 v26, 16, v[vgprValuC+26]             // convert C to bf16
buffer_store_b16 v26, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
