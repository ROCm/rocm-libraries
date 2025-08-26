
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1300"
.text
.protected Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1300_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1
.globl Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1300_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1
.p2align 8
.type Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1300_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1300_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 56 // vgprs
  .amdhsa_next_free_sgpr 76 // sgprs
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
/* Num VGPR   =56 */
/* Num AccVGPR=0 */
/* Num SGPR   =76 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 1 */
/* SubGroup= 2 x 16 */
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
  - .name: Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1300_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1
    .symbol: 'Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1300_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1.kd'
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
    .group_segment_fixed_size:   1638
    .kernarg_segment_align:      8
    .kernarg_segment_size:       104
    .max_flat_workgroup_size:    32
    .private_segment_fixed_size: 0
    .sgpr_count:                 76
    .sgpr_spill_count:           0
    .vgpr_count:                 56
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_HSS_BH_UserArgs_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_ISA1300_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR0_PLR0_PKA0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1:
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
.set vgprBase, 14
.set vgprLocalWriteAddrA, 10
.set vgprLocalWriteAddrB, 11
.set vgprGlobalReadOffsetA, 8
.set vgprGlobalReadOffsetB, 9
.set vgprLocalReadAddrA, 12
.set vgprLocalReadAddrB, 13
.set vgprSerial, 30

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuA_X0_I0_D0_PACK, vgprBase+4
.set vgprValuB_X0_I0_BASE, vgprBase+8
.set vgprValuB_X0_I0_D0_PACK, vgprBase+12
.set vgprG2LA_BASE, vgprBase+0
.set vgprG2LB_BASE, vgprBase+8
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuB_X0_I0_D1, vgprValuB_X0_I0_D0_PACK+0
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
.set SrdShiftLeftA, 1
.set SrdShiftLeftB, 1
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
//.set BufferOOB, 0x80000000
.set BufferOOB, 0x8000000

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
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]   // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 1, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]   // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0], 1, v[\vgprAddr+0]    // offset *= bytes/element
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* workaround for work group id */
s_mov_b32 s[sgprWorkGroup0], ttmp9                 // workaround
s_and_b32 s[sgprWorkGroup1], 0xffff, ttmp7         // workaround
s_lshr_b32 s[sgprWorkGroup2], ttmp7, 0x10

/* Load num of Gemms */
s_load_b32 s39, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s41, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s42, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_wait_kmcnt 0                                     // load args
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
s_wait_kmcnt 0                                     // wait for args to load
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
v_cvt_f32_u32 v6, s49                              // wg//CU_Count
v_rcp_iflag_f32 v6, v6                             // wg//CU_Count
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // wg//CU_Count
v_mul_f32 v6, v6, v7                               // wg//CU_Count
v_cvt_u32_f32 v6, v6                               // wg//CU_Count
v_mul_u32_u24 v7, v6, s49                          // wg//CU_Count
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // wg//CU_Count
v_cmp_eq_u32 vcc_lo, v7, s49                       // wg//CU_Count
s_mov_b32 exec_lo vcc_lo                           // wg//CU_Count
v_add_nc_u32 v6, 1, v6                             // wg//CU_Count
v_mov_b32 v7, 0                                    // wg//CU_Count
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s49                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s49                          // re-calculate remainder
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s45, v6                        // quotient
v_readfirstlane_b32 s46, v7                        // remainder
s_mul_i32 s45, s45, s49
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s46, s46, s48
/* temp0 = temp0 + temp1 */
s_add_u32 s45, s45, s46
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f32_u32 v6, s49                              // WGs//CU_Count
v_rcp_iflag_f32 v6, v6                             // WGs//CU_Count
v_cvt_f32_u32 v7, s42                              // WGs//CU_Count
v_mul_f32 v6, v6, v7                               // WGs//CU_Count
v_cvt_u32_f32 v6, v6                               // WGs//CU_Count
v_mul_u32_u24 v7, v6, s49                          // WGs//CU_Count
v_sub_nc_u32 v7, s42, v7                           // WGs//CU_Count
v_cmp_eq_u32 vcc_lo, v7, s49                       // WGs//CU_Count
s_mov_b32 exec_lo vcc_lo                           // WGs//CU_Count
v_add_nc_u32 v6, 1, v6                             // WGs//CU_Count
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s49                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
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

/* init: add vgpr [14...44) to pool */
/* init: add vgpr [0...8) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

// local read addresses: tile assignments a/b
// lr0I
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 1, v1                                // 1. N offset: nIdx = wtid % MI_N(2)
v_lshlrev_b32 v0, 3, v0
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
// Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 1, v1                            // 5. K offset: kIdx = wtid / (MIN(2) * MIBB(1))
v_lshl_add_u32 v0, v1, 4, v0                       // 5. K offset: lrKOffset = kIdx * mStride(16); 6. offset in wave: lrOffset = bnOffset + lrKOffset

// lr1J
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 1, v2                                // 1. N offset: nIdx = wtid % MI_N(2)
v_lshlrev_b32 v1, 3, v1
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
// Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v2, 1, v2                            // 5. K offset: kIdx = wtid / (MIN(2) * MIBB(1))
v_lshl_add_u32 v1, v2, 4, v1                       // 5. K offset: lrKOffset = kIdx * mStride(16); 6. offset in wave: lrOffset = bnOffset + lrKOffset

// local read addresses: final offsets a
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 0, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s41, 256                                 // LSU offset: stride = lsuStride(16)*(MT0(16) + PAD0(0))
v_mul_lo_u32 v2, s41, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrA], v2, v0, 0x1  // Final Offset: offset = (lro0+lsuoffset)*bpeDS

// local read addresses: final offsets b
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 0, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(16)*(MT1(16) + PAD1(0)) (dup assign opt.)
v_mul_lo_u32 v0, s41, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS

// local read addresses: declare addresses a

// local read addresses: declare addresses b
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)


/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // 0 = Serial / 16
v_and_b32 v1, 15, v[vgprSerial]                    // 1 = Serial % 16
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // 2 = Serial / 16
v_and_b32 v3, 15, v[vgprSerial]                    // 3 = Serial % 16
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/*
// debug: local write address
// local write addresses: first offset a 
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v4     // lwAL**(MTA + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS

// local write addresses: first offset b 
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v5     // lwBL**(MTB + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpeDS
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x200, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=512
*/

// debug: local write address a (new)
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v0
v_add_lshl_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA], 0x1

// debug: local write address b (new)
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v2
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x200, v[vgprLocalWriteAddrB]


s_wait_kmcnt 0                                     // wait for 88/0 bytes of kern args
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
.set sgprScalarGlobalReadOffsetB, 65
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 2  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 2  // pre-pad to make room for possible pointer shift
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
s_and_b32 s72, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s72, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s72, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s72, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v6, s72                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_mul_u32_u24 v7, v6, s72                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_cmp_eq_u32 vcc_lo, v7, s72                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_add_nc_u32 v6, 1, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s72
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s72                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s72                          // re-calculate remainder
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
s_sext_i32_i16 s[sgprWGM], s[sgprWGM]              // Restore WGM
s_cmp_gt_i32 s[sgprWGM], 1                         // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s[sgprWGM], 0                         // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s[sgprWGM], s[sgprWGM]                   // abs(WGM)
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // WGM
v_cmp_eq_u32 vcc_lo, v7, s[sgprWGM]                // WGM
s_mov_b32 exec_lo vcc_lo                           // WGM
v_add_nc_u32 v6, 1, v6                             // WGM
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprWGM]                // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s74, v6                        // quotient
s_mul_i32 s75, s74, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s75, s[sgprWorkGroup0], s75              // WorkGroup0=remainder
s_mul_i32 s75, s75, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s75, s75, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprNumWorkGroups0]            // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_nc_u32 v7, s[sgprNumWorkGroups0], v7         // WGM
v_cmp_eq_u32 vcc_lo, v7, s[sgprWGM]                // WGM
s_mov_b32 exec_lo vcc_lo                           // WGM
v_add_nc_u32 v6, 1, v6                             // WGM
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprWGM]                // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s72, v6                        // quotient
s_mul_i32 s73, s[sgprWGM], s72                     // quotient * non-magic divisor
s_sub_u32 s73, s[sgprNumWorkGroups0], s73          // NumWorkGroups0=remainder
s_cmp_eq_u32 s73, 0                                // remainder == 0 ?
s_cmov_b32 s73, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s74, s72                              // blockId >= numFullBlocks ?
s_cselect_b32 s72, s73, s[sgprWGM]
v_cvt_f32_u32 v6, s72                              // s[sgprWorkGroup1] = s75 / s72
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s75 / s72
v_cvt_f32_u32 v7, s75                              // s[sgprWorkGroup1] = s75 / s72
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s75 / s72
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s75 / s72
v_mul_u32_u24 v7, v6, s72                          // s[sgprWorkGroup1] = s75 / s72
v_sub_nc_u32 v7, s75, v7                           // s[sgprWorkGroup1] = s75 / s72
v_cmp_eq_u32 vcc_lo, v7, s72                       // s[sgprWorkGroup1] = s75 / s72
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup1] = s75 / s72
v_add_nc_u32 v6, 1, v6                             // s[sgprWorkGroup1] = s75 / s72
v_mov_b32 v7, 0                                    // s[sgprWorkGroup0] = s75 % s72
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s72                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s72                          // re-calculate remainder
v_sub_nc_u32 v7, s75, v7                           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v7          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s72 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s75, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s74, s74, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s74 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s[sgprWGM], s[sgprWGM]                   // WGM
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7             // WGM
v_cmp_eq_u32 vcc_lo, v7, s[sgprWGM]                // WGM
s_mov_b32 exec_lo vcc_lo                           // WGM
v_add_nc_u32 v6, 1, v6                             // WGM
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprWGM]                // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s74, v6                        // quotient
s_mul_i32 s75, s74, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s75, s[sgprWorkGroup1], s75              // WorkGroup1=remainder
s_mul_i32 s75, s75, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s75, s75, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprNumWorkGroups1]            // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_nc_u32 v7, s[sgprNumWorkGroups1], v7         // WGM
v_cmp_eq_u32 vcc_lo, v7, s[sgprWGM]                // WGM
s_mov_b32 exec_lo vcc_lo                           // WGM
v_add_nc_u32 v6, 1, v6                             // WGM
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s[sgprWGM]                // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s72, v6                        // quotient
s_mul_i32 s73, s[sgprWGM], s72                     // quotient * non-magic divisor
s_sub_u32 s73, s[sgprNumWorkGroups1], s73          // NumWorkGroups1=remainder
s_cmp_eq_u32 s73, 0                                // remainder == 0 ?
s_cmov_b32 s73, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s74, s72                              // blockId >= numFullBlocks ?
s_cselect_b32 s72, s73, s[sgprWGM]
v_cvt_f32_u32 v6, s72                              // s[sgprWorkGroup0] = s75 / s72
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup0] = s75 / s72
v_cvt_f32_u32 v7, s75                              // s[sgprWorkGroup0] = s75 / s72
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup0] = s75 / s72
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup0] = s75 / s72
v_mul_u32_u24 v7, v6, s72                          // s[sgprWorkGroup0] = s75 / s72
v_sub_nc_u32 v7, s75, v7                           // s[sgprWorkGroup0] = s75 / s72
v_cmp_eq_u32 vcc_lo, v7, s72                       // s[sgprWorkGroup0] = s75 / s72
s_mov_b32 exec_lo vcc_lo                           // s[sgprWorkGroup0] = s75 / s72
v_add_nc_u32 v6, 1, v6                             // s[sgprWorkGroup0] = s75 / s72
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s75 % s72
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v7, s72                       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v6, v6, 1                             // quotient - 1
v_mul_u32_u24 v7, v6, s72                          // re-calculate remainder
v_sub_nc_u32 v7, s75, v7                           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup0], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v7          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s72 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s75, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s74, s74, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s74 // wg1 += blockId * WGM
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

/* global read addresses: final offsets a */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  1,  0, 14 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideA0I], 2 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+0], s[sgprScalarGlobalReadOffsetA+0], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+1], s[sgprStrideA0I], 4 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+1], s[sgprScalarGlobalReadOffsetA+1], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+2], s[sgprStrideA0I], 6 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+2], s[sgprScalarGlobalReadOffsetA+2], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+3], s[sgprStrideA0I], 8 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+3], s[sgprScalarGlobalReadOffsetA+3], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+4], s[sgprStrideA0I], 10 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+4], s[sgprScalarGlobalReadOffsetA+4], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+5], s[sgprStrideA0I], 12 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+5], s[sgprScalarGlobalReadOffsetA+5], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+6], s[sgprStrideA0I], 14 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+6], s[sgprScalarGlobalReadOffsetA+6], 0x1 // scalar offset *= bytes/element

/* global read addresses: final offsets b */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 14 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideB1J], 2 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+1], s[sgprStrideB1J], 4 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+1], s[sgprScalarGlobalReadOffsetB+1], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+2], s[sgprStrideB1J], 6 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+2], s[sgprScalarGlobalReadOffsetB+2], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+3], s[sgprStrideB1J], 8 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+3], s[sgprScalarGlobalReadOffsetB+3], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+4], s[sgprStrideB1J], 10 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+4], s[sgprScalarGlobalReadOffsetB+4], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+5], s[sgprStrideB1J], 12 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+5], s[sgprScalarGlobalReadOffsetB+5], 0x1 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+6], s[sgprStrideB1J], 14 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+6], s[sgprScalarGlobalReadOffsetB+6], 0x1 // scalar offset *= bytes/element

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s75, s[sgprWorkGroup0], 16            // WorkGroup[01] * MT
s_mul_i32 s74, s[sgprWorkGroup0], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s75, s74, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s74, s74, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s73, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s72, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 4 // s[LoopCounterL] = s[sgprSizesSum] / 16
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
s_mul_i32 s73, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s72, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s73, s73, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s72, s72, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s72, s72, s73                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s73, s72, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s72, s72, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_add_u32 s74, s74, s72                            // accum GsuOffset term to tilestart
s_addc_u32 s75, s75, s73                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s72, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s73, constStrideAL, s72               // stride x (size-1)
s_mul_i32 s72, constStrideAL, s72                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // sum tensor size
s_sub_u32 s72, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s73, s[sgprStrideA0I], s72            // stride x (size-1)
s_mul_i32 s72, s[sgprStrideA0I], s72               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 2 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s73, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s72, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s74, s74, s72                            // accum wg term to tilestart
s_addc_u32 s75, s75, s73                           // accum wg term to tilestart
s_lshl_b64 s[74:75], s[74:75], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s74    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s75   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s75, s[sgprWorkGroup1], 16            // WorkGroup[01] * MT
s_mul_i32 s74, s[sgprWorkGroup1], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s75, s74, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s74, s74, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s73, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s72, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 4 // s[LoopCounterL] = s[sgprSizesSum] / 16
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
s_mul_i32 s73, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s72, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s73, s73, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s72, s72, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s72, s72, s73                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s73, s72, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s72, s72, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_B_End:
s_add_u32 s74, s74, s72                            // accum GsuOffset term to tilestart
s_addc_u32 s75, s75, s73                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s72, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s73, constStrideBL, s72               // stride x (size-1)
s_mul_i32 s72, constStrideBL, s72                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // sum tensor size
s_sub_u32 s72, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s73, s[sgprStrideB1J], s72            // stride x (size-1)
s_mul_i32 s72, s[sgprStrideB1J], s72               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 2 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s73, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s72, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s74, s74, s72                            // accum wg term to tilestart
s_addc_u32 s75, s75, s73                           // accum wg term to tilestart
s_lshl_b64 s[74:75], s[74:75], 1                   // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s74    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s75   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: increments a */
s_and_b32 s73, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s73, s73, DepthU*BpeAGR                  // GSU*DepthU*Bpe
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeAGR, s73 // incrA (unrollIdx)

/* global read addresses: increments b */
s_and_b32 s73, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s73, s73, DepthU*BpeBGR                  // GSU*DepthU*Bpe
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeBGR, s73 // incrB (unrollIdx)
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...8) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [14...30) from pool */
v_mov_b32 v[vgprValuC+0], 0.0                        // initC
v_mov_b32 v[vgprValuC+1], 0.0                        // initC
v_mov_b32 v[vgprValuC+2], 0.0                        // initC
v_mov_b32 v[vgprValuC+3], 0.0                        // initC
v_mov_b32 v[vgprValuC+4], 0.0                        // initC
v_mov_b32 v[vgprValuC+5], 0.0                        // initC
v_mov_b32 v[vgprValuC+6], 0.0                        // initC
v_mov_b32 v[vgprValuC+7], 0.0                        // initC
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_and_b32 s72, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s72, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v31, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v31, v31                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v32, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v31, v31, v32                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v31, v31                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v32, v31, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v32, s[sgprLoopCounterL], v32         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v32, s[sgprGSUSumIdx+1]       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo vcc_lo                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v31, 1, v31                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v32, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v32, s[sgprGSUSumIdx+1]       // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v31, v31, 1                           // quotient - 1
v_mul_u32_u24 v32, v31, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_nc_u32 v32, s[sgprLoopCounterL], v32         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v31       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v32        // remainder
s_add_u32 s72, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s72                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s74, s[sgprStaggerU], 0x1f00
s_lshr_b32 s74, s74, 0x8
s_and_b32 s75, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s72, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s73, s72, s74                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s73           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s72, s72, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s73, s72, 1                              // staggerU mask
s_cmp_ge_u32 s72, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s73, 0          // set Mask
s_cmp_eq_u32 s75, 0x0
s_cbranch_scc1 label_StaggerUMapping_1
s_mov_b32 s72, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s75, 0x2000
s_cbranch_scc1 label_StaggerUMapping_2
s_mov_b32 s72, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s75, 0x4000
s_cbranch_scc1 label_StaggerUMapping_3
s_mov_b32 s72, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s75, 0x6000
s_cbranch_scc1 label_StaggerUMapping_4
s_mul_i32 s73, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s72, s72, s73
s_add_u32 s72, s72, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s75, 0x8000
s_cbranch_scc1 label_staggerInputEnd
s_mov_b32 s72, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s72 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s74 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s73, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s72, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s73, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s72, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // limit -= inc)
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

//v_cvt_f32_u32 v[vgprValuC+0], s[sgprSrdA]

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

// debug: buffer load
buffer_load_d16_b16 v[vgprG2LA+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], null offen offset:0
buffer_load_d16_b16 v[vgprG2LA+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 
buffer_load_d16_b16 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+4], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+5], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+7], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], null offen offset:0
buffer_load_d16_b16 v[vgprG2LB+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+4], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+5], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+6], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+7], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6] offen offset:0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s72, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s73, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s72, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s73, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_wait_loadcnt 0                                   // 5wait for global read
// Skip barrier: NumThreads=32PGR=0, prior iter done reading lds                                  // 5wait for global read

// debug: ds_store
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:64
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:128
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:192
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+4] offset:256
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+5] offset:320
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+6] offset:384
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+7] offset:448

ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:64
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:128
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:192
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:256
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:320
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:384
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:448

s_wait_dscnt 0

// debug: ds_load
/*
ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:2
ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:8
ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:10
ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:16
ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:18
ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:24
ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:26

ds_load_u16 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:0
ds_load_u16_d16_hi v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:2
ds_load_u16 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:8
ds_load_u16_d16_hi v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:10
ds_load_u16 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:16
ds_load_u16_d16_hi v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:18
ds_load_u16 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:24
ds_load_u16_d16_hi v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:26
*/

//ds_load_b128 v[vgprValuA_X0_I0:vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:0
//ds_load_b128 v[vgprValuB_X0_I0:vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:0


ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:2
ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:4
ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:6
ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:8
ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:10
ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:12
ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:14

ds_load_u16 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:0
ds_load_u16_d16_hi v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:2
ds_load_u16 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:4
ds_load_u16_d16_hi v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:6
ds_load_u16 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:8
ds_load_u16_d16_hi v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:10
ds_load_u16 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:12
ds_load_u16_d16_hi v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:14


s_wait_dscnt 0

// debug: v_wmma
v_wmma_f32_16x16_f16 v[0:7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[0:7] // left value = v[0+0:7+0]

// debug: to print A/B
//v_cvt_f32_f16 v[vgprValuC+0], v[vgprG2LA+0]
//v_cvt_f32_f16 v[vgprValuC+1], v[vgprG2LA+1]
//v_cvt_f32_f16 v[vgprValuC+2], v[vgprG2LA+2]
//v_cvt_f32_f16 v[vgprValuC+3], v[vgprG2LA+3]
//v_cvt_f32_f16 v[vgprValuC+4], v[vgprG2LA+4]
//v_cvt_f32_f16 v[vgprValuC+5], v[vgprG2LA+5]
//v_cvt_f32_f16 v[vgprValuC+6], v[vgprG2LA+6]
//v_cvt_f32_f16 v[vgprValuC+7], v[vgprG2LA+7]

//v_cvt_f32_f16 v[vgprValuC+0], v[vgprG2LB+0]
//v_cvt_f32_f16 v[vgprValuC+1], v[vgprG2LB+1]
//v_cvt_f32_f16 v[vgprValuC+2], v[vgprG2LB+2]
//v_cvt_f32_f16 v[vgprValuC+3], v[vgprG2LB+3]
//v_cvt_f32_f16 v[vgprValuC+4], v[vgprG2LB+4]
//v_cvt_f32_f16 v[vgprValuC+5], v[vgprG2LB+5]
//v_cvt_f32_f16 v[vgprValuC+6], v[vgprG2LB+6]
//v_cvt_f32_f16 v[vgprValuC+7], v[vgprG2LB+7]

//v_cvt_f32_f16 v[vgprValuC+0], v[vgprValuA_X0_I0+0]
//v_cvt_f32_f16 v[vgprValuC+1], v[vgprValuA_X0_I0_D1+0]
//v_cvt_f32_f16 v[vgprValuC+2], v[vgprValuA_X0_I0+1]
//v_cvt_f32_f16 v[vgprValuC+3], v[vgprValuA_X0_I0_D1+1]
//v_cvt_f32_f16 v[vgprValuC+4], v[vgprValuA_X0_I0+2]
//v_cvt_f32_f16 v[vgprValuC+5], v[vgprValuA_X0_I0_D1+2]
//v_cvt_f32_f16 v[vgprValuC+6], v[vgprValuA_X0_I0+3]
//v_cvt_f32_f16 v[vgprValuC+7], v[vgprValuA_X0_I0_D1+3]

//v_cvt_f32_f16 v[vgprValuC+0], v[vgprValuB_X0_I0+0]
//v_cvt_f32_f16 v[vgprValuC+1], v[vgprValuB_X0_I0_D1+0]
//v_cvt_f32_f16 v[vgprValuC+2], v[vgprValuB_X0_I0+1]
//v_cvt_f32_f16 v[vgprValuC+3], v[vgprValuB_X0_I0_D1+1]
//v_cvt_f32_f16 v[vgprValuC+4], v[vgprValuB_X0_I0+2]
//v_cvt_f32_f16 v[vgprValuC+5], v[vgprValuB_X0_I0_D1+2]
//v_cvt_f32_f16 v[vgprValuC+6], v[vgprValuB_X0_I0+3]
//v_cvt_f32_f16 v[vgprValuC+7], v[vgprValuB_X0_I0_D1+3]

//v_cvt_f32_u32 v[vgprValuC+0], v[vgprGlobalReadOffsetA]
//v_cvt_f32_u32 v[vgprValuC+1], v[vgprGlobalReadOffsetB]
//v_cvt_f32_u32 v[vgprValuC+2], v[vgprLocalWriteAddrA]
//v_cvt_f32_u32 v[vgprValuC+3], v[vgprLocalWriteAddrB]
//v_cvt_f32_u32 v[vgprValuC+4], v[vgprLocalReadAddrA]
//v_cvt_f32_u32 v[vgprValuC+5], v[vgprLocalReadAddrB]

//v_cvt_f32_u32 v[vgprValuC+0], s[sgprWorkGroup0]
//v_cvt_f32_u32 v[vgprValuC+0], s[sgprLoopCounterL]
//v_cvt_f32_u32 v[vgprValuC+0], s[sgprSizesSum]
//v_cvt_f32_u32 v[vgprValuC+0], s[sgprStrideD1J]

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/* Tail: add ValuA/B vgpr buffer [14...30) to pool */

/* Tail: add address/G2L vgpr [30...30) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (8,8,0) */
.set vgprG2LA_BASE, 14
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 22
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s73, s[sgprSizesSum], 4                 // s73 = s[sgprSizesSum] / 16
s_and_b32 s74, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v31, s74                             // s72 = s73 / s74
v_rcp_iflag_f32 v31, v31                           // s72 = s73 / s74
v_cvt_f32_u32 v32, s73                             // s72 = s73 / s74
v_mul_f32 v31, v31, v32                            // s72 = s73 / s74
v_cvt_u32_f32 v31, v31                             // s72 = s73 / s74
v_mul_u32_u24 v32, v31, s74                        // s72 = s73 / s74
v_sub_nc_u32 v32, s73, v32                         // s72 = s73 / s74
v_cmp_eq_u32 vcc_lo, v32, s74                      // s72 = s73 / s74
s_mov_b32 exec_lo vcc_lo                           // s72 = s73 / s74
v_add_nc_u32 v31, 1, v31                           // s72 = s73 / s74
v_mov_b32 v32, 0                                   // s[sgprGSUSumIdx+1] = s73 % s74
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v32, s74                      // overflow happened in remainder
s_mov_b32 exec_lo vcc_lo                           // overflow happened in remainder
v_sub_nc_u32 v31, v31, 1                           // quotient - 1
v_mul_u32_u24 v32, v31, s74                        // re-calculate remainder
v_sub_nc_u32 v32, s73, v32                         // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s72, v31                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v32        // remainder
s_sub_u32 s73, s74, 1                              // GSU-1
s_cmp_eq_u32 s72, 0                                // quotient == 0
s_cselect_b32 s72, s[sgprGSUSumIdx+1], s73         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s72                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s72, 2, s[sgprStaggerUIter]
s_cmp_ge_i32 s72, 0
s_cbranch_scc0 label_Negative_T8JHFHKM7BO5OHXW
s_mul_hi_u32 s73, s72, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s72, s72, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_YSQ29IP70005TTFS
label_Negative_T8JHFHKM7BO5OHXW:
s_abs_i32 s72, s72
s_mul_hi_u32 s73, s72, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s72, s72, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s72, s72, 0xffffffff
s_xor_b32 s73, s73, 0xffffffff
s_add_u32 s72, s72, 0x1
s_addc_u32 s73, s73, 0
label_MultiplyDone_YSQ29IP70005TTFS:
s_sub_u32 s72, s72, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s73, s73, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s72, 2, s[sgprStaggerUIter]
s_cmp_ge_i32 s72, 0
s_cbranch_scc0 label_Negative_S4FDBQ587JJL6NOU
s_mul_hi_u32 s73, s72, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s72, s72, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_L43KTIIJOOEN7J6P
label_Negative_S4FDBQ587JJL6NOU:
s_abs_i32 s72, s72
s_mul_hi_u32 s73, s72, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s72, s72, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s72, s72, 0xffffffff
s_xor_b32 s73, s73, 0xffffffff
s_add_u32 s72, s72, 0x1
s_addc_u32 s73, s73, 0
label_MultiplyDone_L43KTIIJOOEN7J6P:
s_sub_u32 s72, s72, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s73, s73, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

buffer_load_d16_b16 v[vgprG2LA+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], null offen offset:0
buffer_load_d16_b16 v[vgprG2LA+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 
buffer_load_d16_b16 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+4], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+3] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+5], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+4] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+5] offen offset:0
buffer_load_d16_b16 v[vgprG2LA+7], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+6] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], null offen offset:0
buffer_load_d16_b16 v[vgprG2LB+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+4], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+5], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+6], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5] offen offset:0
buffer_load_d16_b16 v[vgprG2LB+7], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6] offen offset:0
s_wait_loadcnt 0 

v_and_b32 v31, 31, v[vgprSerial]                   // v31 = v[vgprSerial] % 32
v_and_b32 v31, 15, v31                             // v31 = v31 % 16
v_cmp_ge_i32 s72, v31, s[sgprLoopCounterL]         // check K index >= Size L

v_cndmask_b32 v[vgprG2LA+0], v[vgprG2LA+0], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LA+1], v[vgprG2LA+1], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LA+2], v[vgprG2LA+2], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LA+3], v[vgprG2LA+3], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LA+4], v[vgprG2LA+4], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LA+5], v[vgprG2LA+5], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LA+6], v[vgprG2LA+6], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LA+7], v[vgprG2LA+7], 0, s72 // set 0 if K_idx >= sizeL

v_cndmask_b32 v[vgprG2LB+0], v[vgprG2LB+0], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LB+1], v[vgprG2LB+1], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LB+2], v[vgprG2LB+2], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LB+3], v[vgprG2LB+3], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LB+4], v[vgprG2LB+4], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LB+5], v[vgprG2LB+5], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LB+6], v[vgprG2LB+6], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprG2LB+7], v[vgprG2LB+7], 0, s72 // set 0 if K_idx >= sizeL

/*
v_cvt_f32_f16 v[vgprValuC+0], v[vgprG2LA+0]
v_cvt_f32_f16 v[vgprValuC+1], v[vgprG2LA+1]
v_cvt_f32_f16 v[vgprValuC+2], v[vgprG2LA+2]
v_cvt_f32_f16 v[vgprValuC+3], v[vgprG2LA+3]
v_cvt_f32_f16 v[vgprValuC+4], v[vgprG2LA+4]
v_cvt_f32_f16 v[vgprValuC+5], v[vgprG2LA+5]
v_cvt_f32_f16 v[vgprValuC+6], v[vgprG2LA+6]
v_cvt_f32_f16 v[vgprValuC+7], v[vgprG2LA+7]

v_cvt_f32_f16 v[vgprValuC+0], v[vgprG2LB+0]
v_cvt_f32_f16 v[vgprValuC+1], v[vgprG2LB+1]
v_cvt_f32_f16 v[vgprValuC+2], v[vgprG2LB+2]
v_cvt_f32_f16 v[vgprValuC+3], v[vgprG2LB+3]
v_cvt_f32_f16 v[vgprValuC+4], v[vgprG2LB+4]
v_cvt_f32_f16 v[vgprValuC+5], v[vgprG2LB+5]
v_cvt_f32_f16 v[vgprValuC+6], v[vgprG2LB+6]
v_cvt_f32_f16 v[vgprValuC+7], v[vgprG2LB+7]
*/

ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:64
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:128
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:192
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+4] offset:256
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+5] offset:320
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+6] offset:384
ds_store_b16 v[vgprLocalWriteAddrA], v[vgprG2LA+7] offset:448

ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:64
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:128
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:192
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:256
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:320
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:384
ds_store_b16 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:448
s_wait_dscnt 0


.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF

.set vgprValuA_X0_I0_BASE, 14
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X0_I0_D0_PACK, 18
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuB_X0_I0_BASE, 22
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X0_I0_D0_PACK, 26
.set vgprValuB_X0_I0_D1, vgprValuB_X0_I0_D0_PACK+0

/* tail loop: macs */
label_TailLoopBeginL:

ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0
ds_load_u16_d16_hi v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:2
ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:8
ds_load_u16_d16_hi v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:10
ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:16
ds_load_u16_d16_hi v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:18
ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:24
ds_load_u16_d16_hi v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:26

ds_load_u16 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:0
ds_load_u16_d16_hi v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:2
ds_load_u16 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:8
ds_load_u16_d16_hi v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:10
ds_load_u16 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:16
ds_load_u16_d16_hi v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:18
ds_load_u16 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:24
ds_load_u16_d16_hi v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:26
s_wait_dscnt 0


//v_cvt_f32_f16 v[vgprValuC+1], v[vgprValuA_X0_I0+0]


/*
// local read inc a
s_mov_b32 s72, 0x200                                                       // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc_lo, s72, v[vgprLocalReadAddrA+0] // lrA += 512 ((MT+PAD)*bpeDS)

// local read inc b
                                                                           // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, s72, v[vgprLocalReadAddrB+0] // lrB += 512 ((MT+PAD)*bpeDS)
s_wait_dscnt 0                                                             // 4wait for local read
*/

/*
v_and_b32 v31, 31, v[vgprSerial]                   // v31 = v[vgprSerial] % 32
v_lshrrev_b32 v31, 4, v31                          // v31 = v31 / 16
v_lshlrev_b32 v31, 3, v31                          // v31 = v31 * 8
v_add_nc_u32 v32, v31, 0
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], 0, s72 // set 0 if K_idx >= sizeL
v_add_nc_u32 v32, v32, 4                           // add part of K
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+2], v[vgprValuA_X0_I0+0+0+0+2], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0+3], 0, s72 // set 0 if K_idx >= sizeL

v_and_b32 v31, 31, v[vgprSerial]                   // v31 = v[vgprSerial] % 32
v_lshrrev_b32 v31, 4, v31                          // 31 = 31 / 16
v_lshlrev_b32 v31, 3, v31                          // v31 = v31 * 8
v_add_nc_u32 v32, v31, 0
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0, s72 // set 0 if K_idx >= sizeL
v_add_nc_u32 v32, v32, 4                           // add part of K
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], 0, s72 // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], 0, s72 // set 0 if K_idx >= sizeL

s_and_b32 s74, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s74, 8, s74                              // use shift to fill 0 for outside element
s_lshl_b32 s74, s74, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[34:35], s74, v[vgprValuA_X0_I0+0+0+0+0:vgprValuA_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[36:37], s74, v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1]
v_add_nc_u32 v32, v31, 4                           // add part of K
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v34, s72
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v35, s72
v_add_nc_u32 v32, v32, 4                           // add part of K
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+2], v[vgprValuA_X0_I0+0+0+0+2], v36, s72
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0+3], v37, s72
v_lshlrev_b64 v[34:35], s74, v[vgprValuB_X0_I0+0+0+0+0:vgprValuB_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[36:37], s74, v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1]
v_add_nc_u32 v32, v31, 4                           // add part of K
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v34, s72
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v35, s72
v_add_nc_u32 v32, v32, 4                           // add part of K
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], v36, s72
v_cmp_ge_i32 s72, v32, s[sgprLoopCounterL]         // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], v37, s72
s_nop 1
*/

v_wmma_f32_16x16_f16 v[0:7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], v[0:7] // left value = v[0+0:7+0]


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
.set vgprValuB_X0_I0_D0_PACK, UNDEF
.set vgprValuB_X0_I0_D1, UNDEF

/* Tail: add MISC Vgpr [8...14) to pool */
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
//.set sgprSrdC, 32
.set sgprSrdD, 24

/* Mapping of Acc register -> C Vgpr register */

// debug global write srdD
/* Multiply MI out register with Alpha -> C Vgpr register */
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
/*
//v_lshrrev_b32 v12, 5, v[vgprSerial]                // v12 = Serial / 32
//v_lshrrev_b32 v13, 0, v12                          // v13 = v12 / 1
//v_mul_lo_u32 v9, 0x10, v13                         // wave coordination offset 1
//v_lshrrev_b32 v13, 1, v[vgprSerial]                // v13 = v[vgprSerial] / 2
//v_add_lshl_u32 v9, v13, v9, 0                      // coordination 1 = vwB *(wave_id1 + tid1)
//v_mul_lo_u32 v10, v9, s[sgprStrideC1J]             //  offset 1

v_lshrrev_b32 v9, 1, v[vgprSerial]
v_mul_lo_u32 v11, v9, s[sgprStrideD1J]             //  offset 1

//v_and_b32 v13, 0, v12                              // v13 = v12 % 1
//v_mul_lo_u32 v13, 0x10, v13                        // wave coordination offset 0
//v_and_b32 v8, 31, v[vgprSerial]                    // v8 = v[vgprSerial] % 32
//v_and_b32 v8, 1, v8                                // v8 = v8 % 2

v_and_b32 v8, 1, v[vgprSerial]
v_lshlrev_b32 v8, 1, v8                            // thread0 * continuous_output

//v_add_lshl_u32 v8, v13, v8, 0                      // coordination 0 = vwA *(wave_id0 + tid0)
//s_mul_i32 s8, 16, s[sgprWorkGroup0]                // wgp0 * MT0
//v_add_nc_u32 v8, s8, v8                            // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
//s_mul_i32 s8, 16, s[sgprWorkGroup1]                // wgp1 * MT1
//v_add_nc_u32 v9, s8, v9                            // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
*/

// debug: global write address (new)

v_lshrrev_b32 v9, 1, v[vgprSerial]               // v9 = serial / 2
v_and_b32 v8, 1, v[vgprSerial]                   // v8 = serial % 2
v_lshlrev_b32 v8, 0x1, v8

s_mul_i32 s8, 16, s[sgprWorkGroup0]               // MT0*SG0
v_add_nc_u32 v8, s8, v8                         // coord 0 += MT0*SG0
s_mul_i32 s8, 16, s[sgprWorkGroup1]               // MT1*SG1
v_add_nc_u32 v9, s8, v9                         // coord 1 += MT1*SG1

v_mul_lo_u32 v11, v9, s[sgprStrideD1J]


// debug: print data
/*
v_lshrrev_b32 v8, 5, v[vgprSerial]                // v8 = v[vgprSerial] / 32
v_and_b32 v9, 31, v[vgprSerial]                   // v9 = v[vgprSerial] % 32

v_lshrrev_b32 v10, 4, v9                          // v10 = v9 / 16
v_and_b32 v11, 15, v9                             // v11 = v9 % 16

s_mul_i32 s8, 16, s[sgprWorkGroup0]               // MT0*SG0
v_add_nc_u32 v8, s8, v11                          // coord 0 += MT0*SG0
s_mul_i32 s8, 16, s[sgprWorkGroup1]               // MT1*SG1
v_add_nc_u32 v9, s8, v10                          // coord 1 += MT1*SG1

v_mul_lo_u32 v11, v9, s[sgprStrideD1J]
*/


/* not-LocalSplitU: global write */
s_and_b32 s40, 15, s[sgprSizeI]                    // s40 = s[sgprSizeI] % 16
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rMT0
s_mov_b32 s8, 0
s_cmp_gt_u32 s40, s8                               // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
s_and_b32 s40, 15, s[sgprSizeJ]                    // s40 = s[sgprSizeJ] % 16
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rMT1
s_mov_b32 s8, 0
s_cmp_gt_u32 s40, s8                               // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required

label_GW_B0_E0_1:


/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
v_add_lshl_u32 v15, v11, v8, 0x2                   // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3)] */
v_mov_b32 v[vgprValuC+17], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+18], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+19], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+7]          // Rearrange MI out reg

//debug: buffer_store
/* apply mask, calc new C and issue writes */
// debug: print data
/*
buffer_store_b32 v17, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v18, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:128 // store D
buffer_store_b32 v19, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:256 // store D
buffer_store_b32 v20, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:384 // store D
buffer_store_b32 v21, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:512 // store D
buffer_store_b32 v22, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:640 // store D
buffer_store_b32 v23, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:768 // store D
buffer_store_b32 v24, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:896 // store D
*/

//buffer_store_b32 v17, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
//buffer_store_b32 v18, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:136 // store D
//buffer_store_b32 v19, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:272 // store D
//buffer_store_b32 v20, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:408 // store D
//buffer_store_b32 v21, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:544 // store D
//buffer_store_b32 v22, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:680 // store D
//buffer_store_b32 v23, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:816 // store D
//buffer_store_b32 v24, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:952 // store D

// debug: new buffer store
buffer_store_b32 v17, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v18, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:4 // store D
buffer_store_b32 v19, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:16 // store D
buffer_store_b32 v20, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:20 // store D
buffer_store_b32 v21, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:32 // store D
buffer_store_b32 v22, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:36 // store D
buffer_store_b32 v23, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:48 // store D
buffer_store_b32 v24, v15, s[sgprSrdD:sgprSrdD+3], null offen offset:52 // store D

s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
//s_wait_storecnt 0


label_GW_B0_E1_1:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=18 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,1,0,0:vw1); (0,1,0,1:vw1); (0,1,0,2:vw1); (0,1,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v14, BufferOOB

/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s40, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v23, v11, v8, 0x2                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v23, v14, v23, s42                   // LDD clip if OOB. offset

/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v12, vcc_lo, v8, 1                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v24, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v24, v14, v24, s42                   // LDD clip if OOB. offset

/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v12, vcc_lo, v8, 4                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v25, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v25, v14, v25, s42                   // LDD clip if OOB. offset

/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v12, vcc_lo, v8, 5                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v26, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v14, v26, s42                   // LDD clip if OOB. offset

/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v12, vcc_lo, v8, 8                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v27, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v14, v27, s42                   // LDD clip if OOB. offset

/* (d1,vc1,d0,vc0)=(0,0,1,1) */
v_add_co_u32 v12, vcc_lo, v8, 9                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v28, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v14, v28, s42                   // LDD clip if OOB. offset

/* (d1,vc1,d0,vc0)=(0,0,1,2) */
v_add_co_u32 v12, vcc_lo, v8, 12                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v29, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v29, v14, v29, s42                   // LDD clip if OOB. offset

/* (d1,vc1,d0,vc0)=(0,0,1,3) */
v_add_co_u32 v12, vcc_lo, v8, 13                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s40, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s42, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s42, s40, s42                            // in0 && in1
v_add_lshl_u32 v31, v11, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v14, v31, s42                   // LDD clip if OOB. offset


/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3)] */

v_mov_b32 v[vgprValuC+15], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+16], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+17], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+18], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+19], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+7]          // Rearrange MI out reg

//v_mov_b32 v[vgprValuC+15], 0
//v_mov_b32 v[vgprValuC+16], 0
//v_mov_b32 v[vgprValuC+17], 0
//v_mov_b32 v[vgprValuC+18], 0
//v_mov_b32 v[vgprValuC+19], 0
//v_mov_b32 v[vgprValuC+20], 0
//v_mov_b32 v[vgprValuC+21], 0
//v_mov_b32 v[vgprValuC+22], 0

//v_cvt_f32_u32 v15, v25
//v_cvt_f32_u32 v16, 0x1

/* apply mask, calc new C and issue writes */
buffer_store_b32 v15, v23, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v16, v24, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v17, v25, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v18, v26, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v19, v27, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v20, v28, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v21, v29, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D
buffer_store_b32 v22, v31, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D

s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end

label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel