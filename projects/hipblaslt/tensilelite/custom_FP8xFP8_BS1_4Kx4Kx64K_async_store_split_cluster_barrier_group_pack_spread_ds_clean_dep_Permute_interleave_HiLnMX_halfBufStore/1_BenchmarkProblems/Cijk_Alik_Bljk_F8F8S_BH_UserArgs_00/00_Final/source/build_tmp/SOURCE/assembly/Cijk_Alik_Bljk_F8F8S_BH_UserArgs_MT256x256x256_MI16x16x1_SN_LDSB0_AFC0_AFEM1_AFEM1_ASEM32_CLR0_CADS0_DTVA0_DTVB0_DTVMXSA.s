/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.protected Cijk_Alik_Bljk_F8F8S_BH_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM32_CLR0_CADS0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_EPS0_FDSI1_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT8_8_MO40_M1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR0_PKA0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TDMI3_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1
.globl Cijk_Alik_Bljk_F8F8S_BH_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM32_CLR0_CADS0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_EPS0_FDSI1_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT8_8_MO40_M1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR0_PKA0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TDMI3_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1
.p2align 8
.type Cijk_Alik_Bljk_F8F8S_BH_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM32_CLR0_CADS0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_EPS0_FDSI1_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT8_8_MO40_M1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR0_PKA0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TDMI3_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_F8F8S_BH_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM32_CLR0_CADS0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_EPS0_FDSI1_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT8_8_MO40_M1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR0_PKA0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TDMI3_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 1024 // vgprs
  .amdhsa_next_free_sgpr 106 // sgprs
  .amdhsa_group_segment_fixed_size 322560 // lds bytes
  .amdhsa_wavefront_size32 1 // 32-thread wavefronts
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_user_sgpr_count 28
  .amdhsa_user_sgpr_kernarg_preload_length 26
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_fp16_overflow 1
  .amdhsa_inst_pref_size 148
.end_amdhsa_kernel
.text
/* Num VGPR   =1024 */
/* Num AccVGPR=0 */
/* Num SGPR   =106 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 64 x 8 */
/* SubGroup= 4 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=16, GlobalReadVectorWidthB=16 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
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
  - .name: Cijk_Alik_Bljk_F8F8S_BH_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM32_CLR0_CADS0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_EPS0_FDSI1_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT8_8_MO40_M1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR0_PKA0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TDMI3_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1
    .symbol: 'Cijk_Alik_Bljk_F8F8S_BH_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM32_CLR0_CADS0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_EPS0_FDSI1_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT8_8_MO40_M1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR0_PKA0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TDMI3_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1.kd'
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
        .value_type:      fp8
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      fp8
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      fp8
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      fp8
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
    .group_segment_fixed_size:   322560
    .kernarg_segment_align:      8
    .kernarg_segment_size:       104
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 106
    .sgpr_spill_count:           0
    .vgpr_count:                 1024
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_F8F8S_BH_UserArgs_MT256x256x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM32_CLR0_CADS0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_EPS0_FDSI1_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT8_8_MO40_M1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR1_PLR0_PKA0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TDMI3_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1:

/******************************************/
/* VGPR Assignments for MX                */
/******************************************/

/******************************************/
/* VGPR Macro Assignments for MX          */
/******************************************/
.set vgprValuMXSA_X0_I0, 0
.set vgprValuMXSA_X1_I0, 4
.set vgprValuMXSB_X0_I0, 8
.set vgprValuMXSB_X1_I0, 12
.set vgprAsyncAddr, 16
.set vgprAsyncLds, 24
.set vgprAsyncTmp, 28

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [32-544), serializedStore enabled  */
.set vgprValuC, 32
.set vgprGlobalReadOffsetA, 544
.set vgprLocalReadAddrMXSA, 544
.set vgprLocalReadAddrMXSB, 545
.set vgprLocalReadAddrA, 546
.set vgprLocalReadAddrB, 548
.set vgprBase, 550
.set vgprValuB_X0_I0_BASE, vgprBase+0
.set vgprValuA_X0_I0_BASE, vgprBase+256
.set vgprValuB_Y0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_Y1, vgprValuB_X0_I0_BASE+64
.set vgprValuB_Y2, vgprValuB_X0_I0_BASE+128
.set vgprValuB_Y3, vgprValuB_X0_I0_BASE+192
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X00, vgprValuA_X0_I0+0
.set vgprValuA_X01, vgprValuA_X0_I0+32
.set vgprValuA_X10, vgprValuA_X0_I0+64
.set vgprValuA_X11, vgprValuA_X0_I0+96
.set vgprValuA_X20, vgprValuA_X0_I0+128
.set vgprValuA_X21, vgprValuA_X0_I0+160
.set vgprLocalReadSwapAddrA, 998
.set vgprLocalReadSwapAddrMXSA, 999
.set vgprLocalReadSwapAddrB, 1000
.set vgprLocalReadSwapAddrMXSB, 1001
.set vgprStoreAddr, 1002
.set vgprSRSeed, 1003
.set vgprGL2PrefetchAB, 1004
.set vgprGL2PrefetchMX, 1006
.set vgprGL2PrefetchABNoWG, 1008
.set vgprGL2PrefetchMXNoWG, 1010
.set vgprGL2PrefetchABNext, 1012
.set vgprGL2PrefetchMXNext, 1014
.set vgprSerial, 1023

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprMulticastMask, 5
.set sgprArgType, 6
.set sgprGSULog2BpeC, 7
.set sgprGSUSumIdx, 8
.set sgprALdsSave, 9
.set sgprGSULog2BpeD, 10
.set sgprStaggerU, 11
.set sgprWGM, 12
.set sgprLoopCounterL, 13
.set sgprNumWorkGroups0, 15
.set sgprNumWorkGroups1, 16
.set sgprSizesFree, 20
.set sgprSizesSum, 23
.set sgprAddressD, 24
.set sgprAddressC, 26
.set sgprAddressA, 28
.set sgprAddressMXSA, 30
.set sgprAddressB, 32
.set sgprAddressMXSB, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesMXSA, 42
.set sgprStridesB, 44
.set sgprStridesMXSB, 46
.set sgprAlpha, 48
.set sgprBeta, 49
.set sgprGSU, 50
.set sgprTDMAddrSwapA, 93
.set sgprTDMAddrSwapMXSA, 94
.set sgprTDMSplitA, 95
.set sgprTDMGlobalSplitA, 96
.set sgprIter, 97
.set sgprTDMAddrABNoWG, 98
.set sgprTDMAddrMXABNoWG, 100
.set sgprTDMAddrABNext, 76
.set sgprTDMAddrMXABNext, 78
.set sgprWorkGroup0Next, 80
.set sgprWorkGroup1Next, 81
.set sgprMXSALdsSave, 82
.set sgprWaveId, 83
.set sgprPrefetchIncMX, 102
.set sgprPrefetchIncAB, 104
.set sgprtdmAGroup0, 52
.set sgprtdmAGroup1, 56
.set sgprtdmMXSAGroup0, 64
.set sgprtdmMXSAGroup1, 68
.set sgprtdmBGroup0, sgprtdmAGroup0+0
.set sgprtdmBGroup1, sgprtdmAGroup1+0
.set sgprtdmMXSBGroup0, sgprtdmMXSAGroup0+0
.set sgprtdmMXSBGroup1, sgprtdmMXSAGroup1+0
.set sgprtdmABIncs, 17
.set sgprtdmMXSAMXSBIncs, 18
.set sgprGlobalReadIncsA, 51
.set sgprGlobalReadIncsB, 84
.set sgprGlobalReadIncsMXSA, 85
.set sgprGlobalReadIncsMXSB, 86
.set sgprAsyncScratch, sgprGlobalReadIncsB+0
.set sgprAsyncWG0, sgprGlobalReadIncsMXSB+0

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set constStrideMXSAL, 1
.set sgprStrideMXSA0I, sgprStridesMXSA+0
.set constStrideMXSBL, 1
.set sgprStrideMXSB1J, sgprStridesMXSB+0

.set MT0, 256
.set MT1, 256
.set DepthU, 256
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 16
.set SrdShiftLeftMXSA, 16
.set SrdShiftLeftB, 16
.set SrdShiftLeftMXSB, 16
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000


/******************************************/
/* Bits 127:96 of SRD.                    */
/******************************************/
.set Srd127_96, 0x0

s_nop 0

.macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    s_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\vgprDstIdx+0], \dividend, \magicA
    v_add_nc_u32 v[\vgprDstIdx+0], v[\vgprDstIdx+0], v[\vgprDstIdx+1]
    v_lshrrev_b32 v[\vgprDstIdx+0], \magicShift, v[\vgprDstIdx+0]
.endm


s_mov_b32 m0, 0x4EC00                              // LDS clamp at 322560 bytes (287744 A/B/MX double-buffer + 34816 store-staging 256x128@(256+16))
s_set_vgpr_msb 192                                 // src0: 0, src1: 0, src2: 0, dst: 3
v_mov_b32 v[vgprSerial-768], v0                    // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero
s_mov_b32 s[sgprIter], 3                           // non-persistent: pin to the last persist index so each WG runs exactly one tile
s_nop 0
label_ASM_Start:  /// Main body of the asm kernel
/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req, vgprOffsetL:req, vgprOffset0I:req, vgprTmp:req
    s_nop 0
    s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideA0I], v[\vgprOffset0I] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x10, v[\vgprAddr+0]  // add prepad for pointer shift
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x10, v[\vgprAddr+0]  // add prepad for pointer shift
.endm

.macro GLOBAL_OFFSET_MXSA vgprAddr:req, vgprOffsetL:req, vgprOffset0I:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideMXSA0I], v[\vgprOffset0I] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x10, v[\vgprAddr+0]  // add prepad for pointer shift
.endm

.macro GLOBAL_OFFSET_MXSB vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideMXSB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x10, v[\vgprAddr+0]  // add prepad for pointer shift
.endm


/******************************************/
/* Allocate Resources                     */
/******************************************/

label_Preload_Offset_Start:
s_and_b32 s51, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s52, s2, 0x1e                           // Get arg type
s_mov_b32 s53, s3                                  // Preload internal args
s_cmp_eq_u32 s52, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs

/* Load Kernel Args */
s_mov_b32 s[sgprBeta], s27                         // move preload data to correct sgpr
s_mov_b32 s[sgprAlpha], s26                        // move preload data to correct sgpr
s_mov_b64 s[sgprStridesB:sgprStridesB+1], s[24:25] // move preload data to correct sgpr
s_mov_b64 s[sgprStridesA:sgprStridesA+1], s[22:23] // move preload data to correct sgpr
s_mov_b64 s[sgprStridesC:sgprStridesC+1], s[20:21] // move preload data to correct sgpr
s_mov_b64 s[sgprStridesD:sgprStridesD+1], s[18:19] // move preload data to correct sgpr
s_mov_b64 s[sgprAddressB:sgprAddressB+1], s[16:17] // move preload data to correct sgpr
s_mov_b64 s[sgprAddressA:sgprAddressA+1], s[14:15] // move preload data to correct sgpr
s_mov_b64 s[sgprAddressC:sgprAddressC+1], s[12:13] // move preload data to correct sgpr
s_mov_b64 s[sgprAddressD:sgprAddressD+1], s[10:11] // move preload data to correct sgpr
s_mov_b64 s[22:23], s[8:9]                         // move preload data to correct sgpr
s_mov_b64 s[20:21], s[6:7]                         // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
s_mov_b32 s54, s5                                  // Load num of WGs
s_and_b32 s[sgprStaggerU], s53, 0xffff0000         // Restore StaggerU related vars
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s53, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s52
s_mov_b32 s[sgprWorkGroup0], 0
s_mov_b32 s[sgprWorkGroup1], 0
s_mov_b32 s[sgprWorkGroup2], 0
s_mov_b32 s17, s51
s_mov_b32 s18, s52
s_mov_b32 s51, s54


/******************************************/
/* Local Read Addresses                   */
/******************************************/

s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s92, v[vgprSerial-768]         // first tId
s_lshr_b32 s[sgprWaveId], s92, 5                   // wId=fTid // wavelen

/* local read addresses: tile assignments a/b */
/* lr0I */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v1, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 8, v0                            // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v0, v1, 4, v0                       // 5. K offset: lrKOffset = kIdx * mStride(16); 6. offset in wave: lrOffset = bnOffset + lrKOffset
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v4, 5, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 12, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096); 7. final local read offset: flrOffset = lrOffset + WOffset
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v2, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 2, v1                            // 1. N offset: nOffset = nIdx * nStride
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v3, 5, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 6, v1                       // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride; 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v3, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v2, 15, v3                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v2, 8, v2                            // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshrrev_b32 v3, 4, v3                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v2, v3, 4, v2                       // 5. K offset: lrKOffset = kIdx * mStride(16); 6. offset in wave: lrOffset = bnOffset + lrKOffset
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v6, 6, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v6, 1, v6                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v2, v6, 12, v2                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096); 7. final local read offset: flrOffset = lrOffset + WOffset
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v4, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v3, 15, v4                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v3, 2, v3                            // 1. N offset: nOffset = nIdx * nStride(8)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v5, 6, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v5, 1, v5                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v3, v5, 6, v3                       // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(128); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v4, 5, v[vgprSerial-768]             // 4 = Serial / 32
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v4, 2, v4                            // LSU offset: Get LSU wave_id
s_mov_b32 s19, 256                                 // LSU offset: stride = lsuStride(256) when umlds==True
v_mul_lo_u32 v4, s19, v4                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
s_set_vgpr_msb 128                                 // src0: 0, src1: 0, src2: 0, dst: 2
v_add_nc_u32 v[vgprLocalReadAddrA-512], v4, v0     // Final Offset: offset = (lro0+lsuoffset)*bpeDS
s_set_vgpr_msb 8                                   // src0: 0, src1: 2, src2: 0, dst: 0
v_lshrrev_b32 v5, 8, v[vgprLocalReadAddrA-512]     // Final Offset: padding 16 per block 256
s_set_vgpr_msb 160                                 // src0: 0, src1: 0, src2: 2, dst: 2
v_lshl_add_u32 v[vgprLocalReadAddrA-512], v5, 4, v[vgprLocalReadAddrA-512] // Final Offset: padding 16 per block 256

s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v0, 5, v[vgprSerial-768]             // 0 = Serial / 32
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s19, 8                                   // LSU offset: stride = lsuStride(8) when umlds==True
v_mul_lo_u32 v0, s19, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
s_set_vgpr_msb 128                                 // src0: 0, src1: 0, src2: 0, dst: 2

s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v0, 5, v[vgprSerial-768]             // 0 = Serial / 32
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
v_mul_lo_u32 v0, s19, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
s_set_vgpr_msb 128                                 // src0: 0, src1: 0, src2: 0, dst: 2

/* local read addresses: final offsets b */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v0, 5, v[vgprSerial-768]             // 0 = Serial / 32
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s19, 256                                 // LSU offset: stride = lsuStride(256) when umlds==True
v_mul_lo_u32 v0, s19, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
s_set_vgpr_msb 128                                 // src0: 0, src1: 0, src2: 0, dst: 2
v_add_nc_u32 v[vgprLocalReadAddrB-512], v0, v2     // Final Offset: offset = (lro1+lsuoffset)*bpeDS
s_set_vgpr_msb 8                                   // src0: 0, src1: 2, src2: 0, dst: 0
v_lshrrev_b32 v1, 8, v[vgprLocalReadAddrB-512]     // Final Offset: padding 16 per block 256
s_set_vgpr_msb 160                                 // src0: 0, src1: 0, src2: 2, dst: 2
v_lshl_add_u32 v[vgprLocalReadAddrB-512], v1, 4, v[vgprLocalReadAddrB-512] // Final Offset: padding 16 per block 256

/* local read addresses: declare addresses a */
s_set_vgpr_msb 136                                 // src0: 0, src1: 2, src2: 0, dst: 2
v_add_nc_u32 v[vgprLocalReadAddrA+1-512], 65536, v[vgprLocalReadAddrA+0-512] // Final vgprLocalReadAddrA+1 Offset Plus 64K

s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v0, 16, v[vgprSerial-768]                // v0 = Serial & 16 (=0 lanes 0..15, =16 lanes 16..31, w32)
s_set_vgpr_msb 160                                 // src0: 0, src1: 0, src2: 2, dst: 2

s_set_vgpr_msb 136                                 // src0: 0, src1: 2, src2: 0, dst: 2
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v0, 16, v[vgprSerial-768]                // v0 = Serial & 16 (=0 lanes 0..15, =16 lanes 16..31, w32)
s_set_vgpr_msb 160                                 // src0: 0, src1: 0, src2: 2, dst: 2

/* local read addresses: declare addresses b */
s_set_vgpr_msb 136                                 // src0: 0, src1: 2, src2: 0, dst: 2
v_add_co_u32 v[vgprLocalReadAddrB+0-512], vcc_lo, 0x12200, v[vgprLocalReadAddrB+0-512] //  += LdsOffsetB (lower)
s_delay_alu instid0(VALU_DEP_1)
v_add_nc_u32 v[vgprLocalReadAddrB+1-512], 65536, v[vgprLocalReadAddrB+0-512] // Final vgprLocalReadAddrB+1 Offset Plus 64K
s_set_vgpr_msb 200                                 // src0: 0, src1: 2, src2: 0, dst: 3
v_add_nc_u32 v[vgprLocalReadSwapAddrA-768], 143872, v[vgprLocalReadAddrA-512] // Calculate starting lds addr of second buffer
s_set_vgpr_msb 203                                 // src0: 3, src1: 2, src2: 0, dst: 3
v_xor_b32 v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // xor both lds buffer offsets to enable swapping
v_add_nc_u32 v[vgprLocalReadSwapAddrB-768], 143872, v[vgprLocalReadAddrB-512] // Calculate starting lds addr of second buffer
v_xor_b32 v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // xor both lds buffer offsets to enable swapping


/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

/* local write addresses: first offset b */
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v2, MT0                                  // set MT0 into sgpr
v_mov_b32 v1, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_4)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
v_mov_b32 v2, MT1                                  // set MT1 into sgpr
v_mov_b32 v1, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v0      // set back to numWorkGroup0
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_4)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
v_readfirstlane_b32 s[sgprNumWorkGroups1], v0      // set back to numWorkGroup1

/* Short circuit condition if Alpha == 0, then sumDims=0 */
s_setreg_IMM32_b32 hwreg(26,2,1), 1                // Disable WMMA arb stall


/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_mov_b32 s88, ttmp6                               // Read TTMP6 register
s_delay_alu instid0(SALU_CYCLE_1)
s_and_b32 s88, s88, 0xfffffff                      // Filter unused bits.
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_eq_u32 s88, 0                                // ttmp6 == 0x0 ?
s_cbranch_scc0 label_EnableCluster
s_mov_b32 s[sgprWorkGroup0], ttmp9                 // workaround
s_and_b32 s[sgprWorkGroup1], 0xffff, ttmp7         // workaround
s_lshr_b32 s[sgprWorkGroup2], ttmp7, 0x10
s_branch label_RemapWorkGroupDone
label_EnableCluster:
s_mov_b32 s89, ttmp7                               // Read TTMP7 register,                                                                        cluster_z | cluster_y.
s_bfe_u32 s90, s88, 262148                         // Etract wg_y.
s_bfe_u32 s91, s88, 262160                         // Etract nwg_y. Value is nwg_y - 1
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s91, s91, 1
s_and_b32 s92, s89, 0xffff                         // Etract cluster_y.
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s[sgprWorkGroup1], s92, s91              // cluster_y * nwg_y
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s90 // WorkGroup1 = (cluster_y * nwg_y) + wg_y
s_and_b32 s89, s88, 0xf                            // Etract wg_x.
s_bfe_u32 s91, s88, 262156                         // Etract nwg_x. Value is nwg_x - 1
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s91, s91, 1
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s[sgprWorkGroup0], ttmp9, s91            // cluster_x * nwg_x
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s89 // WorkGroup0 = (cluster_x * nwg_x) + wg_x
s_bfe_u32 s92, s88, 262152                         // Etract wg_z.
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s[sgprWorkGroup2], 0, s92                // WorkGroup2 = (cluster_z(0) * nwg_z(1)) + wg_z
label_RemapWorkGroupDone:
s_nop 0
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 5                             // wId=fTid // wavelen
s_delay_alu instid0(SALU_CYCLE_1)
s_bitcmp1_b32 s87, 0                               // Check parity of wId
s_cbranch_scc1 label_setMulticastMask_OddWave      // Jump if wId is odd
label_setMulticastMask_EvenWave:
s_lshl_b32 s[sgprMulticastMask], 0x1111, s89       // Setting maskA for even wave
s_branch label_setMulticastMaskEnd
label_setMulticastMask_OddWave:
s_mul_i32 s90, s90, s91                            // Shift factor: wg_y * nwg_x
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s[sgprMulticastMask], 0xf, s90          // Setting maskB for odd wave
label_setMulticastMaskEnd:

// /*** Tony Modify 2 ****/
// non-persistent: the launch grid is the full tile grid, so WorkGroup0/1 already
// address this WG's only tile and there is no next tile to walk to.
s_mov_b32 s[sgprWorkGroup0Next], s[sgprWorkGroup0]
s_mov_b32 s[sgprWorkGroup1Next], s[sgprWorkGroup1]
s_nop 7

label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 0
s_mov_b32 s[sgprGSULog2BpeD], 0
label_GSU_End:
label_TDMGlobalOffsetA:
s_wait_kmcnt 0
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 5                             // wId=fTid // wavelen
s_delay_alu instid0(SALU_CYCLE_1)
s_bitcmp1_b32 s87, 0                               // Check parity of wId
s_cbranch_scc1 label_TDMGlobalOffsetB              // Jump to B if wId is odd
// TDM wave separated calc start addr of A
s_mov_b64 s[88:89], 0
s_mul_i32 s88, s[sgprStrideA0I], 256               // stride * MT(256) * bpe(1.0)
s_mul_i32 s88, s88, s[sgprWorkGroup0]              // *= wgId)
v_readfirstlane_b32 s90, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s90, s90, 6                             // wCompId = fTid // wavelen(32) // numComp(2)
s_delay_alu instid0(SALU_CYCLE_1)
// gl2 prefetch addr calc start
s_mul_i32 s92, s90, 128                            // 2 waves divide 256 rows into 2 groups of 128 rows
s_mul_i32 s92, s92, s[sgprStrideA0I]               // woffset *= stride
s_and_b32 s93, s[sgprWorkGroup1], 3                // Update WG Idx to 4 clusters
s_mul_i32 s93, s93, s[sgprStrideA0I]               // WG offset *= stride
s_add_u32 s92, s93, s92                            // Add WG & wave offsets
s_add_u32 s92, s[sgprAddressA+0], s92              // Add tile start address
s_addc_u32 s93, s[sgprAddressA+1], 0               // Add tile start address(H)
s_set_vgpr_msb 195                                 // src0: 3, src1: 0, src2: 0, dst: 3
v_and_b32 v[vgprGL2PrefetchABNoWG-768], v[vgprSerial-768], 31 // Restrict thread index to 32
v_mul_lo_u32 v[vgprGL2PrefetchABNoWG-768], v[vgprGL2PrefetchABNoWG-768], 4 // Multiply by 4
v_mul_lo_u32 v[vgprGL2PrefetchABNoWG-768], v[vgprGL2PrefetchABNoWG-768], s[sgprStrideA0I] // Jump to correct row
v_mov_b32 v[vgprGL2PrefetchABNoWG+1-768], 0        // Set upper 32-bit to 0
s_set_vgpr_msb 49356
v_add_nc_u64 v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1], s[92:93], v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1]
s_add_u32 s92, s88, 512                            // Offset by 2 buffers (DU*Bpe*2(preloads))
s_mov_b32 s93, 0
v_add_nc_u64 v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1], s[92:93], v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1]
s_mul_i32 s92, s[sgprStrideA0I], 256               // stride * MT(256) * bpe(1.0)
s_mul_i32 s92, s92, s[sgprWorkGroup0Next]          // *= wgId)
v_add_nc_u64 v[vgprGL2PrefetchABNext+0-768:vgprGL2PrefetchABNext+0-768+1], s[92:93], v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1]
s_mov_b32 s[sgprPrefetchIncAB], 256                // Data pointer increment
s_mov_b32 s[sgprPrefetchIncAB+1], 0
// gl2 prefetch addr calc end
s_mul_i32 s90, s90, 128                            // woffset = wCompId * mt // numComp(2) * bpe(1.0)
s_mul_i32 s90, s90, s[sgprStrideA0I]               // woffset *= stride
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s90, s90, 1                             // / (2tdm)
s_delay_alu instid0(SALU_CYCLE_1)
// persist: offset w/o WG
s_add_u32 s[sgprTDMAddrABNoWG], s90, s[sgprAddressA] // += baseAddr(lo)
s_addc_u32 s[sgprTDMAddrABNoWG+1], 0, s[sgprAddressA+1] // += baseAddr(hi)
// next WG addr
s_mul_i32 s91, s[sgprWorkGroup0Next], 256
s_mul_i32 s91, s91, s[sgprStrideA0I]
s_add_u32 s[sgprTDMAddrABNext], s[sgprTDMAddrABNoWG], s91
s_addc_u32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNoWG+1], 0
s_or_b32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNext+1], 0x80000000 // set type field to 2(image)

s_add_u32 s88, s88, s90                            // += woffset
s_add_u32 s[sgprAddressA], s88, s[sgprAddressA]    // += baseAddr(lo)
s_addc_u32 s[sgprAddressA+1], s89, s[sgprAddressA+1] // += baseAddr(hi)
s_branch label_TDMGlobalOffsetABEnd
label_TDMGlobalOffsetB:
// TDM wave separated calc start addr of B
s_mov_b64 s[88:89], 0
s_mul_i32 s88, s[sgprStrideB1J], 256               // stride * MT(256) * bpe(1.0)
s_mul_i32 s88, s88, s[sgprWorkGroup1]              // *= wgId)
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s90, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s90, s90, 6                             // wCompId = fTid // wavelen(32) // numComp(2)
s_delay_alu instid0(SALU_CYCLE_1)
// gl2 prefetch addr calc start
s_mul_i32 s92, s90, 128                            // 2 waves divide 256 rows into 2 groups of 128 rows
s_mul_i32 s92, s92, s[sgprStrideB1J]               // woffset *= stride
s_and_b32 s93, s[sgprWorkGroup0], 3                // Update WG Idx to 4 clusters
s_mul_i32 s93, s93, s[sgprStrideB1J]               // WG offset *= stride
s_add_u32 s92, s93, s92                            // Add WG & wave offsets
s_add_u32 s92, s[sgprAddressB+0], s92              // Add tile start address
s_addc_u32 s93, s[sgprAddressB+1], 0               // Add tile start address(H)
s_set_vgpr_msb 195                                 // src0: 3, src1: 0, src2: 0, dst: 3
v_and_b32 v[vgprGL2PrefetchABNoWG-768], v[vgprSerial-768], 31 // Restrict thread index to 32
v_mul_lo_u32 v[vgprGL2PrefetchABNoWG-768], v[vgprGL2PrefetchABNoWG-768], 4 // Multiply by 4
v_mul_lo_u32 v[vgprGL2PrefetchABNoWG-768], v[vgprGL2PrefetchABNoWG-768], s[sgprStrideB1J] // Jump to correct row
v_mov_b32 v[vgprGL2PrefetchABNoWG+1-768], 0        // Set upper 32-bit to 0
s_set_vgpr_msb 49356
v_add_nc_u64 v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1], s[92:93], v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1]
s_add_u32 s92, s88, 512                            // Offset by 2 buffers (DU*Bpe*2(preloads))
s_mov_b32 s93, 0
v_add_nc_u64 v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1], s[92:93], v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1]
s_mul_i32 s92, s[sgprStrideB1J], 256               // stride * MT(256) * bpe(1.0)
s_mul_i32 s92, s92, s[sgprWorkGroup1Next]          // *= wgId)
v_add_nc_u64 v[vgprGL2PrefetchABNext+0-768:vgprGL2PrefetchABNext+0-768+1], s[92:93], v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1]
s_mov_b32 s[sgprPrefetchIncAB], 256                // Data pointer increment
s_mov_b32 s[sgprPrefetchIncAB+1], 0
// gl2 prefetch addr calc end
s_mul_i32 s90, s90, 128                            // woffset = wCompId * mt // numComp(2) * bpe(1.0)
s_mul_i32 s90, s90, s[sgprStrideB1J]               // woffset *= stride
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s90, s90, 1                             // / (2tdm)
s_delay_alu instid0(SALU_CYCLE_1)
// persist: offset w/o WG
s_add_u32 s[sgprTDMAddrABNoWG], s90, s[sgprAddressB] // += baseAddr(lo)
s_addc_u32 s[sgprTDMAddrABNoWG+1], 0, s[sgprAddressB+1] // += baseAddr(hi)
// next WG addr
s_mul_i32 s91, s[sgprWorkGroup1Next], 256
s_mul_i32 s91, s91, s[sgprStrideB1J]
s_add_u32 s[sgprTDMAddrABNext], s[sgprTDMAddrABNoWG], s91
s_addc_u32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNoWG+1], 0
s_or_b32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNext+1], 0x80000000 // set type field to 2(image)

s_add_u32 s88, s88, s90                            // += woffset
s_add_u32 s[sgprAddressB], s88, s[sgprAddressB]    // += baseAddr(lo)
s_addc_u32 s[sgprAddressB+1], s89, s[sgprAddressB+1] // += baseAddr(hi)
label_TDMGlobalOffsetABEnd:
label_TDMGlobalOffsetMXSA:
s_nop 0
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 5                             // wId=fTid // wavelen
s_delay_alu instid0(SALU_CYCLE_1)
s_bitcmp1_b32 s87, 0                               // Check parity of wId
s_cbranch_scc1 label_TDMGlobalOffsetMXSB           // Jump to B if wId is odd
// TDM wave separated calc start addr of MXSA
s_mov_b64 s[88:89], 0
s_mul_i32 s88, s[sgprWorkGroup0], 1024             // stride * MT(256) * bpe(1)
v_readfirstlane_b32 s90, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s90, s90, 6                             // wCompId = fTid // wavelen(32) // numComp(2)
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s90, s90, s[sgprSizeI]                   // woffset = wCompId * SizeI * K0(4) // numComp(2) * bpe(1)
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s90, s90, 4                              // woffset = wCompId * SizeI * K0(4) // numComp(2) * bpe(1)
s_delay_alu instid0(SALU_CYCLE_1)
// persist: offset w/o WG
// next WG addr
s_mul_i32 s91, s[sgprWorkGroup0Next], 1024         // * MT(256) * bpe(1.0)
// GL2 prefetch addr calc start
s_and_b32 s92, s[sgprWorkGroup1], 3                // Update WG Idx to 4 clusters
s_mul_i32 s92, s92, 256                            // WG offset *= 256(rows)
s_add_u32 s92, s90, s92
s_mov_b32 s93, 0
s_set_vgpr_msb 960
s_set_vgpr_msb 49356
s_mul_i32 s92, s[sgprWorkGroup0Next], 1024         // stride * MT(256) * bpe(1)
// gl2 prefetch addr calc end
s_add_u32 s88, s88, s90                            // += woffset
s_branch label_TDMGlobalOffsetMXSAMXSBEnd
label_TDMGlobalOffsetMXSB:
// TDM wave separated calc start addr of MXSB 
s_mov_b64 s[88:89], 0
s_mul_i32 s88, s[sgprWorkGroup1], 1024             // stride * MT(256) * bpe(1)
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s90, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s90, s90, 6                             // wCompId = fTid // wavelen(32) // numComp(2)
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s90, s90, s[sgprSizeJ]                   // woffset = wCompId * SizeI * K0(4) // numComp(2) * bpe(1)
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s90, s90, 4                              // woffset = wCompId * SizeI * K0(4) // numComp(2) * bpe(1)
s_delay_alu instid0(SALU_CYCLE_1)
// persist: offset w/o WG
// next WG addr
s_mul_i32 s91, s[sgprWorkGroup1Next], 1024         // * MT(256) * bpe(1.0)
// gl2 prefetch addr calc start
s_and_b32 s92, s[sgprWorkGroup0], 3                // Update WG Idx to 4 clusters
s_mul_i32 s92, s92, 256                            // WG offset *= 256(rows)
s_add_u32 s92, s90, s92
s_mov_b32 s93, 0
s_set_vgpr_msb 960
s_set_vgpr_msb 49356
s_mul_i32 s92, s[sgprWorkGroup1Next], 1024         // stride * MT(256) * bpe(1)
// gl2 prefetch addr calc end
s_add_u32 s88, s88, s90                            // += woffset
label_TDMGlobalOffsetMXSAMXSBEnd:
label_TDMInitA:
s_nop 0
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 5                             // wId=fTid // wavelen
s_delay_alu instid0(SALU_CYCLE_1)
s_bitcmp1_b32 s87, 0                               // Check parity of wId
s_cbranch_scc1 label_TDMInitB                      // Jump to B if wId is odd
s_mov_b32 s[sgprtdmAGroup0+0], 1
s_mov_b32 s[sgprtdmAGroup0+1], 0
s_mov_b32 s[sgprtdmAGroup0+2], 0
s_mov_b32 s[sgprtdmAGroup0+3], 0
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0x80000000 // set type field to 2(image)
s_mov_b32 s[sgprtdmAGroup1+0], 0
s_mov_b32 s[sgprtdmAGroup1+1], 0
s_mov_b32 s[sgprtdmAGroup1+2], 0
s_mov_b32 s[sgprtdmAGroup1+3], 0
s_mov_b32 s[sgprtdmAGroup1+4], 0
s_mov_b32 s[sgprtdmAGroup1+5], 0
s_mov_b32 s[sgprtdmAGroup1+6], 0
s_mov_b32 s[sgprtdmAGroup1+7], 0
s_and_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0xfffcffff // Reset data_size
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0x0 // Set data_size to 0
// TDM set global addr
s_mov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprAddressA:sgprAddressA+1]
s_or_b32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0x80000000 // set type field to 2(image)
s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMask]
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 6                             // wId=fTid // wavelen // numComp
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s87, s87, 34816                          // woffset = wId * (mt // numComp * du * bpe + mt // numComp * du * bpe // ldsBlockSizePerPad * ldsPadSize)
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s87, 1                             // / (2tdm)
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s87, s87, 0                              // ldsOffset = woffset + ldsConstOffset
// TDM set LDS addr
s_mov_b32 s[sgprtdmAGroup0+1], s87
s_and_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0xfff7ffff
// TDM set padding
s_or_b32 s[sgprtdmAGroup1+0], s[sgprtdmAGroup1+0], 0x7500000 // set padding 16 per block 256
// TDM set tensor dim 0
s_and_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], 0xffff
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff0000
s_lshl_b32 s87, s[sgprSizeL], 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], s87
s_lshr_b32 s87, s[sgprSizeL], 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s87
// TDM set tensor dim 1
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff
s_and_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0xffff0000
s_lshl_b32 s87, s[sgprSizeI], 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s87
s_lshr_b32 s87, s[sgprSizeI], 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], s87
// TDM set tensor tile 0
s_and_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0xffff
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0x1000000 // set tile0 to 256
// TDM set tensor tile 1
s_and_b32 s[sgprtdmAGroup1+4], s[sgprtdmAGroup1+4], 0xffff0000
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmAGroup1+4], s[sgprtdmAGroup1+4], 0x40 // set tile1 to 64
s_mov_b32 s[sgprtdmAGroup1+5], s[sgprStrideA0I]
s_branch label_TDMInitABEnd
label_TDMInitB:
s_mov_b32 s[sgprtdmBGroup0+0], 1
s_mov_b32 s[sgprtdmBGroup0+1], 0
s_mov_b32 s[sgprtdmBGroup0+2], 0
s_mov_b32 s[sgprtdmBGroup0+3], 0
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0x80000000 // set type field to 2(image)
s_mov_b32 s[sgprtdmBGroup1+0], 0
s_mov_b32 s[sgprtdmBGroup1+1], 0
s_mov_b32 s[sgprtdmBGroup1+2], 0
s_mov_b32 s[sgprtdmBGroup1+3], 0
s_mov_b32 s[sgprtdmBGroup1+4], 0
s_mov_b32 s[sgprtdmBGroup1+5], 0
s_mov_b32 s[sgprtdmBGroup1+6], 0
s_mov_b32 s[sgprtdmBGroup1+7], 0
s_and_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0xfffcffff // Reset data_size
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0x0 // Set data_size to 0
// TDM set global addr
s_mov_b64 s[sgprtdmBGroup0+2:sgprtdmBGroup0+2+1], s[sgprAddressB:sgprAddressB+1]
s_or_b32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0x80000000 // set type field to 2(image)
s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMask]
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 6                             // wId=fTid // wavelen // numComp
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s87, s87, 34816                          // woffset = wId * (mt // numComp * du * bpe + mt // numComp * du * bpe // ldsBlockSizePerPad * ldsPadSize)
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s87, 1                             // / (2tdm)
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s87, s87, 74240                          // ldsOffset = woffset + ldsConstOffset
// TDM set LDS addr
s_mov_b32 s[sgprtdmBGroup0+1], s87
s_and_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0xfff7ffff
// TDM set padding
s_or_b32 s[sgprtdmBGroup1+0], s[sgprtdmBGroup1+0], 0x7500000 // set padding 16 per block 256
// TDM set tensor dim 0
s_and_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], 0xffff
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff0000
s_lshl_b32 s87, s[sgprSizeL], 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], s87
s_lshr_b32 s87, s[sgprSizeL], 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s87
// TDM set tensor dim 1
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff0000
s_lshl_b32 s87, s[sgprSizeJ], 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s87
s_lshr_b32 s87, s[sgprSizeJ], 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], s87
// TDM set tensor tile 0
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0x1000000 // set tile0 to 256
// TDM set tensor tile 1
s_and_b32 s[sgprtdmBGroup1+4], s[sgprtdmBGroup1+4], 0xffff0000
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprtdmBGroup1+4], s[sgprtdmBGroup1+4], 0x40 // set tile1 to 64
s_mov_b32 s[sgprtdmBGroup1+5], s[sgprStrideB1J]
label_TDMInitABEnd:
label_TDMInitMXSA:
s_nop 0
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 5                             // wId=fTid // wavelen
s_delay_alu instid0(SALU_CYCLE_1)
s_bitcmp1_b32 s87, 0                               // Check parity of wId
s_cbranch_scc1 label_TDMInitMXSB                   // Jump to B if wId is odd
s_delay_alu instid0(SALU_CYCLE_1)
s_delay_alu instid0(SALU_CYCLE_1)
// TDM set global addr
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 6                             // wId=fTid // wavelen // numComp
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s87, s87, 1024                           // woffset = wId * (mt // numComp * du * bpe + mt // numComp * du * bpe // ldsBlockSizePerPad * ldsPadSize)
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s87, s87, 69632                          // ldsOffset = woffset + ldsConstOffset
// TDM set LDS addr
// TDM set padding
// TDM set tensor dim 0
s_lshl_b32 s87, s[sgprSizeI], 0x2
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s87, s[sgprSizeI], 0x2
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
// TDM set tensor dim 1
s_lshr_b32 s87, s[sgprSizeL], 0x7                  // SizeL // 32 // 4
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s[sgprSizeL], 0x7                  // SizeL // 32 // 4
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
// TDM set tensor tile 0
s_delay_alu instid0(SALU_CYCLE_1)
// TDM set tensor tile 1
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s87, s[sgprSizeI], 4
s_delay_alu instid0(SALU_CYCLE_1)
s_branch label_TDMInitMXSAMXSBEnd
label_TDMInitMXSB:
s_delay_alu instid0(SALU_CYCLE_1)
s_delay_alu instid0(SALU_CYCLE_1)
// TDM set global addr
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s87, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s87, s87, 6                             // wId=fTid // wavelen // numComp
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s87, s87, 1024                           // woffset = wId * (mt // numComp * du * bpe + mt // numComp * du * bpe // ldsBlockSizePerPad * ldsPadSize)
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s87, s87, 71936                          // ldsOffset = woffset + ldsConstOffset
// TDM set LDS addr
// TDM set padding
// TDM set tensor dim 0
s_lshl_b32 s87, s[sgprSizeJ], 0x2
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s87, s[sgprSizeJ], 0x2
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
// TDM set tensor dim 1
s_lshr_b32 s87, s[sgprSizeL], 0x7                  // SizeL // 32 // 4
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s[sgprSizeL], 0x7                  // SizeL // 32 // 4
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
// TDM set tensor tile 0
s_delay_alu instid0(SALU_CYCLE_1)
// TDM set tensor tile 1
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s87, s[sgprSizeJ], 4
s_delay_alu instid0(SALU_CYCLE_1)
label_TDMInitMXSAMXSBEnd:
.set sgprMulticastMask, UNDEF

/* global read addresses: increments a */
s_and_b32 s89, s[sgprGSU], 0x3fff                  // Restore GSU
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s89, s89, 256                            // GSU*DepthU*Bpe
s_and_b32 s88, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], 256, s89   // incrA (unrollIdx)

s_mul_i32 s89, s[sgprSizeI], 4
s_delay_alu instid0(SALU_CYCLE_1)

s_mul_i32 s89, 4, s[sgprSizeJ]
s_delay_alu instid0(SALU_CYCLE_1)

/* global read addresses: increments b */
s_and_b32 s89, s[sgprGSU], 0x3fff                  // Restore GSU
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s89, s89, 256                            // GSU*DepthU*Bpe
s_and_b32 s88, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], 256, s89   // incrB (unrollIdx)
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s[sgprtdmABIncs], v[vgprSerial-768] // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s[sgprtdmABIncs], s[sgprtdmABIncs], 5   // wId=fTid // wavelen
s_delay_alu instid0(SALU_CYCLE_1)
s_bitcmp1_b32 s[sgprtdmABIncs], 0                  // Check parity of wId
s_cselect_b32 s[sgprtdmABIncs], s[sgprGlobalReadIncsB], s[sgprGlobalReadIncsA]
s_delay_alu instid0(NO_DEP)
s_delay_alu instid0(SALU_CYCLE_1)
// calculate loop iters
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 8 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 256

/* prefetch: global -> local */
// set TDM addr swap sgpr for A/B
s_add_i32 s[sgprTDMAddrSwapA], s[sgprtdmAGroup0+1], 143872 // high buffer
s_delay_alu instid0(SALU_CYCLE_1)
s_xor_b32 s[sgprTDMAddrSwapA], s[sgprTDMAddrSwapA], s[sgprtdmAGroup0+1]
s_delay_alu instid0(SALU_CYCLE_1)
// set 2TDM adjustment
s_mov_b32 s[sgprTDMSplitA], 34816                  // (256+16)*128 = 34816 bytes
s_lshl_b32 s[sgprTDMGlobalSplitA], s[sgprSizeL], 0x7 // K * 128

// cluster sync
s_barrier_signal -1
s_barrier_wait -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_cbranch_scc0 label_Skip_Signal_0
s_barrier_signal -3                                // Cluster barrier
label_Skip_Signal_0:

s_barrier_wait -3
s_barrier_signal -1
s_barrier_wait -1
// PGR prefetch 1
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split

s_barrier_signal -1
s_barrier_wait -1

tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
s_mov_b32 s[sgprALdsSave], s[sgprtdmAGroup0+1]     // save A LDS base+split for next-tile reinit

// TDM increment global addr
s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
s_delay_alu instid0(SALU_CYCLE_1)
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split


/******************************************/
/* End setupNewTile                       */
/******************************************/
/* TDM swap lds a */
// TDM LDS swap(aligned pow2: False)
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]

// TDM LDS swap(aligned pow2: False)

s_barrier_signal -1
s_barrier_wait -1

// PGR prefetch 2
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM increment

s_barrier_signal -1
s_barrier_wait -1

tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]


s_barrier_signal -1
s_barrier_wait -1

// GL2 prefetch 1
s_set_vgpr_msb 771
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch

// GL2 prefetch pointers increment
s_set_vgpr_msb 972
v_add_nc_u64 v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1]
// GL2 prefetch 2
s_set_vgpr_msb 52227
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch

// GL2 prefetch pointers increment
s_set_vgpr_msb 972
v_add_nc_u64 v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1]

// PLR
s_wait_tensorcnt 3                                 // wait for prefetch 1
s_ttracedata_imm 0
s_barrier_signal -1
s_barrier_wait -1
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0

/* local read prefetch b0 */
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+0-512:vgprValuB_Y0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:0
ds_load_b128 v[vgprValuB_Y0+4-512:vgprValuB_Y0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:32
ds_load_b128 v[vgprValuB_Y0+8-512:vgprValuB_Y0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:64
ds_load_b128 v[vgprValuB_Y0+12-512:vgprValuB_Y0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:96
ds_load_b128 v[vgprValuB_Y0+16-512:vgprValuB_Y0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:8704
ds_load_b128 v[vgprValuB_Y0+20-512:vgprValuB_Y0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:8736
ds_load_b128 v[vgprValuB_Y0+24-512:vgprValuB_Y0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:8768
ds_load_b128 v[vgprValuB_Y0+28-512:vgprValuB_Y0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:8800
ds_load_b128 v[vgprValuB_Y0+32-512:vgprValuB_Y0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:17408
ds_load_b128 v[vgprValuB_Y0+36-512:vgprValuB_Y0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:17440
ds_load_b128 v[vgprValuB_Y0+40-512:vgprValuB_Y0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:17472
ds_load_b128 v[vgprValuB_Y0+44-512:vgprValuB_Y0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:17504
ds_load_b128 v[vgprValuB_Y0+48-512:vgprValuB_Y0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:26112
ds_load_b128 v[vgprValuB_Y0+52-512:vgprValuB_Y0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:26144
ds_load_b128 v[vgprValuB_Y0+56-512:vgprValuB_Y0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:26176
ds_load_b128 v[vgprValuB_Y0+60-512:vgprValuB_Y0+60-512+3], v[vgprLocalReadAddrB+0-512] offset:26208

/* local read prefetch a0 */
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+0-768:vgprValuA_X00+0-768+3], v[vgprLocalReadAddrA+0-512] offset:0
ds_load_b128 v[vgprValuA_X00+4-768:vgprValuA_X00+4-768+3], v[vgprLocalReadAddrA+0-512] offset:32
ds_load_b128 v[vgprValuA_X00+8-768:vgprValuA_X00+8-768+3], v[vgprLocalReadAddrA+0-512] offset:64
ds_load_b128 v[vgprValuA_X00+12-768:vgprValuA_X00+12-768+3], v[vgprLocalReadAddrA+0-512] offset:96
ds_load_b128 v[vgprValuA_X00+16-768:vgprValuA_X00+16-768+3], v[vgprLocalReadAddrA+0-512] offset:8704
ds_load_b128 v[vgprValuA_X00+20-768:vgprValuA_X00+20-768+3], v[vgprLocalReadAddrA+0-512] offset:8736
ds_load_b128 v[vgprValuA_X00+24-768:vgprValuA_X00+24-768+3], v[vgprLocalReadAddrA+0-512] offset:8768
ds_load_b128 v[vgprValuA_X00+28-768:vgprValuA_X00+28-768+3], v[vgprLocalReadAddrA+0-512] offset:8800
ds_load_b128 v[vgprValuA_X01+0-768:vgprValuA_X01+0-768+3], v[vgprLocalReadAddrA+0-512] offset:17408
ds_load_b128 v[vgprValuA_X01+4-768:vgprValuA_X01+4-768+3], v[vgprLocalReadAddrA+0-512] offset:17440
ds_load_b128 v[vgprValuA_X01+8-768:vgprValuA_X01+8-768+3], v[vgprLocalReadAddrA+0-512] offset:17472
ds_load_b128 v[vgprValuA_X01+12-768:vgprValuA_X01+12-768+3], v[vgprLocalReadAddrA+0-512] offset:17504
ds_load_b128 v[vgprValuA_X01+16-768:vgprValuA_X01+16-768+3], v[vgprLocalReadAddrA+0-512] offset:26112
ds_load_b128 v[vgprValuA_X01+20-768:vgprValuA_X01+20-768+3], v[vgprLocalReadAddrA+0-512] offset:26144
ds_load_b128 v[vgprValuA_X01+24-768:vgprValuA_X01+24-768+3], v[vgprLocalReadAddrA+0-512] offset:26176
ds_load_b128 v[vgprValuA_X01+28-768:vgprValuA_X01+28-768+3], v[vgprLocalReadAddrA+0-512] offset:26208

s_wait_tensorcnt 2                                 // wait for prefetch 1
s_ttracedata_imm 0
s_barrier_signal -1
s_barrier_wait -1

/* local read prefetch b1 */
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+0-512:vgprValuB_Y1+0-512+3], v[vgprLocalReadAddrB+0-512] offset:34816
ds_load_b128 v[vgprValuB_Y1+4-512:vgprValuB_Y1+4-512+3], v[vgprLocalReadAddrB+0-512] offset:34848
ds_load_b128 v[vgprValuB_Y1+8-512:vgprValuB_Y1+8-512+3], v[vgprLocalReadAddrB+0-512] offset:34880
ds_load_b128 v[vgprValuB_Y1+12-512:vgprValuB_Y1+12-512+3], v[vgprLocalReadAddrB+0-512] offset:34912
ds_load_b128 v[vgprValuB_Y1+16-512:vgprValuB_Y1+16-512+3], v[vgprLocalReadAddrB+0-512] offset:43520
ds_load_b128 v[vgprValuB_Y1+20-512:vgprValuB_Y1+20-512+3], v[vgprLocalReadAddrB+0-512] offset:43552
ds_load_b128 v[vgprValuB_Y1+24-512:vgprValuB_Y1+24-512+3], v[vgprLocalReadAddrB+0-512] offset:43584
ds_load_b128 v[vgprValuB_Y1+28-512:vgprValuB_Y1+28-512+3], v[vgprLocalReadAddrB+0-512] offset:43616
ds_load_b128 v[vgprValuB_Y1+32-512:vgprValuB_Y1+32-512+3], v[vgprLocalReadAddrB+0-512] offset:52224
ds_load_b128 v[vgprValuB_Y1+36-512:vgprValuB_Y1+36-512+3], v[vgprLocalReadAddrB+0-512] offset:52256
ds_load_b128 v[vgprValuB_Y1+40-512:vgprValuB_Y1+40-512+3], v[vgprLocalReadAddrB+0-512] offset:52288
ds_load_b128 v[vgprValuB_Y1+44-512:vgprValuB_Y1+44-512+3], v[vgprLocalReadAddrB+0-512] offset:52320
ds_load_b128 v[vgprValuB_Y1+48-512:vgprValuB_Y1+48-512+3], v[vgprLocalReadAddrB+0-512] offset:60928
ds_load_b128 v[vgprValuB_Y1+52-512:vgprValuB_Y1+52-512+3], v[vgprLocalReadAddrB+0-512] offset:60960
ds_load_b128 v[vgprValuB_Y1+56-512:vgprValuB_Y1+56-512+3], v[vgprLocalReadAddrB+0-512] offset:60992
ds_load_b128 v[vgprValuB_Y1+60-512:vgprValuB_Y1+60-512+3], v[vgprLocalReadAddrB+0-512] offset:61024

/* local read prefetch a1 */
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+0-768:vgprValuA_X10+0-768+3], v[vgprLocalReadAddrA+0-512] offset:34816
ds_load_b128 v[vgprValuA_X10+4-768:vgprValuA_X10+4-768+3], v[vgprLocalReadAddrA+0-512] offset:34848
ds_load_b128 v[vgprValuA_X10+8-768:vgprValuA_X10+8-768+3], v[vgprLocalReadAddrA+0-512] offset:34880
ds_load_b128 v[vgprValuA_X10+12-768:vgprValuA_X10+12-768+3], v[vgprLocalReadAddrA+0-512] offset:34912
ds_load_b128 v[vgprValuA_X10+16-768:vgprValuA_X10+16-768+3], v[vgprLocalReadAddrA+0-512] offset:43520
ds_load_b128 v[vgprValuA_X10+20-768:vgprValuA_X10+20-768+3], v[vgprLocalReadAddrA+0-512] offset:43552
ds_load_b128 v[vgprValuA_X10+24-768:vgprValuA_X10+24-768+3], v[vgprLocalReadAddrA+0-512] offset:43584
ds_load_b128 v[vgprValuA_X10+28-768:vgprValuA_X10+28-768+3], v[vgprLocalReadAddrA+0-512] offset:43616


/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
s_setreg_IMM32_b32 hwreg(26,0,2), 2                // set expert mode = 2 in unrolled loop
label_openLoopL:
label_Persist_Start_1:

label_Loop_InitC:
// /* iter 0 */
s_wait_dscnt 24                                    // MX + half A + half B
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+0-768:vgprValuA_X11+0-768+3], v[vgprLocalReadAddrA+0-512] offset:52224
ds_load_b128 v[vgprValuA_X11+4-768:vgprValuA_X11+4-768+3], v[vgprLocalReadAddrA+0-512] offset:52256
ds_load_b128 v[vgprValuA_X11+8-768:vgprValuA_X11+8-768+3], v[vgprLocalReadAddrA+0-512] offset:52288
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+12-768:vgprValuA_X11+12-768+3], v[vgprLocalReadAddrA+0-512] offset:52320
ds_load_b128 v[vgprValuA_X11+16-768:vgprValuA_X11+16-768+3], v[vgprLocalReadAddrA+0-512] offset:60928
ds_load_b128 v[vgprValuA_X11+20-768:vgprValuA_X11+20-768+3], v[vgprLocalReadAddrA+0-512] offset:60960
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+24-768:vgprValuA_X11+24-768+3], v[vgprLocalReadAddrA+0-512] offset:60992
ds_load_b128 v[vgprValuA_X11+28-768:vgprValuA_X11+28-768+3], v[vgprLocalReadAddrA+0-512] offset:61024
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], 0 matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], 0 matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+0-512:vgprValuB_Y2+0-512+3], v[vgprLocalReadAddrB+0-512] offset:128
ds_load_b128 v[vgprValuB_Y2+4-512:vgprValuB_Y2+4-512+3], v[vgprLocalReadAddrB+0-512] offset:160
ds_load_b128 v[vgprValuB_Y2+8-512:vgprValuB_Y2+8-512+3], v[vgprLocalReadAddrB+0-512] offset:192
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+12-512:vgprValuB_Y2+12-512+3], v[vgprLocalReadAddrB+0-512] offset:224
ds_load_b128 v[vgprValuB_Y2+16-512:vgprValuB_Y2+16-512+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_Y2+20-512:vgprValuB_Y2+20-512+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+24-512:vgprValuB_Y2+24-512+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_Y2+28-512:vgprValuB_Y2+28-512+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], 0 matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], 0 matrix_a_reuse

s_wait_dscnt 24                                    // another half B

s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+32-512:vgprValuB_Y2+32-512+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_Y2+36-512:vgprValuB_Y2+36-512+3], v[vgprLocalReadAddrB+0-512] offset:17568
ds_load_b128 v[vgprValuB_Y2+40-512:vgprValuB_Y2+40-512+3], v[vgprLocalReadAddrB+0-512] offset:17600
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+44-512:vgprValuB_Y2+44-512+3], v[vgprLocalReadAddrB+0-512] offset:17632
ds_load_b128 v[vgprValuB_Y2+48-512:vgprValuB_Y2+48-512+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_Y2+52-512:vgprValuB_Y2+52-512+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+56-512:vgprValuB_Y2+56-512+3], v[vgprLocalReadAddrB+0-512] offset:26304
ds_load_b128 v[vgprValuB_Y2+60-512:vgprValuB_Y2+60-512+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], 0 matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], 0 matrix_a_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+0-768:vgprValuA_X20+0-768+3], v[vgprLocalReadAddrA+0-512] offset:128
ds_load_b128 v[vgprValuA_X20+4-768:vgprValuA_X20+4-768+3], v[vgprLocalReadAddrA+0-512] offset:160
ds_load_b128 v[vgprValuA_X20+8-768:vgprValuA_X20+8-768+3], v[vgprLocalReadAddrA+0-512] offset:192
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+12-768:vgprValuA_X20+12-768+3], v[vgprLocalReadAddrA+0-512] offset:224
ds_load_b128 v[vgprValuA_X20+16-768:vgprValuA_X20+16-768+3], v[vgprLocalReadAddrA+0-512] offset:8832
ds_load_b128 v[vgprValuA_X20+20-768:vgprValuA_X20+20-768+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+24-768:vgprValuA_X20+24-768+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X20+28-768:vgprValuA_X20+28-768+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], 0 matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], 0 matrix_b_reuse

s_wait_dscnt 24                                    // another half A

s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_10
s_barrier_signal -3                                // Cluster barrier
label_Skip_Signal_10:

s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+0-768:vgprValuA_X01+0-768+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X01+4-768:vgprValuA_X01+4-768+3], v[vgprLocalReadAddrA+0-512] offset:17568
ds_load_b128 v[vgprValuA_X01+8-768:vgprValuA_X01+8-768+3], v[vgprLocalReadAddrA+0-512] offset:17600
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+12-768:vgprValuA_X01+12-768+3], v[vgprLocalReadAddrA+0-512] offset:17632
ds_load_b128 v[vgprValuA_X01+16-768:vgprValuA_X01+16-768+3], v[vgprLocalReadAddrA+0-512] offset:26240
ds_load_b128 v[vgprValuA_X01+20-768:vgprValuA_X01+20-768+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+24-768:vgprValuA_X01+24-768+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X01+28-768:vgprValuA_X01+28-768+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], 0 matrix_a_reuse
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], 0 matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+0-512:vgprValuB_Y3+0-512+3], v[vgprLocalReadAddrB+0-512] offset:34944
ds_load_b128 v[vgprValuB_Y3+4-512:vgprValuB_Y3+4-512+3], v[vgprLocalReadAddrB+0-512] offset:34976
ds_load_b128 v[vgprValuB_Y3+8-512:vgprValuB_Y3+8-512+3], v[vgprLocalReadAddrB+0-512] offset:35008
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+12-512:vgprValuB_Y3+12-512+3], v[vgprLocalReadAddrB+0-512] offset:35040
ds_load_b128 v[vgprValuB_Y3+16-512:vgprValuB_Y3+16-512+3], v[vgprLocalReadAddrB+0-512] offset:43648
ds_load_b128 v[vgprValuB_Y3+20-512:vgprValuB_Y3+20-512+3], v[vgprLocalReadAddrB+0-512] offset:43680
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+24-512:vgprValuB_Y3+24-512+3], v[vgprLocalReadAddrB+0-512] offset:43712
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+28-768:vgprValuB_Y3+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], 0 matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], 0 matrix_a_reuse
// GL2 prefetch
s_set_vgpr_msb 771
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch

s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+32-768:vgprValuB_Y3+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352
ds_load_b128 v[vgprValuB_Y3+36-768:vgprValuB_Y3+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384
ds_load_b128 v[vgprValuB_Y3+40-768:vgprValuB_Y3+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+44-768:vgprValuB_Y3+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
ds_load_b128 v[vgprValuB_Y3+48-768:vgprValuB_Y3+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
ds_load_b128 v[vgprValuB_Y3+52-768:vgprValuB_Y3+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+56-768:vgprValuB_Y3+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120
ds_load_b128 v[vgprValuB_Y3+60-768:vgprValuB_Y3+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], 0 matrix_a_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 75                                  // src0: 3, src1: 2, src2: 0, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], 0 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], 0 matrix_b_reuse
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], 0 matrix_a_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+0-768:vgprValuA_X21+0-768+3], v[vgprLocalReadAddrA+0-512] offset:34944
ds_load_b128 v[vgprValuA_X21+4-768:vgprValuA_X21+4-768+3], v[vgprLocalReadAddrA+0-512] offset:34976
ds_load_b128 v[vgprValuA_X21+8-768:vgprValuA_X21+8-768+3], v[vgprLocalReadAddrA+0-512] offset:35008
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+12-768:vgprValuA_X21+12-768+3], v[vgprLocalReadAddrA+0-512] offset:35040
ds_load_b128 v[vgprValuA_X21+16-768:vgprValuA_X21+16-768+3], v[vgprLocalReadAddrA+0-512] offset:43648
ds_load_b128 v[vgprValuA_X21+20-768:vgprValuA_X21+20-768+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], 0 matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+24-768:vgprValuA_X21+24-768+3], v[vgprLocalReadAddrA+0-512] offset:43712
ds_load_b128 v[vgprValuA_X21+28-768:vgprValuA_X21+28-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], 0 matrix_b_reuse
s_cmp_eq_i32 s[sgprIter], 3
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], 0 matrix_a_reuse
s_cselect_b32 s92, 0, 1
s_cmp_eq_i32 s[sgprLoopCounterL], 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], 0 matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], 0 matrix_b_reuse
s_cmov_b32 s[sgprtdmAGroup0+0], s92                // Set TDM as NULL in tail loops
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], 0 matrix_b_reuse
s_cmov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1] // update TDM to next AB addr
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], 0
s_wait_alu depctr_vm_vsrc(6)
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
s_branch label_Loop_InitC_End
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/
// /* iter 0 */
s_wait_dscnt 24                                    // MX + half A + half B
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+0:vgprValuC+0+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+0-768:vgprValuA_X11+0-768+3], v[vgprLocalReadAddrA+0-512] offset:52224
ds_load_b128 v[vgprValuA_X11+4-768:vgprValuA_X11+4-768+3], v[vgprLocalReadAddrA+0-512] offset:52256
ds_load_b128 v[vgprValuA_X11+8-768:vgprValuA_X11+8-768+3], v[vgprLocalReadAddrA+0-512] offset:52288
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+8:vgprValuC+8+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+12-768:vgprValuA_X11+12-768+3], v[vgprLocalReadAddrA+0-512] offset:52320
ds_load_b128 v[vgprValuA_X11+16-768:vgprValuA_X11+16-768+3], v[vgprLocalReadAddrA+0-512] offset:60928
ds_load_b128 v[vgprValuA_X11+20-768:vgprValuA_X11+20-768+3], v[vgprLocalReadAddrA+0-512] offset:60960
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+16:vgprValuC+16+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+24-768:vgprValuA_X11+24-768+3], v[vgprLocalReadAddrA+0-512] offset:60992
ds_load_b128 v[vgprValuA_X11+28-768:vgprValuA_X11+28-768+3], v[vgprLocalReadAddrA+0-512] offset:61024
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+24:vgprValuC+24+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+88:vgprValuC+88+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+80:vgprValuC+80+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+72:vgprValuC+72+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+64:vgprValuC+64+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+0-512:vgprValuB_Y2+0-512+3], v[vgprLocalReadAddrB+0-512] offset:128
ds_load_b128 v[vgprValuB_Y2+4-512:vgprValuB_Y2+4-512+3], v[vgprLocalReadAddrB+0-512] offset:160
ds_load_b128 v[vgprValuB_Y2+8-512:vgprValuB_Y2+8-512+3], v[vgprLocalReadAddrB+0-512] offset:192
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+128:vgprValuC+128+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+12-512:vgprValuB_Y2+12-512+3], v[vgprLocalReadAddrB+0-512] offset:224
ds_load_b128 v[vgprValuB_Y2+16-512:vgprValuB_Y2+16-512+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_Y2+20-512:vgprValuB_Y2+20-512+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+136:vgprValuC+136+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+24-512:vgprValuB_Y2+24-512+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_Y2+28-512:vgprValuB_Y2+28-512+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+144:vgprValuC+144+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+152:vgprValuC+152+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+216:vgprValuC+216+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+208:vgprValuC+208+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+200:vgprValuC+200+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+192:vgprValuC+192+7] matrix_a_reuse

s_wait_dscnt 24                                    // another half B

s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+256-256:vgprValuC+256-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+32-512:vgprValuB_Y2+32-512+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_Y2+36-512:vgprValuB_Y2+36-512+3], v[vgprLocalReadAddrB+0-512] offset:17568
ds_load_b128 v[vgprValuB_Y2+40-512:vgprValuB_Y2+40-512+3], v[vgprLocalReadAddrB+0-512] offset:17600
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+264-256:vgprValuC+264-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+44-512:vgprValuB_Y2+44-512+3], v[vgprLocalReadAddrB+0-512] offset:17632
ds_load_b128 v[vgprValuB_Y2+48-512:vgprValuB_Y2+48-512+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_Y2+52-512:vgprValuB_Y2+52-512+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+272-256:vgprValuC+272-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+56-512:vgprValuB_Y2+56-512+3], v[vgprLocalReadAddrB+0-512] offset:26304
ds_load_b128 v[vgprValuB_Y2+60-512:vgprValuB_Y2+60-512+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+280-256:vgprValuC+280-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+344-256:vgprValuC+344-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+336-256:vgprValuC+336-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+328-256:vgprValuC+328-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+320-256:vgprValuC+320-256+7] matrix_a_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+0-768:vgprValuA_X20+0-768+3], v[vgprLocalReadAddrA+0-512] offset:128
ds_load_b128 v[vgprValuA_X20+4-768:vgprValuA_X20+4-768+3], v[vgprLocalReadAddrA+0-512] offset:160
ds_load_b128 v[vgprValuA_X20+8-768:vgprValuA_X20+8-768+3], v[vgprLocalReadAddrA+0-512] offset:192
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+384-256:vgprValuC+384-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+12-768:vgprValuA_X20+12-768+3], v[vgprLocalReadAddrA+0-512] offset:224
ds_load_b128 v[vgprValuA_X20+16-768:vgprValuA_X20+16-768+3], v[vgprLocalReadAddrA+0-512] offset:8832
ds_load_b128 v[vgprValuA_X20+20-768:vgprValuA_X20+20-768+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+392-256:vgprValuC+392-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+24-768:vgprValuA_X20+24-768+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X20+28-768:vgprValuA_X20+28-768+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+400-256:vgprValuC+400-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+408-256:vgprValuC+408-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+472-256:vgprValuC+472-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+464-256:vgprValuC+464-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+456-256:vgprValuC+456-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+448-256:vgprValuC+448-256+7] matrix_b_reuse

s_wait_dscnt 24                                    // another half A

s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_10p
s_barrier_signal -3                                // Cluster barrier
label_Skip_Signal_10p:

s_set_vgpr_msb 171                                 // src0: 3, src1: 2, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+480-512:vgprValuC+480-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+0-768:vgprValuA_X01+0-768+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X01+4-768:vgprValuA_X01+4-768+3], v[vgprLocalReadAddrA+0-512] offset:17568
ds_load_b128 v[vgprValuA_X01+8-768:vgprValuA_X01+8-768+3], v[vgprLocalReadAddrA+0-512] offset:17600
s_set_vgpr_msb 171                                 // src0: 3, src1: 2, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+488-512:vgprValuC+488-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+12-768:vgprValuA_X01+12-768+3], v[vgprLocalReadAddrA+0-512] offset:17632
ds_load_b128 v[vgprValuA_X01+16-768:vgprValuA_X01+16-768+3], v[vgprLocalReadAddrA+0-512] offset:26240
ds_load_b128 v[vgprValuA_X01+20-768:vgprValuA_X01+20-768+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 171                                 // src0: 3, src1: 2, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+496-512:vgprValuC+496-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+24-768:vgprValuA_X01+24-768+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X01+28-768:vgprValuA_X01+28-768+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 171                                 // src0: 3, src1: 2, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+504-512:vgprValuC+504-512+7] matrix_a_reuse
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+440-256:vgprValuC+440-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+432-256:vgprValuC+432-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+424-256:vgprValuC+424-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+416-256:vgprValuC+416-256+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+0-512:vgprValuB_Y3+0-512+3], v[vgprLocalReadAddrB+0-512] offset:34944
ds_load_b128 v[vgprValuB_Y3+4-512:vgprValuB_Y3+4-512+3], v[vgprLocalReadAddrB+0-512] offset:34976
ds_load_b128 v[vgprValuB_Y3+8-512:vgprValuB_Y3+8-512+3], v[vgprLocalReadAddrB+0-512] offset:35008
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+352-256:vgprValuC+352-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+12-512:vgprValuB_Y3+12-512+3], v[vgprLocalReadAddrB+0-512] offset:35040
ds_load_b128 v[vgprValuB_Y3+16-512:vgprValuB_Y3+16-512+3], v[vgprLocalReadAddrB+0-512] offset:43648
ds_load_b128 v[vgprValuB_Y3+20-512:vgprValuB_Y3+20-512+3], v[vgprLocalReadAddrB+0-512] offset:43680
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+360-256:vgprValuC+360-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+24-512:vgprValuB_Y3+24-512+3], v[vgprLocalReadAddrB+0-512] offset:43712
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+28-768:vgprValuB_Y3+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+368-256:vgprValuC+368-256+7] matrix_b_reuse
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+376-256:vgprValuC+376-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+312-256:vgprValuC+312-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+304-256:vgprValuC+304-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+296-256:vgprValuC+296-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+288-256:vgprValuC+288-256+7] matrix_a_reuse
// GL2 prefetch
s_set_vgpr_msb 771
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch

s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+160:vgprValuC+160+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+32-768:vgprValuB_Y3+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352
ds_load_b128 v[vgprValuB_Y3+36-768:vgprValuB_Y3+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384
ds_load_b128 v[vgprValuB_Y3+40-768:vgprValuB_Y3+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+168:vgprValuC+168+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+44-768:vgprValuB_Y3+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
ds_load_b128 v[vgprValuB_Y3+48-768:vgprValuB_Y3+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
ds_load_b128 v[vgprValuB_Y3+52-768:vgprValuB_Y3+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+176:vgprValuC+176+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+56-768:vgprValuB_Y3+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120
ds_load_b128 v[vgprValuB_Y3+60-768:vgprValuB_Y3+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+184:vgprValuC+184+7] matrix_a_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+248-256:vgprValuC+248-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+240-256:vgprValuC+240-256+7] matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+232-256:vgprValuC+232-256+7] matrix_b_reuse
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+224-256:vgprValuC+224-256+7] matrix_a_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+0-768:vgprValuA_X21+0-768+3], v[vgprLocalReadAddrA+0-512] offset:34944
ds_load_b128 v[vgprValuA_X21+4-768:vgprValuA_X21+4-768+3], v[vgprLocalReadAddrA+0-512] offset:34976
ds_load_b128 v[vgprValuA_X21+8-768:vgprValuA_X21+8-768+3], v[vgprLocalReadAddrA+0-512] offset:35008
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+32:vgprValuC+32+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+12-768:vgprValuA_X21+12-768+3], v[vgprLocalReadAddrA+0-512] offset:35040
ds_load_b128 v[vgprValuA_X21+16-768:vgprValuA_X21+16-768+3], v[vgprLocalReadAddrA+0-512] offset:43648
ds_load_b128 v[vgprValuA_X21+20-768:vgprValuA_X21+20-768+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+40:vgprValuC+40+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+24-768:vgprValuA_X21+24-768+3], v[vgprLocalReadAddrA+0-512] offset:43712
ds_load_b128 v[vgprValuA_X21+28-768:vgprValuA_X21+28-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+48:vgprValuC+48+7] matrix_b_reuse
s_cmp_eq_i32 s[sgprIter], 3
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+56:vgprValuC+56+7] matrix_a_reuse
s_cselect_b32 s92, 0, 1
s_cmp_eq_i32 s[sgprLoopCounterL], 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+120:vgprValuC+120+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+112:vgprValuC+112+7] matrix_b_reuse
s_cmov_b32 s[sgprtdmAGroup0+0], s92                // Set TDM as NULL in tail loops
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+104:vgprValuC+104+7] matrix_b_reuse
s_cmov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1] // update TDM to next AB addr
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+96:vgprValuC+96+7]
s_wait_alu depctr_vm_vsrc(6)
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
label_Loop_InitC_End:

s_wait_dscnt 24                                    // MX + half A + half B
s_wait_alu depctr_va_vdst(0)
s_ttracedata_imm 0
s_barrier_wait -3
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

s_wait_tensorcnt 2

s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_4
s_barrier_signal -3
label_Skip_Signal_4:

s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+0:vgprValuC+0+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+0-768:vgprValuA_X11+0-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
ds_load_b128 v[vgprValuA_X11+4-768:vgprValuA_X11+4-768+3], v[vgprLocalReadAddrA+0-512] offset:52384
ds_load_b128 v[vgprValuA_X11+8-768:vgprValuA_X11+8-768+3], v[vgprLocalReadAddrA+0-512] offset:52416
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+8:vgprValuC+8+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+12-768:vgprValuA_X11+12-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X11+16-768:vgprValuA_X11+16-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
ds_load_b128 v[vgprValuA_X11+20-768:vgprValuA_X11+20-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+16:vgprValuC+16+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+24-768:vgprValuA_X11+24-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
ds_load_b128 v[vgprValuA_X11+28-768:vgprValuA_X11+28-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+24:vgprValuC+24+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+88:vgprValuC+88+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+80:vgprValuC+80+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+72:vgprValuC+72+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+64:vgprValuC+64+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+0-512:vgprValuB_Y0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:0
ds_load_b128 v[vgprValuB_Y0+4-512:vgprValuB_Y0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:32
ds_load_b128 v[vgprValuB_Y0+8-512:vgprValuB_Y0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:64
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+128:vgprValuC+128+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+12-512:vgprValuB_Y0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:96
ds_load_b128 v[vgprValuB_Y0+16-512:vgprValuB_Y0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:8704
ds_load_b128 v[vgprValuB_Y0+20-512:vgprValuB_Y0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:8736
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+136:vgprValuC+136+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+24-512:vgprValuB_Y0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:8768
ds_load_b128 v[vgprValuB_Y0+28-512:vgprValuB_Y0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:8800
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+144:vgprValuC+144+7] matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA]
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+152:vgprValuC+152+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+216:vgprValuC+216+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+208:vgprValuC+208+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+200:vgprValuC+200+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+192:vgprValuC+192+7] matrix_a_reuse
s_wait_alu depctr_vm_vsrc(6)
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk

s_wait_dscnt 24                                    // another half B

s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+256-256:vgprValuC+256-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+32-512:vgprValuB_Y0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:17408
ds_load_b128 v[vgprValuB_Y0+36-512:vgprValuB_Y0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:17440
ds_load_b128 v[vgprValuB_Y0+40-512:vgprValuB_Y0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:17472
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+264-256:vgprValuC+264-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+44-512:vgprValuB_Y0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:17504
ds_load_b128 v[vgprValuB_Y0+48-512:vgprValuB_Y0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:26112
ds_load_b128 v[vgprValuB_Y0+52-512:vgprValuB_Y0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:26144
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+272-256:vgprValuC+272-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+56-512:vgprValuB_Y0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:26176
ds_load_b128 v[vgprValuB_Y0+60-512:vgprValuB_Y0+60-512+3], v[vgprLocalReadAddrB+0-512] offset:26208
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+280-256:vgprValuC+280-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+344-256:vgprValuC+344-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+336-256:vgprValuC+336-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+328-256:vgprValuC+328-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+320-256:vgprValuC+320-256+7] matrix_a_reuse
s_wait_alu depctr_va_vdst(8)
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+0-768:vgprValuA_X00+0-768+3], v[vgprLocalReadAddrA+0-512] offset:0
ds_load_b128 v[vgprValuA_X00+4-768:vgprValuA_X00+4-768+3], v[vgprLocalReadAddrA+0-512] offset:32
ds_load_b128 v[vgprValuA_X00+8-768:vgprValuA_X00+8-768+3], v[vgprLocalReadAddrA+0-512] offset:64
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+12-768:vgprValuA_X00+12-768+3], v[vgprLocalReadAddrA+0-512] offset:96
ds_load_b128 v[vgprValuA_X00+16-768:vgprValuA_X00+16-768+3], v[vgprLocalReadAddrA+0-512] offset:8704
ds_load_b128 v[vgprValuA_X00+20-768:vgprValuA_X00+20-768+3], v[vgprLocalReadAddrA+0-512] offset:8736
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+24-768:vgprValuA_X00+24-768+3], v[vgprLocalReadAddrA+0-512] offset:8768
ds_load_b128 v[vgprValuA_X00+28-768:vgprValuA_X00+28-768+3], v[vgprLocalReadAddrA+0-512] offset:8800
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7] matrix_b_reuse

s_wait_dscnt 24                                    // another half A

s_ttracedata_imm 0
s_barrier_wait -3
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

s_wait_tensorcnt 2
s_barrier_signal -1
s_barrier_wait -1

s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+0-768:vgprValuA_X01+0-768+3], v[vgprLocalReadAddrA+0-512] offset:17408
ds_load_b128 v[vgprValuA_X01+4-768:vgprValuA_X01+4-768+3], v[vgprLocalReadAddrA+0-512] offset:17440
ds_load_b128 v[vgprValuA_X01+8-768:vgprValuA_X01+8-768+3], v[vgprLocalReadAddrA+0-512] offset:17472
s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+12-768:vgprValuA_X01+12-768+3], v[vgprLocalReadAddrA+0-512] offset:17504
ds_load_b128 v[vgprValuA_X01+16-768:vgprValuA_X01+16-768+3], v[vgprLocalReadAddrA+0-512] offset:26112
ds_load_b128 v[vgprValuA_X01+20-768:vgprValuA_X01+20-768+3], v[vgprLocalReadAddrA+0-512] offset:26144
s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+24-768:vgprValuA_X01+24-768+3], v[vgprLocalReadAddrA+0-512] offset:26176
ds_load_b128 v[vgprValuA_X01+28-768:vgprValuA_X01+28-768+3], v[vgprLocalReadAddrA+0-512] offset:26208
s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7] matrix_a_reuse
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+0-512:vgprValuB_Y1+0-512+3], v[vgprLocalReadAddrB+0-512] offset:34816
ds_load_b128 v[vgprValuB_Y1+4-512:vgprValuB_Y1+4-512+3], v[vgprLocalReadAddrB+0-512] offset:34848
ds_load_b128 v[vgprValuB_Y1+8-512:vgprValuB_Y1+8-512+3], v[vgprLocalReadAddrB+0-512] offset:34880
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+352-256:vgprValuC+352-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+12-512:vgprValuB_Y1+12-512+3], v[vgprLocalReadAddrB+0-512] offset:34912
ds_load_b128 v[vgprValuB_Y1+16-512:vgprValuB_Y1+16-512+3], v[vgprLocalReadAddrB+0-512] offset:43520
ds_load_b128 v[vgprValuB_Y1+20-512:vgprValuB_Y1+20-512+3], v[vgprLocalReadAddrB+0-512] offset:43552
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+360-256:vgprValuC+360-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+24-512:vgprValuB_Y1+24-512+3], v[vgprLocalReadAddrB+0-512] offset:43584
ds_load_b128 v[vgprValuB_Y1+28-512:vgprValuB_Y1+28-512+3], v[vgprLocalReadAddrB+0-512] offset:43616
s_cmp_eq_i32 s[sgprIter], 3
s_cselect_b32 s90, 1, 0
s_cmp_eq_i32 s[sgprLoopCounterL], 6
s_cselect_b32 s91, 1, 0
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+368-256:vgprValuC+368-256+7] matrix_b_reuse
s_and_b32 s92, s90, s91
s_cmov_b32 s[sgprPrefetchIncAB], 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+376-256:vgprValuC+376-256+7] matrix_a_reuse
s_set_vgpr_msb 204                                 // src0: 0, src1: 3, src2: 0, dst: 3
v_add_nc_u64 v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1]
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+312-256:vgprValuC+312-256+7] matrix_b_reuse
s_not_b32 s90, s90
s_and_b32 s92, s90, s91
v_cmp_eq_i32 vcc_lo, s92, 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+304-256:vgprValuC+304-256+7] matrix_b_reuse
s_set_vgpr_msb 207                                 // src0: 3, src1: 3, src2: 0, dst: 3
v_cndmask_b32 v[vgprGL2PrefetchAB+0-768], v[vgprGL2PrefetchAB+0-768], v[vgprGL2PrefetchABNext+0-768], vcc_lo
v_cndmask_b32 v[vgprGL2PrefetchAB+1-768], v[vgprGL2PrefetchAB+1-768], v[vgprGL2PrefetchABNext+1-768], vcc_lo
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+296-256:vgprValuC+296-256+7] matrix_b_reuse
s_set_vgpr_msb 207                                 // src0: 3, src1: 3, src2: 0, dst: 3
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+288-256:vgprValuC+288-256+7] matrix_a_reuse

s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+160:vgprValuC+160+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+32-512:vgprValuB_Y1+32-512+3], v[vgprLocalReadAddrB+0-512] offset:52224
ds_load_b128 v[vgprValuB_Y1+36-512:vgprValuB_Y1+36-512+3], v[vgprLocalReadAddrB+0-512] offset:52256
ds_load_b128 v[vgprValuB_Y1+40-512:vgprValuB_Y1+40-512+3], v[vgprLocalReadAddrB+0-512] offset:52288
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+168:vgprValuC+168+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+44-512:vgprValuB_Y1+44-512+3], v[vgprLocalReadAddrB+0-512] offset:52320
ds_load_b128 v[vgprValuB_Y1+48-512:vgprValuB_Y1+48-512+3], v[vgprLocalReadAddrB+0-512] offset:60928
ds_load_b128 v[vgprValuB_Y1+52-512:vgprValuB_Y1+52-512+3], v[vgprLocalReadAddrB+0-512] offset:60960
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+176:vgprValuC+176+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+56-512:vgprValuB_Y1+56-512+3], v[vgprLocalReadAddrB+0-512] offset:60992
ds_load_b128 v[vgprValuB_Y1+60-512:vgprValuB_Y1+60-512+3], v[vgprLocalReadAddrB+0-512] offset:61024
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+184:vgprValuC+184+7] matrix_a_reuse
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+248-256:vgprValuC+248-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+240-256:vgprValuC+240-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+232-256:vgprValuC+232-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+224-256:vgprValuC+224-256+7] matrix_a_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+0-768:vgprValuA_X10+0-768+3], v[vgprLocalReadAddrA+0-512] offset:34816
ds_load_b128 v[vgprValuA_X10+4-768:vgprValuA_X10+4-768+3], v[vgprLocalReadAddrA+0-512] offset:34848
ds_load_b128 v[vgprValuA_X10+8-768:vgprValuA_X10+8-768+3], v[vgprLocalReadAddrA+0-512] offset:34880
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+32:vgprValuC+32+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+12-768:vgprValuA_X10+12-768+3], v[vgprLocalReadAddrA+0-512] offset:34912
ds_load_b128 v[vgprValuA_X10+16-768:vgprValuA_X10+16-768+3], v[vgprLocalReadAddrA+0-512] offset:43520
ds_load_b128 v[vgprValuA_X10+20-768:vgprValuA_X10+20-768+3], v[vgprLocalReadAddrA+0-512] offset:43552
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+40:vgprValuC+40+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+24-768:vgprValuA_X10+24-768+3], v[vgprLocalReadAddrA+0-512] offset:43584
ds_load_b128 v[vgprValuA_X10+28-768:vgprValuA_X10+28-768+3], v[vgprLocalReadAddrA+0-512] offset:43616
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+48:vgprValuC+48+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+56:vgprValuC+56+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+120:vgprValuC+120+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+112:vgprValuC+112+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+104:vgprValuC+104+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+96:vgprValuC+96+7]

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 2 // dec counterL by 2 (one full 2DU cycle)
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_eq_i32 s[sgprLoopCounterL], 0                // passA was the LCL==2 NGLL chunk?
s_cbranch_scc0 label_passB_K2
.set sgprSrdD, 28
// non-persistent: Iter is pinned at the last index, so the store SRD init below must always run
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1]
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96
s_and_b32 s5, s[sgprSrdD+2], 127
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s5, s5, 25
s_and_b32 s[sgprSrdD+1], s[sgprSrdD+1], 33554431
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprSrdD+1], s[sgprSrdD+1], s5
s_lshr_b32 s[sgprSrdD+2], s[sgprSrdD+2], 7
label_skip_first_WG_init:
s_mul_i32 s86, MT1, s[sgprWorkGroup1]
s_mul_hi_u32 s85, s86, s[sgprStrideD1J]
s_mul_i32 s84, s86, s[sgprStrideD1J]
s_lshl_b64 s[84:85], s[84:85], s[sgprGSULog2BpeD]
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s84
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s85
label_GW_B0_E0_1:
s_add_u32 s[sgprIter], s[sgprIter], 1
label_glds:
s_nop 0
s_set_vgpr_msb 204                                 // src0: 0, src1: 3, src2: 0, dst: 3
v_lshrrev_b32 v[vgprStoreAddr-768], 6, v[vgprSerial-768] // row: Serial>>6
v_lshlrev_b32 v[vgprStoreAddr-768], 4, v[vgprStoreAddr-768] // row: (Serial>>6)<<4
v_and_b32 v[vgprSRSeed-768], 15, v[vgprSerial-768] // row: Serial&15
s_set_vgpr_msb 207                                 // src0: 3, src1: 3, src2: 0, dst: 3
v_add_nc_u32 v[vgprStoreAddr-768], v[vgprStoreAddr-768], v[vgprSRSeed-768]
v_mul_lo_u32 v[vgprStoreAddr-768], v[vgprStoreAddr-768], s[sgprStrideD1J] // row offset
v_lshrrev_b32 v[vgprSRSeed-768], 4, v[vgprSerial-768] // col: Serial>>4
v_and_b32 v[vgprSRSeed-768], 1, v[vgprSRSeed-768]
v_lshlrev_b32 v[vgprSRSeed-768], 5, v[vgprSRSeed-768] // col: (Serial>>4&1)<<5
v_add_nc_u32 v[vgprStoreAddr-768], v[vgprStoreAddr-768], v[vgprSRSeed-768]
v_lshrrev_b32 v[vgprSRSeed-768], 5, v[vgprSerial-768] // col: Serial>>5
v_and_b32 v[vgprSRSeed-768], 1, v[vgprSRSeed-768]
v_lshlrev_b32 v[vgprSRSeed-768], 4, v[vgprSRSeed-768] // col: (Serial>>5&1)<<4
v_add_nc_u32 v[vgprStoreAddr-768], v[vgprStoreAddr-768], v[vgprSRSeed-768]
s_mul_i32 s84, 256, s[sgprWorkGroup0]
s_delay_alu instid0(SALU_CYCLE_1)
v_add_nc_u32 v[vgprStoreAddr-768], s84, v[vgprStoreAddr-768] // + WG0*256 (FINAL store addr)
s_mov_b32 s[sgprAsyncWG0], s[sgprWorkGroup0]       // save current-tile WG0 for async drain (reinit clobbers WorkGroup0)
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v[vgprAsyncAddr], 6, v[vgprSerial-768] // LWA row: Serial>>6
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshlrev_b32 v[vgprAsyncAddr], 4, v[vgprAsyncAddr] // LWA row: (Serial>>6)<<4
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v[vgprAsyncAddr+1], 15, v[vgprSerial-768] // LWA row: Serial&15
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_add_nc_u32 v[vgprAsyncAddr], v[vgprAsyncAddr], v[vgprAsyncAddr+1] // LWA row
v_lshlrev_b32 v[vgprAsyncAddr+1], 4, v[vgprAsyncAddr] // row*16
v_lshlrev_b32 v[vgprAsyncAddr], 8, v[vgprAsyncAddr] // row*256
v_add_nc_u32 v[vgprAsyncAddr], v[vgprAsyncAddr], v[vgprAsyncAddr+1] // row*(256+16)
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v[vgprAsyncAddr+1], 4, v[vgprSerial-768] // col: Serial>>4
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v[vgprAsyncAddr+1], 1, v[vgprAsyncAddr+1]
v_lshlrev_b32 v[vgprAsyncAddr+1], 5, v[vgprAsyncAddr+1] // col: (Serial>>4&1)<<5
v_add_nc_u32 v[vgprAsyncAddr], v[vgprAsyncAddr], v[vgprAsyncAddr+1]
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v[vgprAsyncAddr+1], 5, v[vgprSerial-768] // col: Serial>>5
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v[vgprAsyncAddr+1], 1, v[vgprAsyncAddr+1]
v_lshlrev_b32 v[vgprAsyncAddr+1], 4, v[vgprAsyncAddr+1] // col: (Serial>>5&1)<<4
v_add_nc_u32 v[vgprAsyncAddr], v[vgprAsyncAddr], v[vgprAsyncAddr+1]
v_add_nc_u32 v[vgprAsyncAddr], v[vgprAsyncAddr], 287744 // + staging base (LWA final)
s_set_vgpr_msb 192                                 // src0: 0, src1: 0, src2: 0, dst: 3
v_prng_b32 v[vgprSRSeed-768], v[vgprValuC+-24]
s_mul_i32 s85, s[sgprStrideD1J], 32
s_mov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1] // A descriptor := next-N-tile (T+1) base [before recompute]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // pre-bias +split so leading -split lands load0 at base
s_mov_b32 s[sgprWorkGroup0], s[sgprWorkGroup0Next]
s_mov_b32 s[sgprWorkGroup1], s[sgprWorkGroup1Next]
s_add_u32 s90, s[sgprIter], 1
s_and_b32 s90, s90, 1
s_cselect_b32 s92, 1, -1
s_cselect_b32 s90, 0, 1
s_lshr_b32 s91, s[sgprNumWorkGroups0], 1
s_mul_i32 s92, s91, s92
s_add_i32 s[sgprWorkGroup0Next], s[sgprWorkGroup0Next], s92
s_lshr_b32 s91, s[sgprNumWorkGroups1], 1
s_mul_i32 s90, s91, s90
s_add_i32 s[sgprWorkGroup1Next], s[sgprWorkGroup1Next], s90
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s90, v[vgprSerial-768]
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s90, s90, 5
s_delay_alu instid0(SALU_CYCLE_1)
s_bitcmp1_b32 s90, 0
s_cbranch_scc1 label_NextAddrB
label_NextAddrA:
s_mul_i32 s91, s[sgprWorkGroup0Next], 256
s_mul_i32 s91, s91, s[sgprStrideA0I]
s_add_u32 s[sgprTDMAddrABNext], s[sgprTDMAddrABNoWG], s91
s_addc_u32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNoWG+1], 0
s_or_b32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNext+1], 0x80000000
s_mul_i32 s91, s[sgprWorkGroup0Next], 1024
s_mov_b32 s91, 0
s_mul_i32 s90, s[sgprStrideA0I], 256
s_mul_i32 s90, s90, s[sgprWorkGroup0Next]
s_set_vgpr_msb 972
v_add_nc_u64 v[vgprGL2PrefetchABNext+0-768:vgprGL2PrefetchABNext+0-768+1], s[90:91], v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1]
s_mul_i32 s90, s[sgprWorkGroup0Next], 1024
s_cmp_ge_u32 s[sgprWorkGroup0Next], s[sgprNumWorkGroups0]
s_cselect_b32 s90, 1, 0
s_cmp_ge_u32 s[sgprWorkGroup1Next], s[sgprNumWorkGroups1]
s_cselect_b32 s91, 1, 0
s_or_b32 s90, s90, s91                             // next-tile OOB (either WG dim past grid)
v_cmp_ne_u32 vcc_lo, s90, 0
s_set_vgpr_msb 207                                 // src0: 3, src1: 3, src2: 0, dst: 3
v_cndmask_b32 v[vgprGL2PrefetchABNext+0-768], v[vgprGL2PrefetchABNext+0-768], v[vgprGL2PrefetchABNoWG+0-768], vcc_lo
v_cndmask_b32 v[vgprGL2PrefetchABNext+1-768], v[vgprGL2PrefetchABNext+1-768], v[vgprGL2PrefetchABNoWG+1-768], vcc_lo
s_branch label_NextAddrEnd
label_NextAddrB:
s_mul_i32 s91, s[sgprWorkGroup1Next], 256
s_mul_i32 s91, s91, s[sgprStrideB1J]
s_add_u32 s[sgprTDMAddrABNext], s[sgprTDMAddrABNoWG], s91
s_addc_u32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNoWG+1], 0
s_or_b32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNext+1], 0x80000000
s_mul_i32 s91, s[sgprWorkGroup1Next], 1024
s_mov_b32 s91, 0
s_mul_i32 s90, s[sgprStrideB1J], 256
s_mul_i32 s90, s90, s[sgprWorkGroup1Next]
s_set_vgpr_msb 52428
v_add_nc_u64 v[vgprGL2PrefetchABNext+0-768:vgprGL2PrefetchABNext+0-768+1], s[90:91], v[vgprGL2PrefetchABNoWG+0-768:vgprGL2PrefetchABNoWG+0-768+1]
s_mul_i32 s90, s[sgprWorkGroup1Next], 1024
s_cmp_ge_u32 s[sgprWorkGroup0Next], s[sgprNumWorkGroups0]
s_cselect_b32 s90, 1, 0
s_cmp_ge_u32 s[sgprWorkGroup1Next], s[sgprNumWorkGroups1]
s_cselect_b32 s91, 1, 0
s_or_b32 s90, s90, s91                             // next-tile OOB (either WG dim past grid)
v_cmp_ne_u32 vcc_lo, s90, 0
s_set_vgpr_msb 207                                 // src0: 3, src1: 3, src2: 0, dst: 3
v_cndmask_b32 v[vgprGL2PrefetchABNext+0-768], v[vgprGL2PrefetchABNext+0-768], v[vgprGL2PrefetchABNoWG+0-768], vcc_lo
v_cndmask_b32 v[vgprGL2PrefetchABNext+1-768], v[vgprGL2PrefetchABNext+1-768], v[vgprGL2PrefetchABNoWG+1-768], vcc_lo
label_NextAddrEnd:
s_barrier_signal -1
s_mov_b32 s[sgprtdmAGroup0+1], s[sgprALdsSave]     // re-anchor A LDS +1 = saved DU0 base+split
s_barrier_wait -1
label_passB_K2:

/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/
// /* iter 0 */
s_wait_dscnt 24                                    // MX + half A + half B
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+0:vgprValuC+0+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+0-768:vgprValuA_X11+0-768+3], v[vgprLocalReadAddrA+0-512] offset:52224
ds_load_b128 v[vgprValuA_X11+4-768:vgprValuA_X11+4-768+3], v[vgprLocalReadAddrA+0-512] offset:52256
ds_load_b128 v[vgprValuA_X11+8-768:vgprValuA_X11+8-768+3], v[vgprLocalReadAddrA+0-512] offset:52288
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+8:vgprValuC+8+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+12-768:vgprValuA_X11+12-768+3], v[vgprLocalReadAddrA+0-512] offset:52320
ds_load_b128 v[vgprValuA_X11+16-768:vgprValuA_X11+16-768+3], v[vgprLocalReadAddrA+0-512] offset:60928
ds_load_b128 v[vgprValuA_X11+20-768:vgprValuA_X11+20-768+3], v[vgprLocalReadAddrA+0-512] offset:60960
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+16:vgprValuC+16+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+24-768:vgprValuA_X11+24-768+3], v[vgprLocalReadAddrA+0-512] offset:60992
ds_load_b128 v[vgprValuA_X11+28-768:vgprValuA_X11+28-768+3], v[vgprLocalReadAddrA+0-512] offset:61024
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+24:vgprValuC+24+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+88:vgprValuC+88+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+80:vgprValuC+80+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+72:vgprValuC+72+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+64:vgprValuC+64+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+0-512:vgprValuB_Y2+0-512+3], v[vgprLocalReadAddrB+0-512] offset:128
ds_load_b128 v[vgprValuB_Y2+4-512:vgprValuB_Y2+4-512+3], v[vgprLocalReadAddrB+0-512] offset:160
ds_load_b128 v[vgprValuB_Y2+8-512:vgprValuB_Y2+8-512+3], v[vgprLocalReadAddrB+0-512] offset:192
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+128:vgprValuC+128+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+12-512:vgprValuB_Y2+12-512+3], v[vgprLocalReadAddrB+0-512] offset:224
ds_load_b128 v[vgprValuB_Y2+16-512:vgprValuB_Y2+16-512+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_Y2+20-512:vgprValuB_Y2+20-512+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+136:vgprValuC+136+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+24-512:vgprValuB_Y2+24-512+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_Y2+28-512:vgprValuB_Y2+28-512+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+144:vgprValuC+144+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+152:vgprValuC+152+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+216:vgprValuC+216+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+208:vgprValuC+208+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+200:vgprValuC+200+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+192:vgprValuC+192+7] matrix_a_reuse

s_wait_dscnt 24                                    // another half B

s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+256-256:vgprValuC+256-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+32-512:vgprValuB_Y2+32-512+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_Y2+36-512:vgprValuB_Y2+36-512+3], v[vgprLocalReadAddrB+0-512] offset:17568
ds_load_b128 v[vgprValuB_Y2+40-512:vgprValuB_Y2+40-512+3], v[vgprLocalReadAddrB+0-512] offset:17600
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+264-256:vgprValuC+264-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+44-512:vgprValuB_Y2+44-512+3], v[vgprLocalReadAddrB+0-512] offset:17632
ds_load_b128 v[vgprValuB_Y2+48-512:vgprValuB_Y2+48-512+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_Y2+52-512:vgprValuB_Y2+52-512+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+272-256:vgprValuC+272-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y2+56-512:vgprValuB_Y2+56-512+3], v[vgprLocalReadAddrB+0-512] offset:26304
ds_load_b128 v[vgprValuB_Y2+60-512:vgprValuB_Y2+60-512+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+280-256:vgprValuC+280-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+344-256:vgprValuC+344-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+336-256:vgprValuC+336-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+328-256:vgprValuC+328-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+320-256:vgprValuC+320-256+7] matrix_a_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+0-768:vgprValuA_X20+0-768+3], v[vgprLocalReadAddrA+0-512] offset:128
ds_load_b128 v[vgprValuA_X20+4-768:vgprValuA_X20+4-768+3], v[vgprLocalReadAddrA+0-512] offset:160
ds_load_b128 v[vgprValuA_X20+8-768:vgprValuA_X20+8-768+3], v[vgprLocalReadAddrA+0-512] offset:192
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+384-256:vgprValuC+384-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+12-768:vgprValuA_X20+12-768+3], v[vgprLocalReadAddrA+0-512] offset:224
ds_load_b128 v[vgprValuA_X20+16-768:vgprValuA_X20+16-768+3], v[vgprLocalReadAddrA+0-512] offset:8832
ds_load_b128 v[vgprValuA_X20+20-768:vgprValuA_X20+20-768+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+392-256:vgprValuC+392-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X20+24-768:vgprValuA_X20+24-768+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X20+28-768:vgprValuA_X20+28-768+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+400-256:vgprValuC+400-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+408-256:vgprValuC+408-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+472-256:vgprValuC+472-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+464-256:vgprValuC+464-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X00+16+0+0-768:vgprValuA_X00+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+456-256:vgprValuC+456-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X00+0+0+0-768:vgprValuA_X00+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+448-256:vgprValuC+448-256+7] matrix_b_reuse

s_wait_dscnt 24                                    // another half A

s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_5
s_barrier_signal -3
label_Skip_Signal_5:

s_set_vgpr_msb 171                                 // src0: 3, src1: 2, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+480-512:vgprValuC+480-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+0-768:vgprValuA_X01+0-768+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X01+4-768:vgprValuA_X01+4-768+3], v[vgprLocalReadAddrA+0-512] offset:17568
ds_load_b128 v[vgprValuA_X01+8-768:vgprValuA_X01+8-768+3], v[vgprLocalReadAddrA+0-512] offset:17600
s_set_vgpr_msb 171                                 // src0: 3, src1: 2, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+488-512:vgprValuC+488-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+12-768:vgprValuA_X01+12-768+3], v[vgprLocalReadAddrA+0-512] offset:17632
ds_load_b128 v[vgprValuA_X01+16-768:vgprValuA_X01+16-768+3], v[vgprLocalReadAddrA+0-512] offset:26240
ds_load_b128 v[vgprValuA_X01+20-768:vgprValuA_X01+20-768+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 171                                 // src0: 3, src1: 2, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+496-512:vgprValuC+496-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+24-768:vgprValuA_X01+24-768+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X01+28-768:vgprValuA_X01+28-768+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 171                                 // src0: 3, src1: 2, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+48+0+0-512:vgprValuB_Y1+48+0+0-512+15], v[vgprValuC+504-512:vgprValuC+504-512+7] matrix_a_reuse
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+440-256:vgprValuC+440-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+432-256:vgprValuC+432-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+424-256:vgprValuC+424-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+32+0+0-512:vgprValuB_Y1+32+0+0-512+15], v[vgprValuC+416-256:vgprValuC+416-256+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+0-512:vgprValuB_Y3+0-512+3], v[vgprLocalReadAddrB+0-512] offset:34944
ds_load_b128 v[vgprValuB_Y3+4-512:vgprValuB_Y3+4-512+3], v[vgprLocalReadAddrB+0-512] offset:34976
ds_load_b128 v[vgprValuB_Y3+8-512:vgprValuB_Y3+8-512+3], v[vgprLocalReadAddrB+0-512] offset:35008
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+352-256:vgprValuC+352-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+12-512:vgprValuB_Y3+12-512+3], v[vgprLocalReadAddrB+0-512] offset:35040
ds_load_b128 v[vgprValuB_Y3+16-512:vgprValuB_Y3+16-512+3], v[vgprLocalReadAddrB+0-512] offset:43648
ds_load_b128 v[vgprValuB_Y3+20-512:vgprValuB_Y3+20-512+3], v[vgprLocalReadAddrB+0-512] offset:43680
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+360-256:vgprValuC+360-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y3+24-512:vgprValuB_Y3+24-512+3], v[vgprLocalReadAddrB+0-512] offset:43712
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+28-768:vgprValuB_Y3+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+368-256:vgprValuC+368-256+7] matrix_b_reuse
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+16+0+0-512:vgprValuB_Y1+16+0+0-512+15], v[vgprValuC+376-256:vgprValuC+376-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+312-256:vgprValuC+312-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+304-256:vgprValuC+304-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+296-256:vgprValuC+296-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y1+0+0+0-512:vgprValuB_Y1+0+0+0-512+15], v[vgprValuC+288-256:vgprValuC+288-256+7] matrix_a_reuse
// GL2 prefetch
s_set_vgpr_msb 771
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch

s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+160:vgprValuC+160+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+32-768:vgprValuB_Y3+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352
ds_load_b128 v[vgprValuB_Y3+36-768:vgprValuB_Y3+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384
ds_load_b128 v[vgprValuB_Y3+40-768:vgprValuB_Y3+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+168:vgprValuC+168+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+44-768:vgprValuB_Y3+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
ds_load_b128 v[vgprValuB_Y3+48-768:vgprValuB_Y3+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
ds_load_b128 v[vgprValuB_Y3+52-768:vgprValuB_Y3+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+176:vgprValuC+176+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_Y3+56-768:vgprValuB_Y3+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120
ds_load_b128 v[vgprValuB_Y3+60-768:vgprValuB_Y3+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+32+0+0-512:vgprValuB_Y0+32+0+0-512+15], v[vgprValuC+184:vgprValuC+184+7] matrix_a_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+248-256:vgprValuC+248-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+240-256:vgprValuC+240-256+7] matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+232-256:vgprValuC+232-256+7] matrix_b_reuse
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+48+0+0-512:vgprValuB_Y0+48+0+0-512+15], v[vgprValuC+224-256:vgprValuC+224-256+7] matrix_a_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+0-768:vgprValuA_X21+0-768+3], v[vgprLocalReadAddrA+0-512] offset:34944
ds_load_b128 v[vgprValuA_X21+4-768:vgprValuA_X21+4-768+3], v[vgprLocalReadAddrA+0-512] offset:34976
ds_load_b128 v[vgprValuA_X21+8-768:vgprValuA_X21+8-768+3], v[vgprLocalReadAddrA+0-512] offset:35008
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+32:vgprValuC+32+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+12-768:vgprValuA_X21+12-768+3], v[vgprLocalReadAddrA+0-512] offset:35040
ds_load_b128 v[vgprValuA_X21+16-768:vgprValuA_X21+16-768+3], v[vgprLocalReadAddrA+0-512] offset:43648
ds_load_b128 v[vgprValuA_X21+20-768:vgprValuA_X21+20-768+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+40:vgprValuC+40+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X21+24-768:vgprValuA_X21+24-768+3], v[vgprLocalReadAddrA+0-512] offset:43712
ds_load_b128 v[vgprValuA_X21+28-768:vgprValuA_X21+28-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+48:vgprValuC+48+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+0+0+0-512:vgprValuB_Y0+0+0+0-512+15], v[vgprValuC+56:vgprValuC+56+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+120:vgprValuC+120+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+112:vgprValuC+112+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X10+16+0+0-768:vgprValuA_X10+16+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+104:vgprValuC+104+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X10+0+0+0-768:vgprValuA_X10+0+0+0-768+15], v[vgprValuB_Y0+16+0+0-512:vgprValuB_Y0+16+0+0-512+15], v[vgprValuC+96:vgprValuC+96+7]
s_wait_alu depctr_vm_vsrc(6)
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
s_cmp_eq_i32 s[sgprLoopCounterL], 0                // last K3 -> store-interleaved NLL
s_cbranch_scc1 label_NLL_store

s_wait_dscnt 24                                    // MX + half A + half B
s_wait_alu depctr_va_vdst(0)

s_ttracedata_imm 0
s_barrier_wait -3
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

s_wait_tensorcnt 2
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_6
s_barrier_signal -3
label_Skip_Signal_6:

s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+0:vgprValuC+0+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+0-768:vgprValuA_X11+0-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
ds_load_b128 v[vgprValuA_X11+4-768:vgprValuA_X11+4-768+3], v[vgprLocalReadAddrA+0-512] offset:52384
ds_load_b128 v[vgprValuA_X11+8-768:vgprValuA_X11+8-768+3], v[vgprLocalReadAddrA+0-512] offset:52416
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+8:vgprValuC+8+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+12-768:vgprValuA_X11+12-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X11+16-768:vgprValuA_X11+16-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
ds_load_b128 v[vgprValuA_X11+20-768:vgprValuA_X11+20-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+16:vgprValuC+16+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+24-768:vgprValuA_X11+24-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
ds_load_b128 v[vgprValuA_X11+28-768:vgprValuA_X11+28-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+24:vgprValuC+24+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+88:vgprValuC+88+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+80:vgprValuC+80+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+72:vgprValuC+72+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+64:vgprValuC+64+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+0-512:vgprValuB_Y0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:0
ds_load_b128 v[vgprValuB_Y0+4-512:vgprValuB_Y0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:32
ds_load_b128 v[vgprValuB_Y0+8-512:vgprValuB_Y0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:64
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+128:vgprValuC+128+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+12-512:vgprValuB_Y0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:96
ds_load_b128 v[vgprValuB_Y0+16-512:vgprValuB_Y0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:8704
ds_load_b128 v[vgprValuB_Y0+20-512:vgprValuB_Y0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:8736
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+136:vgprValuC+136+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+24-512:vgprValuB_Y0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:8768
ds_load_b128 v[vgprValuB_Y0+28-512:vgprValuB_Y0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:8800
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+144:vgprValuC+144+7] matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA]
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+152:vgprValuC+152+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+216:vgprValuC+216+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+208:vgprValuC+208+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+200:vgprValuC+200+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+192:vgprValuC+192+7] matrix_a_reuse
s_wait_alu depctr_vm_vsrc(6)
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk

s_wait_dscnt 24                                    // another half B

s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+256-256:vgprValuC+256-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+32-512:vgprValuB_Y0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:17408
ds_load_b128 v[vgprValuB_Y0+36-512:vgprValuB_Y0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:17440
ds_load_b128 v[vgprValuB_Y0+40-512:vgprValuB_Y0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:17472
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+264-256:vgprValuC+264-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+44-512:vgprValuB_Y0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:17504
ds_load_b128 v[vgprValuB_Y0+48-512:vgprValuB_Y0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:26112
ds_load_b128 v[vgprValuB_Y0+52-512:vgprValuB_Y0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:26144
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+272-256:vgprValuC+272-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+56-512:vgprValuB_Y0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:26176
ds_load_b128 v[vgprValuB_Y0+60-512:vgprValuB_Y0+60-512+3], v[vgprLocalReadAddrB+0-512] offset:26208
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+280-256:vgprValuC+280-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+344-256:vgprValuC+344-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+336-256:vgprValuC+336-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+328-256:vgprValuC+328-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+320-256:vgprValuC+320-256+7] matrix_a_reuse
s_wait_alu depctr_va_vdst(8)
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+0-768:vgprValuA_X00+0-768+3], v[vgprLocalReadAddrA+0-512] offset:0
ds_load_b128 v[vgprValuA_X00+4-768:vgprValuA_X00+4-768+3], v[vgprLocalReadAddrA+0-512] offset:32
ds_load_b128 v[vgprValuA_X00+8-768:vgprValuA_X00+8-768+3], v[vgprLocalReadAddrA+0-512] offset:64
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+12-768:vgprValuA_X00+12-768+3], v[vgprLocalReadAddrA+0-512] offset:96
ds_load_b128 v[vgprValuA_X00+16-768:vgprValuA_X00+16-768+3], v[vgprLocalReadAddrA+0-512] offset:8704
ds_load_b128 v[vgprValuA_X00+20-768:vgprValuA_X00+20-768+3], v[vgprLocalReadAddrA+0-512] offset:8736
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+24-768:vgprValuA_X00+24-768+3], v[vgprLocalReadAddrA+0-512] offset:8768
ds_load_b128 v[vgprValuA_X00+28-768:vgprValuA_X00+28-768+3], v[vgprLocalReadAddrA+0-512] offset:8800
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7] matrix_b_reuse

s_wait_dscnt 24                                    // another half A
s_ttracedata_imm 0
s_barrier_wait -3
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

s_wait_tensorcnt 2
s_barrier_signal -1
s_barrier_wait -1

s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+0-768:vgprValuA_X01+0-768+3], v[vgprLocalReadAddrA+0-512] offset:17408
ds_load_b128 v[vgprValuA_X01+4-768:vgprValuA_X01+4-768+3], v[vgprLocalReadAddrA+0-512] offset:17440
ds_load_b128 v[vgprValuA_X01+8-768:vgprValuA_X01+8-768+3], v[vgprLocalReadAddrA+0-512] offset:17472
s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+12-768:vgprValuA_X01+12-768+3], v[vgprLocalReadAddrA+0-512] offset:17504
ds_load_b128 v[vgprValuA_X01+16-768:vgprValuA_X01+16-768+3], v[vgprLocalReadAddrA+0-512] offset:26112
ds_load_b128 v[vgprValuA_X01+20-768:vgprValuA_X01+20-768+3], v[vgprLocalReadAddrA+0-512] offset:26144
s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+24-768:vgprValuA_X01+24-768+3], v[vgprLocalReadAddrA+0-512] offset:26176
ds_load_b128 v[vgprValuA_X01+28-768:vgprValuA_X01+28-768+3], v[vgprLocalReadAddrA+0-512] offset:26208
s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7] matrix_a_reuse
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+0-512:vgprValuB_Y1+0-512+3], v[vgprLocalReadAddrB+0-512] offset:34816
ds_load_b128 v[vgprValuB_Y1+4-512:vgprValuB_Y1+4-512+3], v[vgprLocalReadAddrB+0-512] offset:34848
ds_load_b128 v[vgprValuB_Y1+8-512:vgprValuB_Y1+8-512+3], v[vgprLocalReadAddrB+0-512] offset:34880
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+352-256:vgprValuC+352-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+12-512:vgprValuB_Y1+12-512+3], v[vgprLocalReadAddrB+0-512] offset:34912
ds_load_b128 v[vgprValuB_Y1+16-512:vgprValuB_Y1+16-512+3], v[vgprLocalReadAddrB+0-512] offset:43520
ds_load_b128 v[vgprValuB_Y1+20-512:vgprValuB_Y1+20-512+3], v[vgprLocalReadAddrB+0-512] offset:43552
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+360-256:vgprValuC+360-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+24-512:vgprValuB_Y1+24-512+3], v[vgprLocalReadAddrB+0-512] offset:43584
ds_load_b128 v[vgprValuB_Y1+28-512:vgprValuB_Y1+28-512+3], v[vgprLocalReadAddrB+0-512] offset:43616
s_cmp_eq_i32 s[sgprIter], 3
s_cselect_b32 s90, 1, 0
s_cmp_eq_i32 s[sgprLoopCounterL], 6
s_cselect_b32 s91, 1, 0
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+368-256:vgprValuC+368-256+7] matrix_b_reuse
s_and_b32 s92, s90, s91
s_cmov_b32 s[sgprPrefetchIncAB], 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+376-256:vgprValuC+376-256+7] matrix_a_reuse
s_set_vgpr_msb 204                                 // src0: 0, src1: 3, src2: 0, dst: 3
v_add_nc_u64 v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1]
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+312-256:vgprValuC+312-256+7] matrix_b_reuse
s_not_b32 s90, s90
s_and_b32 s92, s90, s91
v_cmp_eq_i32 vcc_lo, s92, 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+304-256:vgprValuC+304-256+7] matrix_b_reuse
s_set_vgpr_msb 207                                 // src0: 3, src1: 3, src2: 0, dst: 3
v_cndmask_b32 v[vgprGL2PrefetchAB+0-768], v[vgprGL2PrefetchAB+0-768], v[vgprGL2PrefetchABNext+0-768], vcc_lo
v_cndmask_b32 v[vgprGL2PrefetchAB+1-768], v[vgprGL2PrefetchAB+1-768], v[vgprGL2PrefetchABNext+1-768], vcc_lo
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+296-256:vgprValuC+296-256+7] matrix_b_reuse
s_set_vgpr_msb 207                                 // src0: 3, src1: 3, src2: 0, dst: 3
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+288-256:vgprValuC+288-256+7] matrix_a_reuse

s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+160:vgprValuC+160+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+32-512:vgprValuB_Y1+32-512+3], v[vgprLocalReadAddrB+0-512] offset:52224
ds_load_b128 v[vgprValuB_Y1+36-512:vgprValuB_Y1+36-512+3], v[vgprLocalReadAddrB+0-512] offset:52256
ds_load_b128 v[vgprValuB_Y1+40-512:vgprValuB_Y1+40-512+3], v[vgprLocalReadAddrB+0-512] offset:52288
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+168:vgprValuC+168+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+44-512:vgprValuB_Y1+44-512+3], v[vgprLocalReadAddrB+0-512] offset:52320
ds_load_b128 v[vgprValuB_Y1+48-512:vgprValuB_Y1+48-512+3], v[vgprLocalReadAddrB+0-512] offset:60928
ds_load_b128 v[vgprValuB_Y1+52-512:vgprValuB_Y1+52-512+3], v[vgprLocalReadAddrB+0-512] offset:60960
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+176:vgprValuC+176+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+56-512:vgprValuB_Y1+56-512+3], v[vgprLocalReadAddrB+0-512] offset:60992
ds_load_b128 v[vgprValuB_Y1+60-512:vgprValuB_Y1+60-512+3], v[vgprLocalReadAddrB+0-512] offset:61024
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+184:vgprValuC+184+7] matrix_a_reuse
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+248-256:vgprValuC+248-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+240-256:vgprValuC+240-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+232-256:vgprValuC+232-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+224-256:vgprValuC+224-256+7] matrix_a_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+0-768:vgprValuA_X10+0-768+3], v[vgprLocalReadAddrA+0-512] offset:34816
ds_load_b128 v[vgprValuA_X10+4-768:vgprValuA_X10+4-768+3], v[vgprLocalReadAddrA+0-512] offset:34848
ds_load_b128 v[vgprValuA_X10+8-768:vgprValuA_X10+8-768+3], v[vgprLocalReadAddrA+0-512] offset:34880
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+32:vgprValuC+32+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+12-768:vgprValuA_X10+12-768+3], v[vgprLocalReadAddrA+0-512] offset:34912
ds_load_b128 v[vgprValuA_X10+16-768:vgprValuA_X10+16-768+3], v[vgprLocalReadAddrA+0-512] offset:43520
ds_load_b128 v[vgprValuA_X10+20-768:vgprValuA_X10+20-768+3], v[vgprLocalReadAddrA+0-512] offset:43552
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+40:vgprValuC+40+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+24-768:vgprValuA_X10+24-768+3], v[vgprLocalReadAddrA+0-512] offset:43584
ds_load_b128 v[vgprValuA_X10+28-768:vgprValuA_X10+28-768+3], v[vgprLocalReadAddrA+0-512] offset:43616
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+48:vgprValuC+48+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+56:vgprValuC+56+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+120:vgprValuC+120+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+112:vgprValuC+112+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+104:vgprValuC+104+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+96:vgprValuC+96+7]

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/
s_branch label_LoopBeginL

/******************************************/
/* NoLoadLoop (NLL) - store-interleaved last K3       */
/******************************************/
label_NLL_store:
s_wait_dscnt 24                                    // MX + half A + half B
s_wait_alu depctr_va_vdst(0)
s_ttracedata_imm 0
s_barrier_wait -3
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

s_wait_tensorcnt 2
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_8
s_barrier_signal -3
label_Skip_Signal_8:

s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+0:vgprValuC+0+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+0-768:vgprValuA_X11+0-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
ds_load_b128 v[vgprValuA_X11+4-768:vgprValuA_X11+4-768+3], v[vgprLocalReadAddrA+0-512] offset:52384
ds_load_b128 v[vgprValuA_X11+8-768:vgprValuA_X11+8-768+3], v[vgprLocalReadAddrA+0-512] offset:52416
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+8:vgprValuC+8+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+12-768:vgprValuA_X11+12-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X11+16-768:vgprValuA_X11+16-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
ds_load_b128 v[vgprValuA_X11+20-768:vgprValuA_X11+20-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+16:vgprValuC+16+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X11+24-768:vgprValuA_X11+24-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
ds_load_b128 v[vgprValuA_X11+28-768:vgprValuA_X11+28-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+24:vgprValuC+24+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+88:vgprValuC+88+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+80:vgprValuC+80+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+72:vgprValuC+72+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+64:vgprValuC+64+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+0-512:vgprValuB_Y0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:0
ds_load_b128 v[vgprValuB_Y0+4-512:vgprValuB_Y0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:32
ds_load_b128 v[vgprValuB_Y0+8-512:vgprValuB_Y0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:64
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+128:vgprValuC+128+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+12-512:vgprValuB_Y0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:96
ds_load_b128 v[vgprValuB_Y0+16-512:vgprValuB_Y0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:8704
ds_load_b128 v[vgprValuB_Y0+20-512:vgprValuB_Y0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:8736
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+136:vgprValuC+136+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+24-512:vgprValuB_Y0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:8768
ds_load_b128 v[vgprValuB_Y0+28-512:vgprValuB_Y0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:8800
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+144:vgprValuC+144+7] matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA]
s_cmp_eq_i32 s[sgprIter], 3
s_cselect_b32 s90, 1, 0
s_cmp_eq_i32 s[sgprLoopCounterL], 6
s_cselect_b32 s91, 1, 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+152:vgprValuC+152+7] matrix_a_reuse
s_and_b32 s92, s90, s91
s_cmov_b32 s[sgprPrefetchIncAB], 0
s_set_vgpr_msb 204                                 // src0: 0, src1: 3, src2: 0, dst: 3
v_add_nc_u64 v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB+0-768:vgprGL2PrefetchAB+0-768+1]
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+216:vgprValuC+216+7] matrix_b_reuse
s_not_b32 s90, s90
s_and_b32 s92, s90, s91
v_cmp_eq_i32 vcc_lo, s92, 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+208:vgprValuC+208+7] matrix_b_reuse
s_set_vgpr_msb 207                                 // src0: 3, src1: 3, src2: 0, dst: 3
v_cndmask_b32 v[vgprGL2PrefetchAB+0-768], v[vgprGL2PrefetchAB+0-768], v[vgprGL2PrefetchABNext+0-768], vcc_lo
v_cndmask_b32 v[vgprGL2PrefetchAB+1-768], v[vgprGL2PrefetchAB+1-768], v[vgprGL2PrefetchABNext+1-768], vcc_lo
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+200:vgprValuC+200+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+192:vgprValuC+192+7] matrix_a_reuse
s_wait_alu depctr_vm_vsrc(6)
s_set_vgpr_msb 139                                 // src0: 3, src1: 2, src2: 0, dst: 2
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk

s_wait_dscnt 24                                    // another half B
s_wait_alu depctr_va_vdst(0)

s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+256-256:vgprValuC+256-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+32-512:vgprValuB_Y0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:17408
ds_load_b128 v[vgprValuB_Y0+36-512:vgprValuB_Y0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:17440
ds_load_b128 v[vgprValuB_Y0+40-512:vgprValuB_Y0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:17472
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+264-256:vgprValuC+264-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+44-512:vgprValuB_Y0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:17504
ds_load_b128 v[vgprValuB_Y0+48-512:vgprValuB_Y0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:26112
ds_load_b128 v[vgprValuB_Y0+52-512:vgprValuB_Y0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:26144
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+272-256:vgprValuC+272-256+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y0+56-512:vgprValuB_Y0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:26176
ds_load_b128 v[vgprValuB_Y0+60-512:vgprValuB_Y0+60-512+3], v[vgprLocalReadAddrB+0-512] offset:26208
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+280-256:vgprValuC+280-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+344-256:vgprValuC+344-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+336-256:vgprValuC+336-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+328-256:vgprValuC+328-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+320-256:vgprValuC+320-256+7] matrix_a_reuse
s_wait_alu depctr_va_vdst(8)
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+0-768:vgprValuA_X00+0-768+3], v[vgprLocalReadAddrA+0-512] offset:0
ds_load_b128 v[vgprValuA_X00+4-768:vgprValuA_X00+4-768+3], v[vgprLocalReadAddrA+0-512] offset:32
ds_load_b128 v[vgprValuA_X00+8-768:vgprValuA_X00+8-768+3], v[vgprLocalReadAddrA+0-512] offset:64
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+12-768:vgprValuA_X00+12-768+3], v[vgprLocalReadAddrA+0-512] offset:96
ds_load_b128 v[vgprValuA_X00+16-768:vgprValuA_X00+16-768+3], v[vgprLocalReadAddrA+0-512] offset:8704
ds_load_b128 v[vgprValuA_X00+20-768:vgprValuA_X00+20-768+3], v[vgprLocalReadAddrA+0-512] offset:8736
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X00+24-768:vgprValuA_X00+24-768+3], v[vgprLocalReadAddrA+0-512] offset:8768
ds_load_b128 v[vgprValuA_X00+28-768:vgprValuA_X00+28-768+3], v[vgprLocalReadAddrA+0-512] offset:8800
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X01+16+0+0-768:vgprValuA_X01+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X01+0+0+0-768:vgprValuA_X01+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X20+16+0+0-768:vgprValuA_X20+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7] matrix_b_reuse
s_set_vgpr_msb 2                                   // src0: 2, src1: 0, src2: 0, dst: 0
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X20+0+0+0-768:vgprValuA_X20+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7]

s_wait_dscnt 24                                    // another half A
s_wait_asynccnt 0
s_wait_alu depctr_va_vdst(0)
s_ttracedata_imm 0
s_barrier_wait -3
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

s_wait_tensorcnt 2
s_barrier_signal -1
s_barrier_wait -1

s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+0:vgprValuC+0+1], v[vgprValuC+0:vgprValuC+0+7], v[vgprSRSeed-768], s[sgprAlpha]
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+0-768:vgprValuA_X01+0-768+3], v[vgprLocalReadAddrA+0-512] offset:17408
ds_load_b128 v[vgprValuA_X01+4-768:vgprValuA_X01+4-768+3], v[vgprLocalReadAddrA+0-512] offset:17440
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+2:vgprValuC+2+1], v[vgprValuC+8:vgprValuC+8+7], v[vgprSRSeed-768], s[sgprAlpha]
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+8-768:vgprValuA_X01+8-768+3], v[vgprLocalReadAddrA+0-512] offset:17472
ds_load_b128 v[vgprValuA_X01+12-768:vgprValuA_X01+12-768+3], v[vgprLocalReadAddrA+0-512] offset:17504
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+4:vgprValuC+4+1], v[vgprValuC+16:vgprValuC+16+7], v[vgprSRSeed-768], s[sgprAlpha]
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+16-768:vgprValuA_X01+16-768+3], v[vgprLocalReadAddrA+0-512] offset:26112
ds_load_b128 v[vgprValuA_X01+20-768:vgprValuA_X01+20-768+3], v[vgprLocalReadAddrA+0-512] offset:26144
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+6:vgprValuC+6+1], v[vgprValuC+24:vgprValuC+24+7], v[vgprSRSeed-768], s[sgprAlpha]
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X01+24-768:vgprValuA_X01+24-768+3], v[vgprLocalReadAddrA+0-512] offset:26176
ds_load_b128 v[vgprValuA_X01+28-768:vgprValuA_X01+28-768+3], v[vgprLocalReadAddrA+0-512] offset:26208
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_permlane16_swap_b32 v[vgprValuC+0], v[vgprValuC+2]
v_permlane16_swap_b32 v[vgprValuC+1], v[vgprValuC+3]
v_permlane16_swap_b32 v[vgprValuC+4], v[vgprValuC+6]
v_permlane16_swap_b32 v[vgprValuC+5], v[vgprValuC+7]
s_mov_b32 s87, s[sgprSrdD+0]                       // PROBLEM2: save A0*B0 base lo (s87, dead in store tail)
s_mov_b32 s88, s[sgprSrdD+1]                       // PROBLEM2: save A0*B0 base hi (s88)
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+64:vgprValuC+64+1], v[vgprValuC+64:vgprValuC+64+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+66:vgprValuC+66+1], v[vgprValuC+72:vgprValuC+72+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+68:vgprValuC+68+1], v[vgprValuC+80:vgprValuC+80+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+70:vgprValuC+70+1], v[vgprValuC+88:vgprValuC+88+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+64], v[vgprValuC+66]
v_permlane16_swap_b32 v[vgprValuC+65], v[vgprValuC+67]
v_permlane16_swap_b32 v[vgprValuC+68], v[vgprValuC+70]
v_permlane16_swap_b32 v[vgprValuC+69], v[vgprValuC+71]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+128:vgprValuC+128+1], v[vgprValuC+128:vgprValuC+128+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+130:vgprValuC+130+1], v[vgprValuC+136:vgprValuC+136+7], v[vgprSRSeed-768], s[sgprAlpha]
s_wait_alu depctr_va_vdst(10)
s_set_vgpr_msb 3
s_clause 1
buffer_store_b128 v[vgprValuC+0:vgprValuC+0+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:0
buffer_store_b128 v[vgprValuC+4:vgprValuC+4+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:64
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s85
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+132:vgprValuC+132+1], v[vgprValuC+144:vgprValuC+144+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+134:vgprValuC+134+1], v[vgprValuC+152:vgprValuC+152+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+128], v[vgprValuC+130]
v_permlane16_swap_b32 v[vgprValuC+129], v[vgprValuC+131]
s_wait_alu depctr_va_vdst(6)
s_set_vgpr_msb 3
s_clause 1
buffer_store_b128 v[vgprValuC+64:vgprValuC+64+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:0
buffer_store_b128 v[vgprValuC+68:vgprValuC+68+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:64
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s85
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_permlane16_swap_b32 v[vgprValuC+132], v[vgprValuC+134]
v_permlane16_swap_b32 v[vgprValuC+133], v[vgprValuC+135]
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+192:vgprValuC+192+1], v[vgprValuC+192:vgprValuC+192+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+194:vgprValuC+194+1], v[vgprValuC+200:vgprValuC+200+7], v[vgprSRSeed-768], s[sgprAlpha]
s_wait_alu depctr_va_vdst(2)
s_set_vgpr_msb 3
s_clause 1
buffer_store_b128 v[vgprValuC+128:vgprValuC+128+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:0
buffer_store_b128 v[vgprValuC+132:vgprValuC+132+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:64
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s85
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+196:vgprValuC+196+1], v[vgprValuC+208:vgprValuC+208+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+198:vgprValuC+198+1], v[vgprValuC+216:vgprValuC+216+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+192], v[vgprValuC+194]
v_permlane16_swap_b32 v[vgprValuC+193], v[vgprValuC+195]
v_permlane16_swap_b32 v[vgprValuC+196], v[vgprValuC+198]
v_permlane16_swap_b32 v[vgprValuC+197], v[vgprValuC+199]
s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+256-256:vgprValuC+256-256+1], v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+258-256:vgprValuC+258-256+1], v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprSRSeed-768], s[sgprAlpha]
s_wait_alu depctr_va_vdst(2)
s_set_vgpr_msb 3
s_clause 1
buffer_store_b128 v[vgprValuC+192:vgprValuC+192+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:0
buffer_store_b128 v[vgprValuC+196:vgprValuC+196+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:64
s_mov_b32 s[sgprSrdD+0], s87                       // PROBLEM2: reset SrdD lo to A0*B0 base for A1*B0
s_mov_b32 s[sgprSrdD+1], s88                       // PROBLEM2: reset SrdD hi to A0*B0 base for A1*B0
s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+260-256:vgprValuC+260-256+1], v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+262-256:vgprValuC+262-256+1], v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+256-256], v[vgprValuC+258-256]
v_permlane16_swap_b32 v[vgprValuC+257-256], v[vgprValuC+259-256]
v_permlane16_swap_b32 v[vgprValuC+260-256], v[vgprValuC+262-256]
v_permlane16_swap_b32 v[vgprValuC+261-256], v[vgprValuC+263-256]
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+0-512:vgprValuB_Y1+0-512+3], v[vgprLocalReadAddrB+0-512] offset:34816
ds_load_b128 v[vgprValuB_Y1+4-512:vgprValuB_Y1+4-512+3], v[vgprLocalReadAddrB+0-512] offset:34848
ds_load_b128 v[vgprValuB_Y1+8-512:vgprValuB_Y1+8-512+3], v[vgprLocalReadAddrB+0-512] offset:34880
s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+320-256:vgprValuC+320-256+1], v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+322-256:vgprValuC+322-256+1], v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprSRSeed-768], s[sgprAlpha]
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+12-512:vgprValuB_Y1+12-512+3], v[vgprLocalReadAddrB+0-512] offset:34912
ds_load_b128 v[vgprValuB_Y1+16-512:vgprValuB_Y1+16-512+3], v[vgprLocalReadAddrB+0-512] offset:43520
ds_load_b128 v[vgprValuB_Y1+20-512:vgprValuB_Y1+20-512+3], v[vgprLocalReadAddrB+0-512] offset:43552
s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+324-256:vgprValuC+324-256+1], v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+326-256:vgprValuC+326-256+1], v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprSRSeed-768], s[sgprAlpha]
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+24-512:vgprValuB_Y1+24-512+3], v[vgprLocalReadAddrB+0-512] offset:43584
ds_load_b128 v[vgprValuB_Y1+28-512:vgprValuB_Y1+28-512+3], v[vgprLocalReadAddrB+0-512] offset:43616
s_set_vgpr_msb 65                                  // src0: 1, src1: 0, src2: 0, dst: 1
v_permlane16_swap_b32 v[vgprValuC+320-256], v[vgprValuC+322-256]
v_permlane16_swap_b32 v[vgprValuC+321-256], v[vgprValuC+323-256]
v_permlane16_swap_b32 v[vgprValuC+324-256], v[vgprValuC+326-256]
v_permlane16_swap_b32 v[vgprValuC+325-256], v[vgprValuC+327-256]
s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+384-256:vgprValuC+384-256+1], v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+386-256:vgprValuC+386-256+1], v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+388-256:vgprValuC+388-256+1], v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+390-256:vgprValuC+390-256+1], v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+384-256], v[vgprValuC+386-256]
v_permlane16_swap_b32 v[vgprValuC+385-256], v[vgprValuC+387-256]
v_permlane16_swap_b32 v[vgprValuC+388-256], v[vgprValuC+390-256]
v_permlane16_swap_b32 v[vgprValuC+389-256], v[vgprValuC+391-256]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+448-256:vgprValuC+448-256+1], v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+450-256:vgprValuC+450-256+1], v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+452-256:vgprValuC+452-256+1], v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+454-256:vgprValuC+454-256+1], v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+448-256], v[vgprValuC+450-256]
v_permlane16_swap_b32 v[vgprValuC+449-256], v[vgprValuC+451-256]
v_permlane16_swap_b32 v[vgprValuC+452-256], v[vgprValuC+454-256]
v_permlane16_swap_b32 v[vgprValuC+453-256], v[vgprValuC+455-256]
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+288-256:vgprValuC+288-256+7] matrix_b_reuse
s_set_vgpr_msb 4                                   // src0: 0, src1: 1, src2: 0, dst: 0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+256-256:vgprValuC+256-256+3] offset:0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+260-256:vgprValuC+260-256+3] offset:64
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+296-256:vgprValuC+296-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+304-256:vgprValuC+304-256+7] matrix_b_reuse
s_set_vgpr_msb 4                                   // src0: 0, src1: 1, src2: 0, dst: 0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+320-256:vgprValuC+320-256+3] offset:8704
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+324-256:vgprValuC+324-256+3] offset:8768
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+0+0+0-512:vgprValuB_Y3+0+0+0-512+15], v[vgprValuC+312-256:vgprValuC+312-256+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+376-256:vgprValuC+376-256+7] matrix_b_reuse
s_set_vgpr_msb 4                                   // src0: 0, src1: 1, src2: 0, dst: 0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+384-256:vgprValuC+384-256+3] offset:17408
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+388-256:vgprValuC+388-256+3] offset:17472
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+368-256:vgprValuC+368-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+360-256:vgprValuC+360-256+7] matrix_b_reuse
s_set_vgpr_msb 4                                   // src0: 0, src1: 1, src2: 0, dst: 0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+448-256:vgprValuC+448-256+3] offset:26112
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+452-256:vgprValuC+452-256+3] offset:26176
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+16+0+0-512:vgprValuB_Y3+16+0+0-512+15], v[vgprValuC+352-256:vgprValuC+352-256+7] matrix_a_reuse
s_set_vgpr_msb 95                                  // src0: 3, src1: 3, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+32+0+0-768:vgprValuB_Y3+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7] matrix_a_reuse
s_set_vgpr_msb 175                                 // src0: 3, src1: 3, src2: 2, dst: 2
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y3+48+0+0-768:vgprValuB_Y3+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7] matrix_a_reuse

s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+288-256:vgprValuC+288-256+1], v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+290-256:vgprValuC+290-256+1], v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+292-256:vgprValuC+292-256+1], v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+294-256:vgprValuC+294-256+1], v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+288-256], v[vgprValuC+290-256]
v_permlane16_swap_b32 v[vgprValuC+289-256], v[vgprValuC+291-256]
v_permlane16_swap_b32 v[vgprValuC+292-256], v[vgprValuC+294-256]
v_permlane16_swap_b32 v[vgprValuC+293-256], v[vgprValuC+295-256]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+352-256:vgprValuC+352-256+1], v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+354-256:vgprValuC+354-256+1], v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+356-256:vgprValuC+356-256+1], v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+358-256:vgprValuC+358-256+1], v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+352-256], v[vgprValuC+354-256]
v_permlane16_swap_b32 v[vgprValuC+353-256], v[vgprValuC+355-256]
v_permlane16_swap_b32 v[vgprValuC+356-256], v[vgprValuC+358-256]
v_permlane16_swap_b32 v[vgprValuC+357-256], v[vgprValuC+359-256]
s_wait_alu depctr_va_vdst(8)
s_set_vgpr_msb 4                                   // src0: 0, src1: 1, src2: 0, dst: 0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+288-256:vgprValuC+288-256+3] offset:128
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+292-256:vgprValuC+292-256+3] offset:192
s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+416-256:vgprValuC+416-256+1], v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+418-256:vgprValuC+418-256+1], v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+420-256:vgprValuC+420-256+1], v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+422-256:vgprValuC+422-256+1], v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprSRSeed-768], s[sgprAlpha]
s_wait_alu depctr_va_vdst(4)
s_set_vgpr_msb 4                                   // src0: 0, src1: 1, src2: 0, dst: 0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+352-256:vgprValuC+352-256+3] offset:8832
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+356-256:vgprValuC+356-256+3] offset:8896
s_set_vgpr_msb 65                                  // src0: 1, src1: 0, src2: 0, dst: 1
v_permlane16_swap_b32 v[vgprValuC+416-256], v[vgprValuC+418-256]
v_permlane16_swap_b32 v[vgprValuC+417-256], v[vgprValuC+419-256]
v_permlane16_swap_b32 v[vgprValuC+420-256], v[vgprValuC+422-256]
v_permlane16_swap_b32 v[vgprValuC+421-256], v[vgprValuC+423-256]
s_set_vgpr_msb 142                                 // src0: 2, src1: 3, src2: 0, dst: 2
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+480-512:vgprValuC+480-512+1], v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+482-512:vgprValuC+482-512+1], v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+484-512:vgprValuC+484-512+1], v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+486-512:vgprValuC+486-512+1], v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+480-512], v[vgprValuC+482-512]
v_permlane16_swap_b32 v[vgprValuC+481-512], v[vgprValuC+483-512]
v_permlane16_swap_b32 v[vgprValuC+484-512], v[vgprValuC+486-512]
v_permlane16_swap_b32 v[vgprValuC+485-512], v[vgprValuC+487-512]
s_wait_alu depctr_va_vdst(8)
s_set_vgpr_msb 4                                   // src0: 0, src1: 1, src2: 0, dst: 0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+416-256:vgprValuC+416-256+3] offset:17536
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+420-256:vgprValuC+420-256+3] offset:17600

s_wait_alu depctr_va_vdst(0)
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+32:vgprValuC+32+7] matrix_b_reuse
s_set_vgpr_msb 8                                   // src0: 0, src1: 2, src2: 0, dst: 0
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+480-512:vgprValuC+480-512+3] offset:26240
ds_store_b128 v[vgprAsyncAddr], v[vgprValuC+484-512:vgprValuC+484-512+3] offset:26304
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+40:vgprValuC+40+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+48:vgprValuC+48+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+0+0+0-512:vgprValuB_Y2+0+0+0-512+15], v[vgprValuC+56:vgprValuC+56+7] matrix_a_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+120:vgprValuC+120+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+112:vgprValuC+112+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+32-512:vgprValuB_Y1+32-512+3], v[vgprLocalReadAddrB+0-512] offset:52224
ds_load_b128 v[vgprValuB_Y1+36-512:vgprValuB_Y1+36-512+3], v[vgprLocalReadAddrB+0-512] offset:52256
ds_load_b128 v[vgprValuB_Y1+40-512:vgprValuB_Y1+40-512+3], v[vgprLocalReadAddrB+0-512] offset:52288
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+104:vgprValuC+104+7] matrix_b_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+44-512:vgprValuB_Y1+44-512+3], v[vgprLocalReadAddrB+0-512] offset:52320
ds_load_b128 v[vgprValuB_Y1+48-512:vgprValuB_Y1+48-512+3], v[vgprLocalReadAddrB+0-512] offset:60928
ds_load_b128 v[vgprValuB_Y1+52-512:vgprValuB_Y1+52-512+3], v[vgprLocalReadAddrB+0-512] offset:60960
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+16+0+0-512:vgprValuB_Y2+16+0+0-512+15], v[vgprValuC+96:vgprValuC+96+7] matrix_a_reuse
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_Y1+56-512:vgprValuB_Y1+56-512+3], v[vgprLocalReadAddrB+0-512] offset:60992
ds_load_b128 v[vgprValuB_Y1+60-512:vgprValuB_Y1+60-512+3], v[vgprLocalReadAddrB+0-512] offset:61024
s_set_vgpr_msb 11                                  // src0: 3, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+160:vgprValuC+160+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+168:vgprValuC+168+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+176:vgprValuC+176+7] matrix_b_reuse
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+32+0+0-512:vgprValuB_Y2+32+0+0-512+15], v[vgprValuC+184:vgprValuC+184+7] matrix_a_reuse
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X11+16+0+0-768:vgprValuA_X11+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+248-256:vgprValuC+248-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+0-768:vgprValuA_X10+0-768+3], v[vgprLocalReadAddrA+0-512] offset:34816
ds_load_b128 v[vgprValuA_X10+4-768:vgprValuA_X10+4-768+3], v[vgprLocalReadAddrA+0-512] offset:34848
ds_load_b128 v[vgprValuA_X10+8-768:vgprValuA_X10+8-768+3], v[vgprLocalReadAddrA+0-512] offset:34880
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X11+0+0+0-768:vgprValuA_X11+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+240-256:vgprValuC+240-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+12-768:vgprValuA_X10+12-768+3], v[vgprLocalReadAddrA+0-512] offset:34912
ds_load_b128 v[vgprValuA_X10+16-768:vgprValuA_X10+16-768+3], v[vgprLocalReadAddrA+0-512] offset:43520
ds_load_b128 v[vgprValuA_X10+20-768:vgprValuA_X10+20-768+3], v[vgprLocalReadAddrA+0-512] offset:43552
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X21+16+0+0-768:vgprValuA_X21+16+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+232-256:vgprValuC+232-256+7] matrix_b_reuse
s_set_vgpr_msb 194                                 // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuA_X10+24-768:vgprValuA_X10+24-768+3], v[vgprLocalReadAddrA+0-512] offset:43584
ds_load_b128 v[vgprValuA_X10+28-768:vgprValuA_X10+28-768+3], v[vgprLocalReadAddrA+0-512] offset:43616
s_set_vgpr_msb 91                                  // src0: 3, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x128_fp8_fp8 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X21+0+0+0-768:vgprValuA_X21+0+0+0-768+15], v[vgprValuB_Y2+48+0+0-512:vgprValuB_Y2+48+0+0-512+15], v[vgprValuC+224-256:vgprValuC+224-256+7]

s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v[vgprAsyncTmp+0], 15, v[vgprSerial-768] // lane&15
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshlrev_b32 v[vgprAsyncTmp+0], 4, v[vgprAsyncTmp+0] // M_lds = (lane&15)*16
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v[vgprAsyncTmp+1], 4, v[vgprSerial-768]
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v[vgprAsyncTmp+1], 1, v[vgprAsyncTmp+1]  // N_sel = (lane>>4)&1
s_mul_i32 s[sgprAsyncScratch], s[sgprWaveId], 32   // WaveId*32
v_add_nc_u32 v[vgprAsyncTmp+1], s[sgprAsyncScratch], v[vgprAsyncTmp+1] // R = WaveId*32 + N_sel
v_mul_lo_u32 v[vgprAsyncAddr+0], v[vgprAsyncTmp+1], s[sgprStrideD1J] // R*StrideD1J (N strided)
v_add_nc_u32 v[vgprAsyncAddr+0], v[vgprAsyncAddr+0], v[vgprAsyncTmp+0] // + M_lds (M raw)
s_mul_i32 s[sgprAsyncScratch], 256, s[sgprAsyncWG0]
v_add_nc_u32 v[vgprAsyncAddr+0], s[sgprAsyncScratch], v[vgprAsyncAddr+0] // + WG0*256 (M-tile, raw)
v_lshlrev_b32 v[vgprAsyncTmp+2], 8, v[vgprAsyncTmp+1] // R*256
v_lshlrev_b32 v[vgprAsyncLds+0], 4, v[vgprAsyncTmp+1] // R*16
v_add_nc_u32 v[vgprAsyncLds+0], v[vgprAsyncTmp+2], v[vgprAsyncLds+0] // R*(256+16)
v_add_nc_u32 v[vgprAsyncLds+0], v[vgprAsyncLds+0], v[vgprAsyncTmp+0] // + M_lds
v_add_nc_u32 v[vgprAsyncLds+0], v[vgprAsyncLds+0], 287744 // + staging base
v_add_nc_u32 v[vgprAsyncLds+1], 4352, v[vgprAsyncLds+0] // group1 LDS = group0 + 16 N rows
s_mul_i32 s[sgprAsyncScratch], s[sgprStrideD1J], 2
s_sub_u32 s[sgprAsyncScratch], s[sgprAsyncScratch], 544 // ptr step = 2*StrideD1J - 2*(256+16)
v_add_nc_u32 v[vgprAsyncAddr+1], v[vgprAsyncAddr+0], s[sgprAsyncScratch]
v_add_nc_u32 v[vgprAsyncAddr+2], v[vgprAsyncAddr+1], s[sgprAsyncScratch]
v_add_nc_u32 v[vgprAsyncAddr+3], v[vgprAsyncAddr+2], s[sgprAsyncScratch]
v_add_nc_u32 v[vgprAsyncAddr+4], v[vgprAsyncAddr+3], s[sgprAsyncScratch]
v_add_nc_u32 v[vgprAsyncAddr+5], v[vgprAsyncAddr+4], s[sgprAsyncScratch]
v_add_nc_u32 v[vgprAsyncAddr+6], v[vgprAsyncAddr+5], s[sgprAsyncScratch]
v_add_nc_u32 v[vgprAsyncAddr+7], v[vgprAsyncAddr+6], s[sgprAsyncScratch]
s_mul_i32 s[sgprAsyncScratch], s[sgprStrideD1J], 128 // lo(128*St): s87 = B (saved before group A)
s_mul_hi_u32 s89, s[sgprStrideD1J], 128            // hi(128*St); s89 dead in store tail
s_add_u32 s[sgprSrdD+0], s87, s[sgprAsyncScratch]  // SrdD = B + 128*StrideD1J (Nhi base)
s_addc_u32 s[sgprSrdD+1], s88, s89

s_wait_dscnt 20                                    // wait B0 staging ds_store done before drain
s_wait_alu depctr_va_vdst(0)
s_barrier_signal -1
s_barrier_wait -1

global_store_async_from_lds_b128 v[vgprAsyncAddr+0], v[vgprAsyncLds+0], s[sgprSrdD:sgprSrdD+1] offset:0 // grp0 N+0/+1 x M0..255 (lanes 0-15 / 16-31)
global_store_async_from_lds_b128 v[vgprAsyncAddr+1], v[vgprAsyncLds+0], s[sgprSrdD:sgprSrdD+1] offset:544 // grp0 N+2/+3 x M0..255 (lanes 0-15 / 16-31)
global_store_async_from_lds_b128 v[vgprAsyncAddr+2], v[vgprAsyncLds+0], s[sgprSrdD:sgprSrdD+1] offset:1088 // grp0 N+4/+5 x M0..255 (lanes 0-15 / 16-31)
global_store_async_from_lds_b128 v[vgprAsyncAddr+3], v[vgprAsyncLds+0], s[sgprSrdD:sgprSrdD+1] offset:1632 // grp0 N+6/+7 x M0..255 (lanes 0-15 / 16-31)
global_store_async_from_lds_b128 v[vgprAsyncAddr+4], v[vgprAsyncLds+0], s[sgprSrdD:sgprSrdD+1] offset:2176 // grp0 N+8/+9 x M0..255 (lanes 0-15 / 16-31)
global_store_async_from_lds_b128 v[vgprAsyncAddr+5], v[vgprAsyncLds+0], s[sgprSrdD:sgprSrdD+1] offset:2720 // grp0 N+10/+11 x M0..255 (lanes 0-15 / 16-31)
global_store_async_from_lds_b128 v[vgprAsyncAddr+6], v[vgprAsyncLds+0], s[sgprSrdD:sgprSrdD+1] offset:3264 // grp0 N+12/+13 x M0..255 (lanes 0-15 / 16-31)
global_store_async_from_lds_b128 v[vgprAsyncAddr+7], v[vgprAsyncLds+0], s[sgprSrdD:sgprSrdD+1] offset:3808 // grp0 N+14/+15 x M0..255 (lanes 0-15 / 16-31)
s_mov_b32 s[sgprSrdD+0], s87                       // reset SrdD = B (s87) for group-B (A1) buffer stores
s_mov_b32 s[sgprSrdD+1], s88

s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+32:vgprValuC+32+1], v[vgprValuC+32:vgprValuC+32+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+34:vgprValuC+34+1], v[vgprValuC+40:vgprValuC+40+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+36:vgprValuC+36+1], v[vgprValuC+48:vgprValuC+48+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+38:vgprValuC+38+1], v[vgprValuC+56:vgprValuC+56+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+32], v[vgprValuC+34]
v_permlane16_swap_b32 v[vgprValuC+33], v[vgprValuC+35]
v_permlane16_swap_b32 v[vgprValuC+36], v[vgprValuC+38]
v_permlane16_swap_b32 v[vgprValuC+37], v[vgprValuC+39]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+96:vgprValuC+96+1], v[vgprValuC+96:vgprValuC+96+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+98:vgprValuC+98+1], v[vgprValuC+104:vgprValuC+104+7], v[vgprSRSeed-768], s[sgprAlpha]
s_wait_alu depctr_va_vdst(2)
s_set_vgpr_msb 3
s_clause 1
buffer_store_b128 v[vgprValuC+32:vgprValuC+32+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:128
buffer_store_b128 v[vgprValuC+36:vgprValuC+36+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:192
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s85
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+100:vgprValuC+100+1], v[vgprValuC+112:vgprValuC+112+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+102:vgprValuC+102+1], v[vgprValuC+120:vgprValuC+120+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+96], v[vgprValuC+98]
v_permlane16_swap_b32 v[vgprValuC+97], v[vgprValuC+99]
v_permlane16_swap_b32 v[vgprValuC+100], v[vgprValuC+102]
v_permlane16_swap_b32 v[vgprValuC+101], v[vgprValuC+103]

s_wait_alu depctr_va_vdst(0)
s_set_vgpr_msb 3
s_clause 1
buffer_store_b128 v[vgprValuC+96:vgprValuC+96+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:128
buffer_store_b128 v[vgprValuC+100:vgprValuC+100+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:192
s_mul_i32 s[sgprAsyncScratch], s[sgprStrideD1J], 144 // lo(144*St) = Nhi group1 base (128+16)
s_mul_hi_u32 s89, s[sgprStrideD1J], 144            // hi(144*St)
s_add_u32 s[sgprSrdD+0], s87, s[sgprAsyncScratch]  // SrdD = B + 144*StrideD1J
s_addc_u32 s[sgprSrdD+1], s88, s89

s_wait_dscnt 16                                    // wait ALL staging ds_store done before drain
s_barrier_signal -1
s_barrier_wait -1
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
global_store_async_from_lds_b128 v[vgprAsyncAddr+0], v[vgprAsyncLds+1], s[sgprSrdD:sgprSrdD+1] offset:0 // grp1 N+16/+17 x M0..255
global_store_async_from_lds_b128 v[vgprAsyncAddr+1], v[vgprAsyncLds+1], s[sgprSrdD:sgprSrdD+1] offset:544 // grp1 N+18/+19 x M0..255
global_store_async_from_lds_b128 v[vgprAsyncAddr+2], v[vgprAsyncLds+1], s[sgprSrdD:sgprSrdD+1] offset:1088 // grp1 N+20/+21 x M0..255
global_store_async_from_lds_b128 v[vgprAsyncAddr+3], v[vgprAsyncLds+1], s[sgprSrdD:sgprSrdD+1] offset:1632 // grp1 N+22/+23 x M0..255
global_store_async_from_lds_b128 v[vgprAsyncAddr+4], v[vgprAsyncLds+1], s[sgprSrdD:sgprSrdD+1] offset:2176 // grp1 N+24/+25 x M0..255
global_store_async_from_lds_b128 v[vgprAsyncAddr+5], v[vgprAsyncLds+1], s[sgprSrdD:sgprSrdD+1] offset:2720 // grp1 N+26/+27 x M0..255
global_store_async_from_lds_b128 v[vgprAsyncAddr+6], v[vgprAsyncLds+1], s[sgprSrdD:sgprSrdD+1] offset:3264 // grp1 N+28/+29 x M0..255
global_store_async_from_lds_b128 v[vgprAsyncAddr+7], v[vgprAsyncLds+1], s[sgprSrdD:sgprSrdD+1] offset:3808 // grp1 N+30/+31 x M0..255
s_mov_b32 s[sgprSrdD+0], s87                       // restore SrdD = B + 2*s85 (group-B sb2/sb3)
s_mov_b32 s[sgprSrdD+1], s88
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+160:vgprValuC+160+1], v[vgprValuC+160:vgprValuC+160+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+162:vgprValuC+162+1], v[vgprValuC+168:vgprValuC+168+7], v[vgprSRSeed-768], s[sgprAlpha]
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s85
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+164:vgprValuC+164+1], v[vgprValuC+176:vgprValuC+176+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+166:vgprValuC+166+1], v[vgprValuC+184:vgprValuC+184+7], v[vgprSRSeed-768], s[sgprAlpha]
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s85
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
v_permlane16_swap_b32 v[vgprValuC+160], v[vgprValuC+162]
v_permlane16_swap_b32 v[vgprValuC+161], v[vgprValuC+163]
v_permlane16_swap_b32 v[vgprValuC+164], v[vgprValuC+166]
v_permlane16_swap_b32 v[vgprValuC+165], v[vgprValuC+167]
s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+224-256:vgprValuC+224-256+1], v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+226-256:vgprValuC+226-256+1], v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprSRSeed-768], s[sgprAlpha]
s_wait_alu depctr_va_vdst(2)
s_set_vgpr_msb 3
s_clause 1
buffer_store_b128 v[vgprValuC+160:vgprValuC+160+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:128
buffer_store_b128 v[vgprValuC+164:vgprValuC+164+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:192
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s85
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
s_set_vgpr_msb 77                                  // src0: 1, src1: 3, src2: 0, dst: 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+228-256:vgprValuC+228-256+1], v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+230-256:vgprValuC+230-256+1], v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprSRSeed-768], s[sgprAlpha]
v_permlane16_swap_b32 v[vgprValuC+224-256], v[vgprValuC+226-256]
v_permlane16_swap_b32 v[vgprValuC+225-256], v[vgprValuC+227-256]
v_permlane16_swap_b32 v[vgprValuC+228-256], v[vgprValuC+230-256]
v_permlane16_swap_b32 v[vgprValuC+229-256], v[vgprValuC+231-256]
s_wait_alu depctr_va_vdst(0)
s_set_vgpr_msb 67
s_clause 1
buffer_store_b128 v[vgprValuC+224-256:vgprValuC+224-256+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:128
buffer_store_b128 v[vgprValuC+228-256:vgprValuC+228-256+3], v[vgprStoreAddr-768], s[sgprSrdD:sgprSrdD+3], null offen offset:192

label_LoopEndL:
s_and_b32 s5, s[sgprGSU], 0x3fff
s_delay_alu instid0(SALU_CYCLE_1)
label_PrefetchGlobalLastIterEnd:
label_Summation_End_S4FDBQ587JJL6NOU:
.set sgprWGM, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressMXSA, UNDEF
.set sgprAddressB, UNDEF
.set sgprAddressMXSB, UNDEF
.set sgprStridesMXSA, UNDEF
.set sgprStridesMXSB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprGlobalReadIncsMXSA, UNDEF
.set sgprGlobalReadIncsMXSB, UNDEF
label_GW_B0_E1_N_1:
label_GW_Beta_1:
label_GW_B0_E1_M_1:
label_GW_End_1:
label_KernelEnd:
s_cmp_ge_u32 s[sgprIter], 4
s_cbranch_scc1 label_ENDPGM
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizeL], 8
s_branch label_Persist_Start_1
label_ENDPGM:
s_endpgm
label_ASM_End:
