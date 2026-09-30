
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
  .amdhsa_group_segment_fixed_size 287744 // lds bytes
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
  .amdhsa_inst_pref_size 240
.end_amdhsa_kernel
.text
/* Num VGPR   =1024 */
/* Num AccVGPR=0 */
/* Num SGPR   =104 */

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
    .group_segment_fixed_size:   287744
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

.set sgprIter, 97          // 1
.set vgprSerial, 1023

s_mov_b32 m0, 0x46400                              // LDS clamp at 287744 bytes
s_set_vgpr_msb 192
v_mov_b32 v[vgprSerial-768], v0                    // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

s_mov_b32 s[sgprIter], 3                           // non-persistent: pin to the last persist index so each WG runs exactly one tile
s_nop 0
label_ASM_Start:  /// Main body of the asm kernel
s_nop 0

.macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    s_set_vgpr_msb 0
    v_mul_hi_u32 v[\vgprDstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\vgprDstIdx+0], \dividend, \magicA
    v_add_nc_u32 v[\vgprDstIdx+0], v[\vgprDstIdx+0], v[\vgprDstIdx+1]
    v_lshrrev_b32 v[\vgprDstIdx+0], \magicShift, v[\vgprDstIdx+0]
.endm

/******************************************/
/* VGPR Assignments for MX                */
/******************************************/
.set vgprMXSBase, 0

/******************************************/
/* VGPR Macro Assignments for MX          */
/******************************************/
.set vgprValuMXSA_X0_I0_BASE, vgprMXSBase+0
.set vgprValuMXSB_X0_I0_BASE, vgprMXSBase+16
.set vgprValuMXSA_X0_I0, vgprValuMXSA_X0_I0_BASE+0
.set vgprValuMXSA_X1_I0, vgprValuMXSA_X0_I0_BASE+8
.set vgprValuMXSB_X0_I0, vgprValuMXSB_X0_I0_BASE+0
.set vgprValuMXSB_X1_I0, vgprValuMXSB_X0_I0_BASE+8
.set vgprG2LMXSA, vgprG2LMXSA_BASE+0
.set vgprG2LMXSB, vgprG2LMXSB_BASE+0

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [32-544), serializedStore enabled */
.set vgprValuC, 32
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 550
.set vgprGlobalReadOffsetA, 544
.set vgprGlobalReadOffsetMXSA, 544
.set vgprGlobalReadOffsetB, 544
.set vgprGlobalReadOffsetMXSB, 544
.set vgprLocalReadAddrA, 546
.set vgprLocalReadAddrMXSA, 544
.set vgprLocalReadAddrB, 548
.set vgprLocalReadAddrMXSB, 545
.set vgprLocalReadSwapAddrA, 998
.set vgprLocalReadSwapAddrB, 1000
.set vgprLocalReadSwapAddrMXSA, 999
.set vgprLocalReadSwapAddrMXSB, 1001
.set vgprGL2PrefetchAB, 1004
.set vgprGL2PrefetchMX, 1006
.set vgprGL2PrefetchABNoWG, 1008
.set vgprGL2PrefetchMXNoWG, 1010
.set vgprGL2PrefetchABNext, 1012
.set vgprGL2PrefetchMXNext, 1014

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0 // 550
.set vgprValuB_X0_I0_BASE, vgprBase+256 // 806
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+128 // 678, vgprValuA_X1_I0 + 90 in msb 3
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_Y0, vgprValuB_X0_I0+0
.set vgprValuB_Y1, vgprValuB_X0_I0+64
.set vgprValuB_Y2, vgprValuB_X0_I0+128
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB, vgprG2LB_BASE+0

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprMulticastMask, 5
.set sgprArgType, 6
.set sgprGSUSumIdx, 8
.set sgprGSULog2BpeC, 7
.set sgprGSULog2BpeD, 10
.set sgprStaggerU, 11
.set sgprWGM, 12
.set sgprLoopCounterL, 13
.set sgprOrigLoopCounter, 14
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
.set sgprTDMAddrABNoWG, 98
.set sgprTDMAddrMXABNoWG, 100
.set sgprTDMAddrABNext, 76
.set sgprTDMAddrMXABNext, 78
.set sgprWorkGroup0Next, 80
.set sgprWorkGroup1Next, 81
.set sgprExitLoopIdx, 82
.set sgprWaveId, 83
.set sgprPrefetchIncMX, 102
.set sgprPrefetchIncAB, 104

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
/* hex: 0x0                               */
/* num_records_upper (6b): 0              */
/* reserved (6b): 0                       */
/* stride (14b): 0                        */
/* stride_scale (2b): 0                   */
/* swizzle_enable (1b): 0                 */
/* oob_select (1b): 0                     */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x0

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req, vgprOffsetL:req, vgprOffset0I:req, vgprTmp:req
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
; s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
; s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

.set sgprUnitScale, 42
s_mov_b32 s[sgprUnitScale], 0x7f7f7f7f            // E8M0 1.0 x4: scaled WMMA acts as plain FP8
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

s_set_vgpr_msb 49155
v_readfirstlane_b32 s92, v[vgprSerial-768]         // first tId
s_lshr_b32 s[sgprWaveId], s92, 5                   // wId=fTid // wavelen

/* local read addresses: tile assignments a/b */
/* lr0I */
s_set_vgpr_msb 780
v_and_b32 v1, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 3072
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 8, v0                            // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v0, v1, 4, v0                       // 5. K offset: lrKOffset = kIdx * mStride(16); 6. offset in wave: lrOffset = bnOffset + lrKOffset
s_set_vgpr_msb 12
v_lshrrev_b32 v4, 5, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
s_set_vgpr_msb 3072
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 12, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096); 7. final local read offset: flrOffset = lrOffset + WOffset
s_set_vgpr_msb 12
v_and_b32 v2, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 3072
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 2, v1                            // 1. N offset: nOffset = nIdx * nStride
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
s_set_vgpr_msb 12
v_lshrrev_b32 v3, 5, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
s_set_vgpr_msb 3072
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 6, v1                       // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride; 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
s_set_vgpr_msb 12
v_and_b32 v3, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 3072
v_and_b32 v2, 15, v3                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v2, 8, v2                            // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 4, v3                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v2, v3, 4, v2                       // 5. K offset: lrKOffset = kIdx * mStride(16); 6. offset in wave: lrOffset = bnOffset + lrKOffset
s_set_vgpr_msb 12
v_lshrrev_b32 v6, 6, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
s_set_vgpr_msb 3072
v_and_b32 v6, 1, v6                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v2, v6, 12, v2                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096); 7. final local read offset: flrOffset = lrOffset + WOffset
s_set_vgpr_msb 12
v_and_b32 v4, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 3072
v_and_b32 v3, 15, v4                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v3, 2, v3                            // 1. N offset: nOffset = nIdx * nStride(8)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
s_set_vgpr_msb 12
v_lshrrev_b32 v5, 6, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
s_set_vgpr_msb 3072
v_and_b32 v5, 1, v5                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v3, v5, 6, v3                       // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(128); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
s_set_vgpr_msb 12
v_lshrrev_b32 v4, 5, v[vgprSerial-768]             // 4 = Serial / 32
s_set_vgpr_msb 3072
v_lshrrev_b32 v4, 2, v4                            // LSU offset: Get LSU wave_id
s_mov_b32 s19, 256                                 // LSU offset: stride = lsuStride(256) when umlds==True
v_mul_lo_u32 v4, s19, v4                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
s_set_vgpr_msb 128
v_add_nc_u32 v[vgprLocalReadAddrA-512], v4, v0     // Final Offset: offset = (lro0+lsuoffset)*bpeDS
                                                   //  (bpe is 1, do nothing)
s_set_vgpr_msb 32776
v_lshrrev_b32 v5, 8, v[vgprLocalReadAddrA-512]     // Final Offset: padding 16 per block 256
s_set_vgpr_msb 2208
v_lshl_add_u32 v[vgprLocalReadAddrA-512], v5, 4, v[vgprLocalReadAddrA-512] // Final Offset: padding 16 per block 256

s_set_vgpr_msb 40972
v_lshrrev_b32 v0, 5, v[vgprSerial-768]             // 0 = Serial / 32
s_set_vgpr_msb 3072
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s19, 8                                   // LSU offset: stride = lsuStride(8) when umlds==True
v_mul_lo_u32 v0, s19, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
s_set_vgpr_msb 128
                                                   //  (bpe is 1, do nothing)

s_set_vgpr_msb 32780
v_lshrrev_b32 v0, 5, v[vgprSerial-768]             // 0 = Serial / 32
s_set_vgpr_msb 3072
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(8) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s19, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
s_set_vgpr_msb 128
                                                   //  (bpe is 1, do nothing)

/* local read addresses: final offsets b */
s_set_vgpr_msb 32780
v_lshrrev_b32 v0, 5, v[vgprSerial-768]             // 0 = Serial / 32
s_set_vgpr_msb 3072
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s19, 256                                 // LSU offset: stride = lsuStride(256) when umlds==True
v_mul_lo_u32 v0, s19, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
s_set_vgpr_msb 128
v_add_nc_u32 v[vgprLocalReadAddrB-512], v0, v2     // Final Offset: offset = (lro1+lsuoffset)*bpeDS
                                                   //  (bpe is 1, do nothing)
s_set_vgpr_msb 32776
v_lshrrev_b32 v1, 8, v[vgprLocalReadAddrB-512]     // Final Offset: padding 16 per block 256
s_set_vgpr_msb 2208
v_lshl_add_u32 v[vgprLocalReadAddrB-512], v1, 4, v[vgprLocalReadAddrB-512] // Final Offset: padding 16 per block 256

/* local read addresses: declare addresses a */
s_set_vgpr_msb 41096
v_add_nc_u32 v[vgprLocalReadAddrA+1-512], 65536, v[vgprLocalReadAddrA+0-512] // Final vgprLocalReadAddrA+1 Offset Plus 64K



/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0-512], vcc_lo, 0x12200, v[vgprLocalReadAddrB+0-512] //  += LdsOffsetB (lower)
s_delay_alu instid0(VALU_DEP_1)
v_add_nc_u32 v[vgprLocalReadAddrB+1-512], 65536, v[vgprLocalReadAddrB+0-512] // Final vgprLocalReadAddrB+1 Offset Plus 64K
s_set_vgpr_msb 35016
v_add_nc_u32 v[vgprLocalReadSwapAddrA-768], 143872, v[vgprLocalReadAddrA-512] // Calculate starting lds addr of second buffer
s_set_vgpr_msb 51403
v_xor_b32 v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // xor both lds buffer offsets to enable swapping
s_set_vgpr_msb 52168
s_set_vgpr_msb 51403
s_set_vgpr_msb 52168
v_add_nc_u32 v[vgprLocalReadSwapAddrB-768], 143872, v[vgprLocalReadAddrB-512] // Calculate starting lds addr of second buffer
s_set_vgpr_msb 51403
v_xor_b32 v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // xor both lds buffer offsets to enable swapping
s_set_vgpr_msb 52168
s_set_vgpr_msb 51403

/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

/* local write addresses: first offset b */
s_set_vgpr_msb 51968
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
.set sgprStaggerUIter, 19
.set sgprWrapUA, 76
.set sgprWrapUB, 78
.set sgprWrapUMXSA, 80
.set sgprWrapUMXSB, 82
.set sgprGlobalReadIncsA, 51
.set sgprGlobalReadIncsB, 84
.set sgprGlobalReadIncsMXSA, 85
.set sgprGlobalReadIncsMXSB, 86

/* Short circuit condition if Alpha == 0, then sumDims=0 */
; v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha] == 0.0f ?
; s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
; s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
; label_AlphaNonZero:
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
s_set_vgpr_msb 3
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

/***** Tony Modify 2 ****/
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
s_set_vgpr_msb 771
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
s_set_vgpr_msb 963
v_and_b32 v[vgprGL2PrefetchABNoWG-768], v[vgprSerial-768], 31    // Restrict thread index to 32
v_mul_lo_u32 v[vgprGL2PrefetchABNoWG-768], v[vgprGL2PrefetchABNoWG-768], 4                   // Multiply by 4
v_mul_lo_u32 v[vgprGL2PrefetchABNoWG-768], v[vgprGL2PrefetchABNoWG-768], s[sgprStrideA0I]     // Jump to correct row
s_set_vgpr_msb 50112
v_mov_b32 v[vgprGL2PrefetchABNoWG-768+1], 0                                        // Set upper 32-bit to 0
s_set_vgpr_msb 49356
v_add_nc_u64 v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1], s[92:93], v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1]
s_add_u32 s92, s88, 512                           // Offset by 2 buffers (DU*Bpe*2(preloads))
s_mov_b32 s93, 0
v_add_nc_u64 v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1], s[92:93], v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1]
s_mul_i32 s92, s[sgprStrideA0I], 256               // stride * MT(256) * bpe(1.0)
s_mul_i32 s92, s92, s[sgprWorkGroup0Next]           // *= wgId)
v_add_nc_u64 v[vgprGL2PrefetchABNext-768+0:vgprGL2PrefetchABNext-768+1], s[92:93], v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1]
s_mov_b32 s[sgprPrefetchIncAB], 256                    // Data pointer increment
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
s_set_vgpr_msb 52227
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
s_set_vgpr_msb 963
v_and_b32 v[vgprGL2PrefetchABNoWG-768], v[vgprSerial-768], 31    // Restrict thread index to 32
v_mul_lo_u32 v[vgprGL2PrefetchABNoWG-768], v[vgprGL2PrefetchABNoWG-768], 4                   // Multiply by 4
v_mul_lo_u32 v[vgprGL2PrefetchABNoWG-768], v[vgprGL2PrefetchABNoWG-768], s[sgprStrideB1J]     // Jump to correct row
s_set_vgpr_msb 50112
v_mov_b32 v[vgprGL2PrefetchABNoWG-768+1], 0                                        // Set upper 32-bit to 0
s_set_vgpr_msb 49356
v_add_nc_u64 v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1], s[92:93], v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1]
s_add_u32 s92, s88, 512                           // Offset by 2 buffers (DU*Bpe*2(preloads))
s_mov_b32 s93, 0
v_add_nc_u64 v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1], s[92:93], v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1]
s_mul_i32 s92, s[sgprStrideB1J], 256               // stride * MT(256) * bpe(1.0)
s_mul_i32 s92, s92, s[sgprWorkGroup1Next]           // *= wgId)
v_add_nc_u64 v[vgprGL2PrefetchABNext-768+0:vgprGL2PrefetchABNext-768+1], s[92:93], v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1]
s_mov_b32 s[sgprPrefetchIncAB], 256                    // Data pointer increment
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
s_set_vgpr_msb 52227
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
s_mul_i32 s91, s[sgprWorkGroup0Next], 1024                         // * MT(256) * bpe(1.0)
// gl2 prefetch addr calc start
s_and_b32 s92, s[sgprWorkGroup1], 3                // Update WG Idx to 4 clusters
s_mul_i32 s92, s92, 256                            // WG offset *= 256(rows)
s_add_u32 s92, s90, s92                         
s_set_vgpr_msb 960
s_mov_b32 s93, 0
s_set_vgpr_msb 49356
s_mul_i32 s92, s[sgprWorkGroup0Next], 1024             // stride * MT(256) * bpe(1)
// gl2 prefetch addr calc end
s_add_u32 s88, s88, s90                            // += woffset
s_branch label_TDMGlobalOffsetMXSAMXSBEnd
label_TDMGlobalOffsetMXSB:
// TDM wave separated calc start addr of MXSB 
s_mov_b64 s[88:89], 0
s_mul_i32 s88, s[sgprWorkGroup1], 1024            // stride * MT(256) * bpe(1)
s_set_vgpr_msb 52227
v_readfirstlane_b32 s90, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s90, s90, 6                             // wCompId = fTid // wavelen(32) // numComp(2)
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s90, s90, s[sgprSizeJ]                   // woffset = wCompId * SizeJ * K0(4) // numComp(2) * bpe(1)
s_delay_alu instid0(SALU_CYCLE_1)
s_mul_i32 s90, s90, 4                              // woffset = wCompId * SizeJ * K0(4) // numComp(2) * bpe(1)
s_delay_alu instid0(SALU_CYCLE_1)
// persist: offset w/o WG
// next WG addr
s_mul_i32 s91, s[sgprWorkGroup1Next], 1024                         // * MT(256) * bpe(1.0)
// GL2 prefetch addr calc start
s_and_b32 s92, s[sgprWorkGroup0], 3                // Update WG Idx to 4 clusters
s_mul_i32 s92, s92, 256                            // WG offset *= 256(rows)
s_add_u32 s92, s90, s92                         
s_set_vgpr_msb 960
s_mov_b32 s93, 0
s_set_vgpr_msb 49356
s_mul_i32 s92, s[sgprWorkGroup1Next], 1024             // stride * MT(256) * bpe(1)
// gl2 prefetch addr calc end
s_add_u32 s88, s88, s90                            // += woffset
label_TDMGlobalOffsetMXSAMXSBEnd:
label_TDMInitA:
s_set_vgpr_msb 52227
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
s_lshr_b32 s87, s87, 1                            // / (2tdm)
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
s_set_vgpr_msb 771
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
s_set_vgpr_msb 771
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
s_lshr_b32 s87, s[sgprSizeL], 0x7 // SizeL // 32 // 4
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s[sgprSizeL], 0x7 // SizeL // 32 // 4
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
s_set_vgpr_msb 771
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
s_lshr_b32 s87, s[sgprSizeL], 0x7
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s87, s87, 0x10
s_delay_alu instid0(SALU_CYCLE_1)
s_delay_alu instid0(SALU_CYCLE_1)
s_lshr_b32 s87, s[sgprSizeL], 0x7
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
s_set_vgpr_msb 771
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
// cluster sync
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_0
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_0:

/* prefetch: global -> local */
; s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

; /* after InitC, skip to end of prefetch last iter if numIter==0 */
; s_cbranch_scc0 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc1
; s_getpc_b64 s[88:89]                               // addr of next instr
; s_add_i32 s90, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
; s_add_u32 s88, s88, s90                            // add target branch offset
; s_addc_u32 s89, s89, 0                             // add high and carry
; s_setpc_b64 s[88:89]                               // branch to label_PrefetchGlobalLastIterEnd
; label_NoBranch_T8JHFHKM7BO5OHXW:
// set TDM addr swap sgpr for A/B
s_add_i32 s[sgprTDMAddrSwapA], s[sgprtdmAGroup0+1], 143872 // high buffer
s_delay_alu instid0(SALU_CYCLE_1)
s_xor_b32 s[sgprTDMAddrSwapA], s[sgprTDMAddrSwapA], s[sgprtdmAGroup0+1]
s_delay_alu instid0(SALU_CYCLE_1)
// set 2TDM adjustment
s_mov_b32 s[sgprTDMSplitA], 34816 // (256+16)*128 = 34816 bytes
s_lshl_b32 s[sgprTDMGlobalSplitA], s[sgprSizeL], 0x7 // K * 128
s_barrier_wait -3
s_barrier_signal -1
s_barrier_wait -1
// PGR prefetch 1
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

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

// PGR prefetch 2
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM increment
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

// GL2 prefetch 1
s_set_vgpr_msb 771 // global vaddr = src0
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null  scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch

// GL2 prefetch pointers increment
s_set_vgpr_msb 972
v_add_nc_u64 v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1]
// GL2 prefetch 2
s_set_vgpr_msb 52227 // global vaddr = src0
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null  scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch

// GL2 prefetch pointers increment
s_set_vgpr_msb 972
v_add_nc_u64 v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1]

// PLR
// set vgprValuB idx
.set vgprValuB_G0, vgprValuB_Y2+0
.set vgprValuB_G1, vgprValuB_Y0+0
.set vgprValuB_G2, vgprValuB_Y1+0
s_wait_tensorcnt 2                                 // wait for prefetch 1
s_ttracedata_imm 0
s_barrier_signal -1 
s_barrier_wait -1
s_set_vgpr_msb 52226

/* local read prefetch b */
s_set_vgpr_msb 706
ds_load_b128 v[vgprValuB_G1+0-768:vgprValuB_G1+0-768+3], v[vgprLocalReadAddrB+0-512] offset:0 
ds_load_b128 v[vgprValuB_G1+4-768:vgprValuB_G1+4-768+3], v[vgprLocalReadAddrB+0-512] offset:32
ds_load_b128 v[vgprValuB_G1+8-768:vgprValuB_G1+8-768+3], v[vgprLocalReadAddrB+0-512] offset:64
ds_load_b128 v[vgprValuB_G1+12-768:vgprValuB_G1+12-768+3], v[vgprLocalReadAddrB+0-512] offset:96
ds_load_b128 v[vgprValuB_G1+16-768:vgprValuB_G1+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8704
ds_load_b128 v[vgprValuB_G1+20-768:vgprValuB_G1+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8736
ds_load_b128 v[vgprValuB_G1+24-768:vgprValuB_G1+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8768
ds_load_b128 v[vgprValuB_G1+28-768:vgprValuB_G1+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8800
ds_load_b128 v[vgprValuB_G1+32-768:vgprValuB_G1+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17408 
ds_load_b128 v[vgprValuB_G1+36-768:vgprValuB_G1+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17440
ds_load_b128 v[vgprValuB_G1+40-768:vgprValuB_G1+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17472 
ds_load_b128 v[vgprValuB_G1+44-768:vgprValuB_G1+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17504
ds_load_b128 v[vgprValuB_G1+48-768:vgprValuB_G1+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26112 
ds_load_b128 v[vgprValuB_G1+52-768:vgprValuB_G1+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26144
ds_load_b128 v[vgprValuB_G1+56-768:vgprValuB_G1+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26176 
ds_load_b128 v[vgprValuB_G1+60-768:vgprValuB_G1+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26208

/* local read prefetch a */
s_set_vgpr_msb 49794
ds_load_b128 v[vgprValuA_X0_I0+0-512:vgprValuA_X0_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:0
ds_load_b128 v[vgprValuA_X0_I0+4-512:vgprValuA_X0_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:32
ds_load_b128 v[vgprValuA_X0_I0+8-512:vgprValuA_X0_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:64 
ds_load_b128 v[vgprValuA_X0_I0+12-512:vgprValuA_X0_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:96 
ds_load_b128 v[vgprValuA_X0_I0+16-512:vgprValuA_X0_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8704 
ds_load_b128 v[vgprValuA_X0_I0+20-512:vgprValuA_X0_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8736 
ds_load_b128 v[vgprValuA_X0_I0+24-512:vgprValuA_X0_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8768 
ds_load_b128 v[vgprValuA_X0_I0+28-512:vgprValuA_X0_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8800
ds_load_b128 v[vgprValuA_X0_I0+32-512:vgprValuA_X0_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17408 
ds_load_b128 v[vgprValuA_X0_I0+36-512:vgprValuA_X0_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17440
ds_load_b128 v[vgprValuA_X0_I0+40-512:vgprValuA_X0_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17472 
ds_load_b128 v[vgprValuA_X0_I0+44-512:vgprValuA_X0_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17504
ds_load_b128 v[vgprValuA_X0_I0+48-512:vgprValuA_X0_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26112 
ds_load_b128 v[vgprValuA_X0_I0+52-512:vgprValuA_X0_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26144
ds_load_b128 v[vgprValuA_X0_I0+56-512:vgprValuA_X0_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26176 
ds_load_b128 v[vgprValuA_X0_I0+60-512:vgprValuA_X0_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26208
s_set_vgpr_msb 33282
s_set_vgpr_msb 642
ds_load_b128 v[vgprValuA_X0_I0+64-512:vgprValuA_X0_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34816
ds_load_b128 v[vgprValuA_X0_I0+68-512:vgprValuA_X0_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34848
ds_load_b128 v[vgprValuA_X0_I0+72-512:vgprValuA_X0_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:34880
ds_load_b128 v[vgprValuA_X0_I0+76-512:vgprValuA_X0_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:34912
ds_load_b128 v[vgprValuA_X0_I0+80-512:vgprValuA_X0_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43520 
ds_load_b128 v[vgprValuA_X0_I0+84-512:vgprValuA_X0_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43552
ds_load_b128 v[vgprValuA_X0_I0+88-512:vgprValuA_X0_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43584
ds_load_b128 v[vgprValuA_X0_I0+92-512:vgprValuA_X0_I0+92-512+3], v[vgprLocalReadAddrA+0-512] offset:43616
ds_load_b128 v[vgprValuA_X0_I0+96-512:vgprValuA_X0_I0+96-512+3], v[vgprLocalReadAddrA+0-512] offset:52224 
ds_load_b128 v[vgprValuA_X0_I0+100-512:vgprValuA_X0_I0+100-512+3], v[vgprLocalReadAddrA+0-512] offset:52256
ds_load_b128 v[vgprValuA_X0_I0+104-512:vgprValuA_X0_I0+104-512+3], v[vgprLocalReadAddrA+0-512] offset:52288 
ds_load_b128 v[vgprValuA_X0_I0+108-512:vgprValuA_X0_I0+108-512+3], v[vgprLocalReadAddrA+0-512] offset:52320
ds_load_b128 v[vgprValuA_X0_I0+112-512:vgprValuA_X0_I0+112-512+3], v[vgprLocalReadAddrA+0-512] offset:60928
ds_load_b128 v[vgprValuA_X0_I0+116-512:vgprValuA_X0_I0+116-512+3], v[vgprLocalReadAddrA+0-512] offset:60960
ds_load_b128 v[vgprValuA_X0_I0+120-512:vgprValuA_X0_I0+120-512+3], v[vgprLocalReadAddrA+0-512] offset:60992
ds_load_b128 v[vgprValuA_X0_I0+124-512:vgprValuA_X0_I0+124-512+3], v[vgprLocalReadAddrA+0-512] offset:61024

/* local read prefetch b */
s_set_vgpr_msb 33474
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34816
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34848
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:34880
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:34912
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43520
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43552
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43584
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43616
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52224
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52256
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52288
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52320
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:60928
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:60960
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:60992
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61024

// cluster sync before loop 
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_wait_alu depctr_sa_sdst(0)
s_cbranch_scc0 label_Skip_Signal_10
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_10:

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
; s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
; s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
label_Persist_Start_1:
s_nop 0
s_setreg_IMM32_b32 hwreg(26,0,2), 2                // set expert mode = 1 in unrolled loop

label_Loop_InitC:
// set vgprValuB idx
.set vgprValuB_G0, vgprValuB_Y0+0
.set vgprValuB_G1, vgprValuB_Y1+0
.set vgprValuB_G2, vgprValuB_Y2+0
/* iter 0 */
s_set_vgpr_msb 49678
s_wait_dscnt 32 // half A + half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:128 
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:160
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:192
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:224
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17568
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17600
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17632 
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26304 
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+0-512:vgprValuA_X1_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:128 
ds_load_b128 v[vgprValuA_X1_I0+4-512:vgprValuA_X1_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:160
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+8-512:vgprValuA_X1_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:192 
ds_load_b128 v[vgprValuA_X1_I0+12-512:vgprValuA_X1_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:224
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+16-512:vgprValuA_X1_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8832 
ds_load_b128 v[vgprValuA_X1_I0+20-512:vgprValuA_X1_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+24-512:vgprValuA_X1_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X1_I0+28-512:vgprValuA_X1_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 33294
s_wait_dscnt 40 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+32-512:vgprValuA_X1_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X1_I0+36-512:vgprValuA_X1_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17568
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+40-512:vgprValuA_X1_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17600
ds_load_b128 v[vgprValuA_X1_I0+44-512:vgprValuA_X1_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17632
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+48-512:vgprValuA_X1_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26240 
ds_load_b128 v[vgprValuA_X1_I0+52-512:vgprValuA_X1_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+56-512:vgprValuA_X1_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X1_I0+60-512:vgprValuA_X1_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+64-512:vgprValuA_X1_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34944 
ds_load_b128 v[vgprValuA_X1_I0+68-512:vgprValuA_X1_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34976
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+72-512:vgprValuA_X1_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:35008 
ds_load_b128 v[vgprValuA_X1_I0+76-512:vgprValuA_X1_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:35040
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+80-512:vgprValuA_X1_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43648 
ds_load_b128 v[vgprValuA_X1_I0+84-512:vgprValuA_X1_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+88-512:vgprValuA_X1_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43712
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+92-768:vgprValuA_X1_I0+92-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
ds_load_b128 v[vgprValuA_X1_I0+96-768:vgprValuA_X1_I0+96-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+100-768:vgprValuA_X1_I0+100-768+3], v[vgprLocalReadAddrA+0-512] offset:52384 
ds_load_b128 v[vgprValuA_X1_I0+104-768:vgprValuA_X1_I0+104-768+3], v[vgprLocalReadAddrA+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+108-768:vgprValuA_X1_I0+108-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X1_I0+112-768:vgprValuA_X1_I0+112-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+116-768:vgprValuA_X1_I0+116-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 49758
s_wait_dscnt 46 // another half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+120-768:vgprValuA_X1_I0+120-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+124-768:vgprValuA_X1_I0+124-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+0-768:vgprValuB_G0+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34944  
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+4-768:vgprValuB_G0+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34976
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+8-768:vgprValuB_G0+8-768+3], v[vgprLocalReadAddrB+0-512] offset:35008 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+12-768:vgprValuB_G0+12-768+3], v[vgprLocalReadAddrB+0-512] offset:35040
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+16-768:vgprValuB_G0+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43648
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+20-768:vgprValuB_G0+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43680 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+24-768:vgprValuB_G0+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43712 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258  
ds_load_b128 v[vgprValuB_G0+28-768:vgprValuB_G0+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+32-768:vgprValuB_G0+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+36-768:vgprValuB_G0+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+40-768:vgprValuB_G0+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+44-768:vgprValuB_G0+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+48-768:vgprValuB_G0+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+52-768:vgprValuB_G0+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+56-768:vgprValuB_G0+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch
s_set_vgpr_msb 24067 // global vaddr = src0
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null  scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch
s_set_vgpr_msb 862
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+60-768:vgprValuB_G0+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_cmp_eq_i32 s[sgprIter], 3
s_cselect_b32 s92, 0, 1
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 2
s_cmov_b32 s[sgprtdmAGroup0+0], s92                       // Set TDM as NULL in tail loops
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1]            // update TDM to next AB addr
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 1
s_cmov_b32 s[sgprtdmAGroup0+0], 0
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 24238
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_branch label_Loop_InitC_End

label_LoopBeginL:
/******************************************/
/* Unrolled Loop 1/3 - Begin              */
/******************************************/
// set vgprValuB idx
.set vgprValuB_G0, vgprValuB_Y0+0
.set vgprValuB_G1, vgprValuB_Y1+0
.set vgprValuB_G2, vgprValuB_Y2+0
/* iter 0 */
s_set_vgpr_msb 44558
s_wait_dscnt 32 // half A + half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+0:vgprValuC+0+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+8:vgprValuC+8+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+16:vgprValuC+16+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+24:vgprValuC+24+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+88:vgprValuC+88+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:128 
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:160
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+80:vgprValuC+80+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:192
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:224
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+72:vgprValuC+72+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+64:vgprValuC+64+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+128:vgprValuC+128+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17568
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+136:vgprValuC+136+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17600
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17632 
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+144:vgprValuC+144+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+152:vgprValuC+152+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26304 
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+216:vgprValuC+216+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+0-512:vgprValuA_X1_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:128 
ds_load_b128 v[vgprValuA_X1_I0+4-512:vgprValuA_X1_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:160
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+208:vgprValuC+208+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+8-512:vgprValuA_X1_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:192 
ds_load_b128 v[vgprValuA_X1_I0+12-512:vgprValuA_X1_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:224
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+200:vgprValuC+200+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+16-512:vgprValuA_X1_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8832 
ds_load_b128 v[vgprValuA_X1_I0+20-512:vgprValuA_X1_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+192:vgprValuC+192+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+24-512:vgprValuA_X1_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X1_I0+28-512:vgprValuA_X1_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 33294
s_wait_dscnt 40 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+32:vgprValuC+32+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+32-512:vgprValuA_X1_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X1_I0+36-512:vgprValuA_X1_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17568
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+40:vgprValuC+40+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+40-512:vgprValuA_X1_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17600
ds_load_b128 v[vgprValuA_X1_I0+44-512:vgprValuA_X1_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17632
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+48:vgprValuC+48+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+48-512:vgprValuA_X1_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26240 
ds_load_b128 v[vgprValuA_X1_I0+52-512:vgprValuA_X1_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+56:vgprValuC+56+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+56-512:vgprValuA_X1_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X1_I0+60-512:vgprValuA_X1_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+120:vgprValuC+120+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+112:vgprValuC+112+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+104:vgprValuC+104+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+96:vgprValuC+96+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+160:vgprValuC+160+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+64-512:vgprValuA_X1_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34944 
ds_load_b128 v[vgprValuA_X1_I0+68-512:vgprValuA_X1_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34976
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+168:vgprValuC+168+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+72-512:vgprValuA_X1_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:35008 
ds_load_b128 v[vgprValuA_X1_I0+76-512:vgprValuA_X1_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:35040
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+176:vgprValuC+176+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+80-512:vgprValuA_X1_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43648 
ds_load_b128 v[vgprValuA_X1_I0+84-512:vgprValuA_X1_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+184:vgprValuC+184+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+88-512:vgprValuA_X1_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43712
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+248-256:vgprValuC+248-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+92-768:vgprValuA_X1_I0+92-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
ds_load_b128 v[vgprValuA_X1_I0+96-768:vgprValuA_X1_I0+96-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+240-256:vgprValuC+240-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+100-768:vgprValuA_X1_I0+100-768+3], v[vgprLocalReadAddrA+0-512] offset:52384 
ds_load_b128 v[vgprValuA_X1_I0+104-768:vgprValuA_X1_I0+104-768+3], v[vgprLocalReadAddrA+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+232-256:vgprValuC+232-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+108-768:vgprValuA_X1_I0+108-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X1_I0+112-768:vgprValuA_X1_I0+112-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+224-256:vgprValuC+224-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+116-768:vgprValuA_X1_I0+116-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 49758
s_wait_dscnt 46 // another half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+256-256:vgprValuC+256-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+264-256:vgprValuC+264-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+272-256:vgprValuC+272-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+120-768:vgprValuA_X1_I0+120-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+280-256:vgprValuC+280-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+124-768:vgprValuA_X1_I0+124-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+288-256:vgprValuC+288-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+0-768:vgprValuB_G0+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34944  
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+296-256:vgprValuC+296-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+4-768:vgprValuB_G0+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34976
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+304-256:vgprValuC+304-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+8-768:vgprValuB_G0+8-768+3], v[vgprLocalReadAddrB+0-512] offset:35008 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+312-256:vgprValuC+312-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+12-768:vgprValuB_G0+12-768+3], v[vgprLocalReadAddrB+0-512] offset:35040
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+320-256:vgprValuC+320-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+16-768:vgprValuB_G0+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43648
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+328-256:vgprValuC+328-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+20-768:vgprValuB_G0+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43680 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+336-256:vgprValuC+336-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+24-768:vgprValuB_G0+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43712 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+344-256:vgprValuC+344-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+28-768:vgprValuB_G0+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+352-256:vgprValuC+352-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+32-768:vgprValuB_G0+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+360-256:vgprValuC+360-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+36-768:vgprValuB_G0+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+368-256:vgprValuC+368-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+40-768:vgprValuB_G0+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+376-256:vgprValuC+376-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+44-768:vgprValuB_G0+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+48-768:vgprValuB_G0+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+52-768:vgprValuB_G0+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+56-768:vgprValuB_G0+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch
s_set_vgpr_msb 24067 // global vaddr = src0
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null  scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch
s_set_vgpr_msb 862
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+60-768:vgprValuB_G0+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_cmp_eq_i32 s[sgprIter], 3

s_cselect_b32 s92, 0, 1
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 2

s_cmov_b32 s[sgprtdmAGroup0+0], s92                       // Set TDM as NULL in tail loops
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1]            // update TDM to next AB addr
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 1

s_cmov_b32 s[sgprtdmAGroup0+0], 0
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 24238
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
label_Loop_InitC_End:
/* iter 1 (reset local read pointers iteration) (swap local read pointers iteration)  */
s_wait_tensorcnt 1             // 1wait for global read
s_ttracedata_imm 0
s_wait_dscnt 32                // MX + half A + half B

// cluster sync in loop 
s_barrier_wait -3
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_4
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_4:


s_set_vgpr_msb 44558
s_wait_alu depctr_va_vdst(0) // lds addr update
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+0:vgprValuC+0+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+8:vgprValuC+8+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+16:vgprValuC+16+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+24:vgprValuC+24+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+88:vgprValuC+88+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+80:vgprValuC+80+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+0-768:vgprValuB_G1+0-768+3], v[vgprLocalReadAddrB+0-512] offset:0 
ds_load_b128 v[vgprValuB_G1+4-768:vgprValuB_G1+4-768+3], v[vgprLocalReadAddrB+0-512] offset:32
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+72:vgprValuC+72+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+8-768:vgprValuB_G1+8-768+3], v[vgprLocalReadAddrB+0-512] offset:64
ds_load_b128 v[vgprValuB_G1+12-768:vgprValuB_G1+12-768+3], v[vgprLocalReadAddrB+0-512] offset:96
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+64:vgprValuC+64+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+16-768:vgprValuB_G1+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8704
ds_load_b128 v[vgprValuB_G1+20-768:vgprValuB_G1+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8736
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+128:vgprValuC+128+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+24-768:vgprValuB_G1+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8768
ds_load_b128 v[vgprValuB_G1+28-768:vgprValuB_G1+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8800
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+136:vgprValuC+136+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+32-768:vgprValuB_G1+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17408 
ds_load_b128 v[vgprValuB_G1+36-768:vgprValuB_G1+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17440
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+144:vgprValuC+144+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+40-768:vgprValuB_G1+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17472 
ds_load_b128 v[vgprValuB_G1+44-768:vgprValuB_G1+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17504
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+152:vgprValuC+152+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+48-768:vgprValuB_G1+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26112 
ds_load_b128 v[vgprValuB_G1+52-768:vgprValuB_G1+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26144
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+216:vgprValuC+216+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+56-768:vgprValuB_G1+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26176 
ds_load_b128 v[vgprValuB_G1+60-768:vgprValuB_G1+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26208
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+208:vgprValuC+208+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+0-512:vgprValuA_X0_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:0
ds_load_b128 v[vgprValuA_X0_I0+4-512:vgprValuA_X0_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:32
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+200:vgprValuC+200+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+8-512:vgprValuA_X0_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:64 
ds_load_b128 v[vgprValuA_X0_I0+12-512:vgprValuA_X0_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:96
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+192:vgprValuC+192+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+16-512:vgprValuA_X0_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8704 
ds_load_b128 v[vgprValuA_X0_I0+20-512:vgprValuA_X0_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8736
s_set_vgpr_msb 33294
s_wait_dscnt 38 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+32:vgprValuC+32+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+24-512:vgprValuA_X0_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8768 
ds_load_b128 v[vgprValuA_X0_I0+28-512:vgprValuA_X0_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8800
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+40:vgprValuC+40+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+32-512:vgprValuA_X0_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17408 
ds_load_b128 v[vgprValuA_X0_I0+36-512:vgprValuA_X0_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17440
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+48:vgprValuC+48+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+40-512:vgprValuA_X0_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17472 
ds_load_b128 v[vgprValuA_X0_I0+44-512:vgprValuA_X0_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17504
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+56:vgprValuC+56+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+48-512:vgprValuA_X0_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26112 
ds_load_b128 v[vgprValuA_X0_I0+52-512:vgprValuA_X0_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26144
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+120:vgprValuC+120+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+56-512:vgprValuA_X0_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26176 
ds_load_b128 v[vgprValuA_X0_I0+60-512:vgprValuA_X0_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26208
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+112:vgprValuC+112+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3842
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+104:vgprValuC+104+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+96:vgprValuC+96+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+160:vgprValuC+160+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+168:vgprValuC+168+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3599
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+176:vgprValuC+176+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+184:vgprValuC+184+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3935
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+248-256:vgprValuC+248-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+240-256:vgprValuC+240-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24414
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+232-256:vgprValuC+232-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+224-256:vgprValuC+224-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_wait_tensorcnt 1             // 1wait for global read
s_ttracedata_imm 0
s_wait_dscnt 32 // another half B

// cluster sync in loop 
s_barrier_wait -3
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_5
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_5:


s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+256-256:vgprValuC+256-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+264-256:vgprValuC+264-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+64-512:vgprValuA_X0_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34816
ds_load_b128 v[vgprValuA_X0_I0+68-512:vgprValuA_X0_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34848
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+272-256:vgprValuC+272-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+72-512:vgprValuA_X0_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:34880
ds_load_b128 v[vgprValuA_X0_I0+76-512:vgprValuA_X0_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:34912
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+280-256:vgprValuC+280-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+80-512:vgprValuA_X0_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43520 
ds_load_b128 v[vgprValuA_X0_I0+84-512:vgprValuA_X0_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43552
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+288-256:vgprValuC+288-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+88-512:vgprValuA_X0_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43584
ds_load_b128 v[vgprValuA_X0_I0+92-512:vgprValuA_X0_I0+92-512+3], v[vgprLocalReadAddrA+0-512] offset:43616
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+296-256:vgprValuC+296-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+96-512:vgprValuA_X0_I0+96-512+3], v[vgprLocalReadAddrA+0-512] offset:52224 
ds_load_b128 v[vgprValuA_X0_I0+100-512:vgprValuA_X0_I0+100-512+3], v[vgprLocalReadAddrA+0-512] offset:52256
s_set_vgpr_msb 33375
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+304-256:vgprValuC+304-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24450
ds_load_b128 v[vgprValuA_X0_I0+104-512:vgprValuA_X0_I0+104-512+3], v[vgprLocalReadAddrA+0-512] offset:52288 
ds_load_b128 v[vgprValuA_X0_I0+108-512:vgprValuA_X0_I0+108-512+3], v[vgprLocalReadAddrA+0-512] offset:52320
s_set_vgpr_msb 33375
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+312-256:vgprValuC+312-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24450
ds_load_b128 v[vgprValuA_X0_I0+112-512:vgprValuA_X0_I0+112-512+3], v[vgprLocalReadAddrA+0-512] offset:60928
ds_load_b128 v[vgprValuA_X0_I0+116-512:vgprValuA_X0_I0+116-512+3], v[vgprLocalReadAddrA+0-512] offset:60960
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+320-256:vgprValuC+320-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+120-512:vgprValuA_X0_I0+120-512+3], v[vgprLocalReadAddrA+0-512] offset:60992
ds_load_b128 v[vgprValuA_X0_I0+124-512:vgprValuA_X0_I0+124-512+3], v[vgprLocalReadAddrA+0-512] offset:61024
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+328-256:vgprValuC+328-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34816
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34848
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+336-256:vgprValuC+336-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:34880
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:34912
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+344-256:vgprValuC+344-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43520
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43552
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+352-256:vgprValuC+352-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43584
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43616
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+360-256:vgprValuC+360-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52224
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52256
s_set_vgpr_msb 49759
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+368-256:vgprValuC+368-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24514
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52288
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52320
s_set_vgpr_msb 49759
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+376-256:vgprValuC+376-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24514
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:60928
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:60960
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:60992
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61024
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprIter], 3
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_cselect_b32 s90, 1, 0
s_cmp_eq_i32 s[sgprLoopCounterL], 5
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_cselect_b32 s91, 1, 0
s_set_vgpr_msb 24159
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_and_b32 s92, s90, s91             // if last persist iter && last 4 unrolled iters
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8

s_cmov_b32 s[sgprPrefetchIncAB], 0          // Set increment to 0
s_set_vgpr_msb 24414
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch pointers increment
s_set_vgpr_msb 24268
s_wait_alu depctr_sa_sdst(0)
v_add_nc_u64 v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1]
s_set_vgpr_msb 52318
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24268
s_set_vgpr_msb 52318
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_not_b32 s90, s90

s_and_b32 s92, s90, s91             // if not last persist iter && last 4 unrolled iters
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
v_cmp_eq_i32 vcc_lo, s92, 1
s_set_vgpr_msb 24238 // 128+2+12+32
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 44751
v_cndmask_b32 v[vgprGL2PrefetchAB-768+0], v[vgprGL2PrefetchAB-768+0], v[vgprGL2PrefetchABNext-768+0], vcc_lo
v_cndmask_b32 v[vgprGL2PrefetchAB-768+1], v[vgprGL2PrefetchAB-768+1], v[vgprGL2PrefetchABNext-768+1], vcc_lo
s_set_vgpr_msb 53166
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44751
s_set_vgpr_msb 53167
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_mov_b32 s[sgprExitLoopIdx], 1
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_wait_alu depctr_sa_sdst(0)
s_cbranch_scc1 label_LoopEndL

/******************************************/
/* Unrolled Loop 2/3 - Begin              */
/******************************************/
// set vgprValuB idx
.set vgprValuB_G0, vgprValuB_Y1+0
.set vgprValuB_G1, vgprValuB_Y2+0
.set vgprValuB_G2, vgprValuB_Y0+0
/* iter 0 */
s_set_vgpr_msb 44814
s_wait_dscnt 32 // half A + half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+0:vgprValuC+0+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+8:vgprValuC+8+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+16:vgprValuC+16+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+24:vgprValuC+24+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+88:vgprValuC+88+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:128 
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:160
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+80:vgprValuC+80+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:192
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:224
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+72:vgprValuC+72+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+64:vgprValuC+64+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+128:vgprValuC+128+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17568
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+136:vgprValuC+136+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17600
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17632 
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+144:vgprValuC+144+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+152:vgprValuC+152+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26304 
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+216:vgprValuC+216+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+0-512:vgprValuA_X1_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:128 
ds_load_b128 v[vgprValuA_X1_I0+4-512:vgprValuA_X1_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:160
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+208:vgprValuC+208+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+8-512:vgprValuA_X1_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:192 
ds_load_b128 v[vgprValuA_X1_I0+12-512:vgprValuA_X1_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:224
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+200:vgprValuC+200+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+16-512:vgprValuA_X1_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8832 
ds_load_b128 v[vgprValuA_X1_I0+20-512:vgprValuA_X1_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+192:vgprValuC+192+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+24-512:vgprValuA_X1_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X1_I0+28-512:vgprValuA_X1_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 33294
s_wait_dscnt 40 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+32:vgprValuC+32+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+32-512:vgprValuA_X1_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X1_I0+36-512:vgprValuA_X1_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17568
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+40:vgprValuC+40+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+40-512:vgprValuA_X1_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17600
ds_load_b128 v[vgprValuA_X1_I0+44-512:vgprValuA_X1_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17632
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+48:vgprValuC+48+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+48-512:vgprValuA_X1_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26240 
ds_load_b128 v[vgprValuA_X1_I0+52-512:vgprValuA_X1_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+56:vgprValuC+56+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+56-512:vgprValuA_X1_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X1_I0+60-512:vgprValuA_X1_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+120:vgprValuC+120+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+112:vgprValuC+112+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+104:vgprValuC+104+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+96:vgprValuC+96+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+160:vgprValuC+160+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+64-512:vgprValuA_X1_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34944 
ds_load_b128 v[vgprValuA_X1_I0+68-512:vgprValuA_X1_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34976
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+168:vgprValuC+168+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+72-512:vgprValuA_X1_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:35008 
ds_load_b128 v[vgprValuA_X1_I0+76-512:vgprValuA_X1_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:35040
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+176:vgprValuC+176+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+80-512:vgprValuA_X1_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43648 
ds_load_b128 v[vgprValuA_X1_I0+84-512:vgprValuA_X1_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+184:vgprValuC+184+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+88-512:vgprValuA_X1_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43712
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+248-256:vgprValuC+248-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+92-768:vgprValuA_X1_I0+92-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
ds_load_b128 v[vgprValuA_X1_I0+96-768:vgprValuA_X1_I0+96-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+240-256:vgprValuC+240-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+100-768:vgprValuA_X1_I0+100-768+3], v[vgprLocalReadAddrA+0-512] offset:52384 
ds_load_b128 v[vgprValuA_X1_I0+104-768:vgprValuA_X1_I0+104-768+3], v[vgprLocalReadAddrA+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+232-256:vgprValuC+232-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+108-768:vgprValuA_X1_I0+108-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X1_I0+112-768:vgprValuA_X1_I0+112-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+224-256:vgprValuC+224-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+116-768:vgprValuA_X1_I0+116-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 49758
s_wait_dscnt 46 // another half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+256-256:vgprValuC+256-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+264-256:vgprValuC+264-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+272-256:vgprValuC+272-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+120-768:vgprValuA_X1_I0+120-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+280-256:vgprValuC+280-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+124-768:vgprValuA_X1_I0+124-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+288-256:vgprValuC+288-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+0-768:vgprValuB_G0+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34944  
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+296-256:vgprValuC+296-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+4-768:vgprValuB_G0+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34976
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+304-256:vgprValuC+304-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+8-768:vgprValuB_G0+8-768+3], v[vgprLocalReadAddrB+0-512] offset:35008 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+312-256:vgprValuC+312-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+12-768:vgprValuB_G0+12-768+3], v[vgprLocalReadAddrB+0-512] offset:35040
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+320-256:vgprValuC+320-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+16-768:vgprValuB_G0+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43648
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+328-256:vgprValuC+328-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+20-768:vgprValuB_G0+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43680 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+336-256:vgprValuC+336-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+24-768:vgprValuB_G0+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43712 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+344-256:vgprValuC+344-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+28-768:vgprValuB_G0+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+352-256:vgprValuC+352-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+32-768:vgprValuB_G0+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+360-256:vgprValuC+360-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+36-768:vgprValuB_G0+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+368-256:vgprValuC+368-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+40-768:vgprValuB_G0+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+376-256:vgprValuC+376-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+44-768:vgprValuB_G0+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+48-768:vgprValuB_G0+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+52-768:vgprValuB_G0+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+56-768:vgprValuB_G0+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch
s_set_vgpr_msb 24067 // global vaddr = src0
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null  scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch
s_set_vgpr_msb 862
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+60-768:vgprValuB_G0+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_cmp_eq_i32 s[sgprIter], 3

s_cselect_b32 s92, 0, 1
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 2

s_cmov_b32 s[sgprtdmAGroup0+0], s92                       // Set TDM as NULL in tail loops
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1]            // update TDM to next AB addr
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 1

s_cmov_b32 s[sgprtdmAGroup0+0], 0
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 24238
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
label_Persist_Start_2:
/* iter 1 (reset local read pointers iteration) (swap local read pointers iteration)  */
s_wait_tensorcnt 1             // 1wait for global read
s_ttracedata_imm 0
s_wait_dscnt 32                // MX + half A + half B

// cluster sync in loop 
s_barrier_wait -3
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_6
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_6:


s_set_vgpr_msb 44558
s_wait_alu depctr_va_vdst(0) // lds addr update
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+0:vgprValuC+0+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+8:vgprValuC+8+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+16:vgprValuC+16+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+24:vgprValuC+24+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+88:vgprValuC+88+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+80:vgprValuC+80+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+0-768:vgprValuB_G1+0-768+3], v[vgprLocalReadAddrB+0-512] offset:0 
ds_load_b128 v[vgprValuB_G1+4-768:vgprValuB_G1+4-768+3], v[vgprLocalReadAddrB+0-512] offset:32
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+72:vgprValuC+72+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+8-768:vgprValuB_G1+8-768+3], v[vgprLocalReadAddrB+0-512] offset:64
ds_load_b128 v[vgprValuB_G1+12-768:vgprValuB_G1+12-768+3], v[vgprLocalReadAddrB+0-512] offset:96
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+64:vgprValuC+64+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+16-768:vgprValuB_G1+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8704
ds_load_b128 v[vgprValuB_G1+20-768:vgprValuB_G1+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8736
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+128:vgprValuC+128+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+24-768:vgprValuB_G1+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8768
ds_load_b128 v[vgprValuB_G1+28-768:vgprValuB_G1+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8800
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+136:vgprValuC+136+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+32-768:vgprValuB_G1+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17408 
ds_load_b128 v[vgprValuB_G1+36-768:vgprValuB_G1+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17440
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+144:vgprValuC+144+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+40-768:vgprValuB_G1+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17472 
ds_load_b128 v[vgprValuB_G1+44-768:vgprValuB_G1+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17504
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+152:vgprValuC+152+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+48-768:vgprValuB_G1+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26112 
ds_load_b128 v[vgprValuB_G1+52-768:vgprValuB_G1+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26144
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+216:vgprValuC+216+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+56-768:vgprValuB_G1+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26176 
ds_load_b128 v[vgprValuB_G1+60-768:vgprValuB_G1+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26208
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+208:vgprValuC+208+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+0-512:vgprValuA_X0_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:0
ds_load_b128 v[vgprValuA_X0_I0+4-512:vgprValuA_X0_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:32
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+200:vgprValuC+200+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+8-512:vgprValuA_X0_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:64 
ds_load_b128 v[vgprValuA_X0_I0+12-512:vgprValuA_X0_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:96
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+192:vgprValuC+192+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+16-512:vgprValuA_X0_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8704 
ds_load_b128 v[vgprValuA_X0_I0+20-512:vgprValuA_X0_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8736
s_set_vgpr_msb 33294
s_wait_dscnt 38 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+32:vgprValuC+32+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+24-512:vgprValuA_X0_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8768 
ds_load_b128 v[vgprValuA_X0_I0+28-512:vgprValuA_X0_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8800
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+40:vgprValuC+40+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+32-512:vgprValuA_X0_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17408 
ds_load_b128 v[vgprValuA_X0_I0+36-512:vgprValuA_X0_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17440
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+48:vgprValuC+48+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+40-512:vgprValuA_X0_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17472 
ds_load_b128 v[vgprValuA_X0_I0+44-512:vgprValuA_X0_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17504
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+56:vgprValuC+56+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+48-512:vgprValuA_X0_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26112 
ds_load_b128 v[vgprValuA_X0_I0+52-512:vgprValuA_X0_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26144
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+120:vgprValuC+120+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+56-512:vgprValuA_X0_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26176 
ds_load_b128 v[vgprValuA_X0_I0+60-512:vgprValuA_X0_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26208
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+112:vgprValuC+112+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3842
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+104:vgprValuC+104+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+96:vgprValuC+96+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+160:vgprValuC+160+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+168:vgprValuC+168+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3599
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+176:vgprValuC+176+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+184:vgprValuC+184+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3935
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+248-256:vgprValuC+248-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+240-256:vgprValuC+240-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24414
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+232-256:vgprValuC+232-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+224-256:vgprValuC+224-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_wait_tensorcnt 1             // 1wait for global read
s_ttracedata_imm 0
s_wait_dscnt 32 // another half B

// cluster sync in loop 
s_barrier_wait -3
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_7
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_7:


s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+256-256:vgprValuC+256-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+264-256:vgprValuC+264-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+64-512:vgprValuA_X0_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34816
ds_load_b128 v[vgprValuA_X0_I0+68-512:vgprValuA_X0_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34848
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+272-256:vgprValuC+272-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+72-512:vgprValuA_X0_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:34880
ds_load_b128 v[vgprValuA_X0_I0+76-512:vgprValuA_X0_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:34912
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+280-256:vgprValuC+280-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+80-512:vgprValuA_X0_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43520 
ds_load_b128 v[vgprValuA_X0_I0+84-512:vgprValuA_X0_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43552
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+288-256:vgprValuC+288-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+88-512:vgprValuA_X0_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43584
ds_load_b128 v[vgprValuA_X0_I0+92-512:vgprValuA_X0_I0+92-512+3], v[vgprLocalReadAddrA+0-512] offset:43616
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+296-256:vgprValuC+296-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+96-512:vgprValuA_X0_I0+96-512+3], v[vgprLocalReadAddrA+0-512] offset:52224 
ds_load_b128 v[vgprValuA_X0_I0+100-512:vgprValuA_X0_I0+100-512+3], v[vgprLocalReadAddrA+0-512] offset:52256
s_set_vgpr_msb 33375
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+304-256:vgprValuC+304-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24450
ds_load_b128 v[vgprValuA_X0_I0+104-512:vgprValuA_X0_I0+104-512+3], v[vgprLocalReadAddrA+0-512] offset:52288 
ds_load_b128 v[vgprValuA_X0_I0+108-512:vgprValuA_X0_I0+108-512+3], v[vgprLocalReadAddrA+0-512] offset:52320
s_set_vgpr_msb 33375
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+312-256:vgprValuC+312-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24450
ds_load_b128 v[vgprValuA_X0_I0+112-512:vgprValuA_X0_I0+112-512+3], v[vgprLocalReadAddrA+0-512] offset:60928
ds_load_b128 v[vgprValuA_X0_I0+116-512:vgprValuA_X0_I0+116-512+3], v[vgprLocalReadAddrA+0-512] offset:60960
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+320-256:vgprValuC+320-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+120-512:vgprValuA_X0_I0+120-512+3], v[vgprLocalReadAddrA+0-512] offset:60992
ds_load_b128 v[vgprValuA_X0_I0+124-512:vgprValuA_X0_I0+124-512+3], v[vgprLocalReadAddrA+0-512] offset:61024
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+328-256:vgprValuC+328-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34816
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34848
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+336-256:vgprValuC+336-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:34880
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:34912
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+344-256:vgprValuC+344-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43520
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43552
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+352-256:vgprValuC+352-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43584
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43616
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+360-256:vgprValuC+360-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52224
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52256
s_set_vgpr_msb 49759
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+368-256:vgprValuC+368-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24514
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52288
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52320
s_set_vgpr_msb 49759
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+376-256:vgprValuC+376-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24514
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:60928
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:60960
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:60992
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61024
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprIter], 3
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_cselect_b32 s90, 1, 0
s_cmp_eq_i32 s[sgprLoopCounterL], 5
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_cselect_b32 s91, 1, 0
s_set_vgpr_msb 24159
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_and_b32 s92, s90, s91             // if last persist iter && last 4 unrolled iters
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8

s_cmov_b32 s[sgprPrefetchIncAB], 0          // Set increment to 0
s_set_vgpr_msb 24414
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch pointers increment
s_set_vgpr_msb 24268
s_wait_alu depctr_sa_sdst(0)
v_add_nc_u64 v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1]
s_set_vgpr_msb 52318
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24268
s_set_vgpr_msb 52318
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_not_b32 s90, s90

s_and_b32 s92, s90, s91             // if not last persist iter && last 4 unrolled iters
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
v_cmp_eq_i32 vcc_lo, s92, 1
s_set_vgpr_msb 24238 // 128+2+12+32
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 44751
v_cndmask_b32 v[vgprGL2PrefetchAB-768+0], v[vgprGL2PrefetchAB-768+0], v[vgprGL2PrefetchABNext-768+0], vcc_lo
v_cndmask_b32 v[vgprGL2PrefetchAB-768+1], v[vgprGL2PrefetchAB-768+1], v[vgprGL2PrefetchABNext-768+1], vcc_lo
s_set_vgpr_msb 53166
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44751
s_set_vgpr_msb 53167
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_mov_b32 s[sgprExitLoopIdx], 2
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_wait_alu depctr_sa_sdst(0)
s_cbranch_scc1 label_LoopEndL

/******************************************/
/* Unrolled Loop 3/3 - Begin              */
/******************************************/
// set vgprValuB idx
.set vgprValuB_G0, vgprValuB_Y2
.set vgprValuB_G1, vgprValuB_Y0
.set vgprValuB_G2, vgprValuB_Y1
/* iter 0 */
s_set_vgpr_msb 44814
s_wait_dscnt 32 // half A + half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+0:vgprValuC+0+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+8:vgprValuC+8+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+16:vgprValuC+16+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+24:vgprValuC+24+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+88:vgprValuC+88+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:128 
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:160
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+80:vgprValuC+80+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:192
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:224
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+72:vgprValuC+72+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+64:vgprValuC+64+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+128:vgprValuC+128+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17568
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+136:vgprValuC+136+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17600
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17632 
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+144:vgprValuC+144+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+152:vgprValuC+152+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26304 
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+216:vgprValuC+216+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+0-512:vgprValuA_X1_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:128 
ds_load_b128 v[vgprValuA_X1_I0+4-512:vgprValuA_X1_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:160
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+208:vgprValuC+208+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+8-512:vgprValuA_X1_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:192 
ds_load_b128 v[vgprValuA_X1_I0+12-512:vgprValuA_X1_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:224
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+200:vgprValuC+200+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+16-512:vgprValuA_X1_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8832 
ds_load_b128 v[vgprValuA_X1_I0+20-512:vgprValuA_X1_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+192:vgprValuC+192+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+24-512:vgprValuA_X1_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X1_I0+28-512:vgprValuA_X1_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 33294
s_wait_dscnt 40 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+32:vgprValuC+32+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+32-512:vgprValuA_X1_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X1_I0+36-512:vgprValuA_X1_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17568
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+40:vgprValuC+40+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+40-512:vgprValuA_X1_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17600
ds_load_b128 v[vgprValuA_X1_I0+44-512:vgprValuA_X1_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17632
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+48:vgprValuC+48+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+48-512:vgprValuA_X1_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26240 
ds_load_b128 v[vgprValuA_X1_I0+52-512:vgprValuA_X1_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+56:vgprValuC+56+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+56-512:vgprValuA_X1_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X1_I0+60-512:vgprValuA_X1_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+120:vgprValuC+120+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+112:vgprValuC+112+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+104:vgprValuC+104+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+96:vgprValuC+96+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+160:vgprValuC+160+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+64-512:vgprValuA_X1_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34944 
ds_load_b128 v[vgprValuA_X1_I0+68-512:vgprValuA_X1_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34976
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+168:vgprValuC+168+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+72-512:vgprValuA_X1_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:35008 
ds_load_b128 v[vgprValuA_X1_I0+76-512:vgprValuA_X1_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:35040
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+176:vgprValuC+176+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+80-512:vgprValuA_X1_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43648 
ds_load_b128 v[vgprValuA_X1_I0+84-512:vgprValuA_X1_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+184:vgprValuC+184+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+88-512:vgprValuA_X1_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43712
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+248-256:vgprValuC+248-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+92-768:vgprValuA_X1_I0+92-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
ds_load_b128 v[vgprValuA_X1_I0+96-768:vgprValuA_X1_I0+96-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+240-256:vgprValuC+240-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+100-768:vgprValuA_X1_I0+100-768+3], v[vgprLocalReadAddrA+0-512] offset:52384 
ds_load_b128 v[vgprValuA_X1_I0+104-768:vgprValuA_X1_I0+104-768+3], v[vgprLocalReadAddrA+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+232-256:vgprValuC+232-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+108-768:vgprValuA_X1_I0+108-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X1_I0+112-768:vgprValuA_X1_I0+112-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+224-256:vgprValuC+224-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+116-768:vgprValuA_X1_I0+116-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 49758
s_wait_dscnt 46 // another half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+256-256:vgprValuC+256-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+264-256:vgprValuC+264-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+272-256:vgprValuC+272-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+120-768:vgprValuA_X1_I0+120-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+280-256:vgprValuC+280-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+124-768:vgprValuA_X1_I0+124-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+288-256:vgprValuC+288-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+0-768:vgprValuB_G0+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34944  
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+296-256:vgprValuC+296-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+4-768:vgprValuB_G0+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34976
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+304-256:vgprValuC+304-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+8-768:vgprValuB_G0+8-768+3], v[vgprLocalReadAddrB+0-512] offset:35008 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], v[vgprValuC+312-256:vgprValuC+312-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+12-768:vgprValuB_G0+12-768+3], v[vgprLocalReadAddrB+0-512] offset:35040
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+320-256:vgprValuC+320-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+16-768:vgprValuB_G0+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43648
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+328-256:vgprValuC+328-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+20-768:vgprValuB_G0+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43680 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+336-256:vgprValuC+336-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+24-768:vgprValuB_G0+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43712 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+344-256:vgprValuC+344-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+28-768:vgprValuB_G0+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+352-256:vgprValuC+352-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+32-768:vgprValuB_G0+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+360-256:vgprValuC+360-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+36-768:vgprValuB_G0+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+368-256:vgprValuC+368-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+40-768:vgprValuB_G0+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], v[vgprValuC+376-256:vgprValuC+376-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+44-768:vgprValuB_G0+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+48-768:vgprValuB_G0+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+52-768:vgprValuB_G0+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+56-768:vgprValuB_G0+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch
s_set_vgpr_msb 24067 // global vaddr = src0
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null  scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch
s_set_vgpr_msb 862
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+60-768:vgprValuB_G0+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_cmp_eq_i32 s[sgprIter], 3

s_cselect_b32 s92, 0, 1
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 2

s_cmov_b32 s[sgprtdmAGroup0+0], s92                       // Set TDM as NULL in tail loops
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1]            // update TDM to next AB addr
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 1

s_cmov_b32 s[sgprtdmAGroup0+0], 0
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 24238
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
label_Persist_Start_3:
/* iter 1 (reset local read pointers iteration) (swap local read pointers iteration)  */
s_wait_tensorcnt 1             // 1wait for global read
s_ttracedata_imm 0
s_wait_dscnt 32                // MX + half A + half B

// cluster sync in loop 
s_barrier_wait -3
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_8
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_8:


s_set_vgpr_msb 44558
s_wait_alu depctr_va_vdst(0) // lds addr update
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+0:vgprValuC+0+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+8:vgprValuC+8+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+16:vgprValuC+16+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+24:vgprValuC+24+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+88:vgprValuC+88+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+80:vgprValuC+80+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+0-768:vgprValuB_G1+0-768+3], v[vgprLocalReadAddrB+0-512] offset:0 
ds_load_b128 v[vgprValuB_G1+4-768:vgprValuB_G1+4-768+3], v[vgprLocalReadAddrB+0-512] offset:32
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+72:vgprValuC+72+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+8-768:vgprValuB_G1+8-768+3], v[vgprLocalReadAddrB+0-512] offset:64
ds_load_b128 v[vgprValuB_G1+12-768:vgprValuB_G1+12-768+3], v[vgprLocalReadAddrB+0-512] offset:96
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+64:vgprValuC+64+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+16-768:vgprValuB_G1+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8704
ds_load_b128 v[vgprValuB_G1+20-768:vgprValuB_G1+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8736
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+128:vgprValuC+128+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+24-768:vgprValuB_G1+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8768
ds_load_b128 v[vgprValuB_G1+28-768:vgprValuB_G1+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8800
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+136:vgprValuC+136+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+32-768:vgprValuB_G1+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17408 
ds_load_b128 v[vgprValuB_G1+36-768:vgprValuB_G1+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17440
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+144:vgprValuC+144+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+40-768:vgprValuB_G1+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17472 
ds_load_b128 v[vgprValuB_G1+44-768:vgprValuB_G1+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17504
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+152:vgprValuC+152+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+48-768:vgprValuB_G1+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26112 
ds_load_b128 v[vgprValuB_G1+52-768:vgprValuB_G1+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26144
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+216:vgprValuC+216+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G1+56-768:vgprValuB_G1+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26176 
ds_load_b128 v[vgprValuB_G1+60-768:vgprValuB_G1+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26208
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+208:vgprValuC+208+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+0-512:vgprValuA_X0_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:0
ds_load_b128 v[vgprValuA_X0_I0+4-512:vgprValuA_X0_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:32
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+200:vgprValuC+200+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+8-512:vgprValuA_X0_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:64 
ds_load_b128 v[vgprValuA_X0_I0+12-512:vgprValuA_X0_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:96
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+192:vgprValuC+192+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+16-512:vgprValuA_X0_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8704 
ds_load_b128 v[vgprValuA_X0_I0+20-512:vgprValuA_X0_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8736
s_set_vgpr_msb 33294
s_wait_dscnt 38 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+32:vgprValuC+32+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+24-512:vgprValuA_X0_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8768 
ds_load_b128 v[vgprValuA_X0_I0+28-512:vgprValuA_X0_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8800
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+40:vgprValuC+40+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X0_I0+32-512:vgprValuA_X0_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17408 
ds_load_b128 v[vgprValuA_X0_I0+36-512:vgprValuA_X0_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17440
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+48:vgprValuC+48+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+40-512:vgprValuA_X0_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17472 
ds_load_b128 v[vgprValuA_X0_I0+44-512:vgprValuA_X0_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17504
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+0+0+0-768:vgprValuB_G2+0+0+0-768+15], v[vgprValuC+56:vgprValuC+56+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+48-512:vgprValuA_X0_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26112 
ds_load_b128 v[vgprValuA_X0_I0+52-512:vgprValuA_X0_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26144
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+120:vgprValuC+120+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3970
ds_load_b128 v[vgprValuA_X0_I0+56-512:vgprValuA_X0_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26176 
ds_load_b128 v[vgprValuA_X0_I0+60-512:vgprValuA_X0_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26208
s_set_vgpr_msb 33295
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+112:vgprValuC+112+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3842
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+104:vgprValuC+104+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+16+0+0-768:vgprValuB_G2+16+0+0-768+15], v[vgprValuC+96:vgprValuC+96+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+160:vgprValuC+160+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+168:vgprValuC+168+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3599
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+176:vgprValuC+176+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+32+0+0-768:vgprValuB_G2+32+0+0-768+15], v[vgprValuC+184:vgprValuC+184+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3935
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+248-256:vgprValuC+248-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+240-256:vgprValuC+240-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24414
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+232-256:vgprValuC+232-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G2+48+0+0-768:vgprValuB_G2+48+0+0-768+15], v[vgprValuC+224-256:vgprValuC+224-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_wait_tensorcnt 1             // 1wait for global read
s_ttracedata_imm 0
s_wait_dscnt 32 // another half B

// cluster sync in loop 
s_barrier_wait -3
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_cbranch_scc0 label_Skip_Signal_9
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_9:


s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+256-256:vgprValuC+256-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+264-256:vgprValuC+264-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+64-512:vgprValuA_X0_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34816
ds_load_b128 v[vgprValuA_X0_I0+68-512:vgprValuA_X0_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34848
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+272-256:vgprValuC+272-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+72-512:vgprValuA_X0_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:34880
ds_load_b128 v[vgprValuA_X0_I0+76-512:vgprValuA_X0_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:34912
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+280-256:vgprValuC+280-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+80-512:vgprValuA_X0_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43520 
ds_load_b128 v[vgprValuA_X0_I0+84-512:vgprValuA_X0_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43552
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+288-256:vgprValuC+288-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+88-512:vgprValuA_X0_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43584
ds_load_b128 v[vgprValuA_X0_I0+92-512:vgprValuA_X0_I0+92-512+3], v[vgprLocalReadAddrA+0-512] offset:43616
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+296-256:vgprValuC+296-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+96-512:vgprValuA_X0_I0+96-512+3], v[vgprLocalReadAddrA+0-512] offset:52224 
ds_load_b128 v[vgprValuA_X0_I0+100-512:vgprValuA_X0_I0+100-512+3], v[vgprLocalReadAddrA+0-512] offset:52256
s_set_vgpr_msb 33375
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+304-256:vgprValuC+304-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24450
ds_load_b128 v[vgprValuA_X0_I0+104-512:vgprValuA_X0_I0+104-512+3], v[vgprLocalReadAddrA+0-512] offset:52288 
ds_load_b128 v[vgprValuA_X0_I0+108-512:vgprValuA_X0_I0+108-512+3], v[vgprLocalReadAddrA+0-512] offset:52320
s_set_vgpr_msb 33375
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], v[vgprValuC+312-256:vgprValuC+312-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24450
ds_load_b128 v[vgprValuA_X0_I0+112-512:vgprValuA_X0_I0+112-512+3], v[vgprLocalReadAddrA+0-512] offset:60928
ds_load_b128 v[vgprValuA_X0_I0+116-512:vgprValuA_X0_I0+116-512+3], v[vgprLocalReadAddrA+0-512] offset:60960
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+320-256:vgprValuC+320-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24194
ds_load_b128 v[vgprValuA_X0_I0+120-512:vgprValuA_X0_I0+120-512+3], v[vgprLocalReadAddrA+0-512] offset:60992
ds_load_b128 v[vgprValuA_X0_I0+124-512:vgprValuA_X0_I0+124-512+3], v[vgprLocalReadAddrA+0-512] offset:61024
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+328-256:vgprValuC+328-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34816
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34848
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+336-256:vgprValuC+336-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:34880
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:34912
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+344-256:vgprValuC+344-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43520
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43552
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+352-256:vgprValuC+352-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43584
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43616
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+360-256:vgprValuC+360-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52224
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52256
s_set_vgpr_msb 49759
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+368-256:vgprValuC+368-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24514
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52288
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52320
s_set_vgpr_msb 49759
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], v[vgprValuC+376-256:vgprValuC+376-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24514
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:60928
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:60960
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+384-256:vgprValuC+384-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:60992
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61024
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+392-256:vgprValuC+392-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+400-256:vgprValuC+400-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+408-256:vgprValuC+408-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprIter], 3
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+416-256:vgprValuC+416-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_cselect_b32 s90, 1, 0
s_cmp_eq_i32 s[sgprLoopCounterL], 5
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+424-256:vgprValuC+424-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_cselect_b32 s91, 1, 0
s_set_vgpr_msb 24159
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+432-256:vgprValuC+432-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_and_b32 s92, s90, s91             // if last persist iter && last 4 unrolled iters
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], v[vgprValuC+440-256:vgprValuC+440-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8

s_cmov_b32 s[sgprPrefetchIncAB], 0          // Set increment to 0
s_set_vgpr_msb 24414
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+448-256:vgprValuC+448-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch pointers increment
s_set_vgpr_msb 24268
s_wait_alu depctr_sa_sdst(0)
v_add_nc_u64 v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1], s[sgprPrefetchIncAB:sgprPrefetchIncAB+1], v[vgprGL2PrefetchAB-768+0:vgprGL2PrefetchAB-768+1]
s_set_vgpr_msb 52318
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+456-256:vgprValuC+456-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24268
s_set_vgpr_msb 52318
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+464-256:vgprValuC+464-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_not_b32 s90, s90

s_and_b32 s92, s90, s91             // if not last persist iter && last 4 unrolled iters
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+472-256:vgprValuC+472-256+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_wait_alu depctr_sa_sdst(0)
v_cmp_eq_i32 vcc_lo, s92, 1
s_set_vgpr_msb 24238 // 128+2+12+32
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X1_I0+64+0+0-512:vgprValuA_X1_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+480-512:vgprValuC+480-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 44751
v_cndmask_b32 v[vgprGL2PrefetchAB-768+0], v[vgprGL2PrefetchAB-768+0], v[vgprGL2PrefetchABNext-768+0], vcc_lo
v_cndmask_b32 v[vgprGL2PrefetchAB-768+1], v[vgprGL2PrefetchAB-768+1], v[vgprGL2PrefetchABNext-768+1], vcc_lo
s_set_vgpr_msb 53166
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X1_I0+80+0+0-512:vgprValuA_X1_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+488-512:vgprValuC+488-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44751
s_set_vgpr_msb 53167
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X1_I0+96+0+0-768:vgprValuA_X1_I0+96+0+0-768+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+496-512:vgprValuC+496-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X1_I0+112+0+0-768:vgprValuA_X1_I0+112+0+0-768+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], v[vgprValuC+504-512:vgprValuC+504-512+7], 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_mov_b32 s[sgprExitLoopIdx], 3
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==1
s_wait_alu depctr_sa_sdst(0)
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:
s_setreg_IMM32_b32 hwreg(26,0,2), 0                // set expert mode = 0 after unrolled loop
/* Before NLL: Check VGPR.checkin for INT8 LW */
s_and_b32 s5, s[sgprGSU], 0x3fff                   // Restore GSU
s_delay_alu instid0(SALU_CYCLE_1)
label_PrefetchGlobalLastIterEnd:
v_nop                                              // Add v_nop before releasing ValuA/B
v_nop                                              // Add v_nop before releasing ValuA/B
v_nop                                              // Add v_nop before releasing ValuA/B
v_nop                                              // Add v_nop before releasing ValuA/B
v_nop                                              // Add v_nop before releasing ValuA/B
v_nop                                              // Add v_nop before releasing ValuA/B
v_nop                                              // Add v_nop before releasing ValuA/B
v_nop                                              // Add v_nop before releasing ValuA/B

/* Tail: add ValuA/B vgpr buffer [550...998) to pool */

/* Tail: add address/G2L vgpr [998...998) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */

/* local write reset offsets b */
/* Check out VGPR (numG2LA,numG2LB,numG2LMetadata) = (128,128,0) */
.set vgprG2LA_BASE, 550
.set vgprG2LB_BASE, 678
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
; s_and_b32 s[sgprLoopCounterL], 255, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 256
; s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
; s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
; s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
; .set vgprValuA_X0_I0_BASE, UNDEF
; .set vgprValuA_X0_I0, UNDEF
; .set vgprValuA_X1_I0, UNDEF
; .set vgprValuB_X0_I0_BASE, UNDEF
; .set vgprValuB_X0_I0, UNDEF
.set vgprLocalWriteAddr, 15
.set vgprGlobalWriteAddr, 24 // use 8
.set vgprFirstTmp, 1020

/* Tail: add MISC Vgpr [544...550) to pool */

/* Tail: add ValuA/B vgpr buffer [0...32) to pool */
label_Summation_End_S4FDBQ587JJL6NOU:
.set sgprWGM, UNDEF
; .set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
; .set sgprtdmABIncs, UNDEF // need 17
; .set sgprtdmMXSAMXSBIncs, UNDEF // need 18
.set sgprStaggerUIter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressMXSA, UNDEF
.set sgprAddressB, UNDEF
.set sgprAddressMXSB, UNDEF
; .set sgprStridesA, UNDEF  // need 40 41
.set sgprStridesMXSA, UNDEF
; .set sgprStridesB, UNDEF  // need 44 45
.set sgprStridesMXSB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
; .set sgprtdmAGroup0, UNDEF  // need 52~55
; .set sgprtdmAGroup1, UNDEF  // need 56~63
; .set sgprtdmMXSAGroup0, UNDEF  // need 64~67
; .set sgprtdmMXSAGroup1, UNDEF  // need 68~75
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprWrapUMXSA, UNDEF
.set sgprWrapUMXSB, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprGlobalReadIncsMXSA, UNDEF
.set sgprGlobalReadIncsMXSB, UNDEF
/* load store sgprs */
.set sgprSrdC, 32
.set sgprSrdD, 28

// non-persistent: Iter is pinned at the last index, so the store SRD init below must always run
/* Multiply MI out register with Alpha -> C Vgpr register */
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD
// Shift num records for gfx125x
s_and_b32 s5, s[sgprSrdD+2], 127
s_delay_alu instid0(SALU_CYCLE_1)
s_lshl_b32 s5, s5, 25
s_and_b32 s[sgprSrdD+1], s[sgprSrdD+1], 33554431
s_delay_alu instid0(SALU_CYCLE_1)
s_or_b32 s[sgprSrdD+1], s[sgprSrdD+1], s5
s_lshr_b32 s[sgprSrdD+2], s[sgprSrdD+2], 7

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
s_set_vgpr_msb 44812
v_lshrrev_b32 v11, 5, v[vgprSerial-768]             // 4 = Serial / 32
s_set_vgpr_msb 3072
v_lshrrev_b32 v12, 1, v11                            // 5 = 4 / 2
v_mul_lo_u32 v8, 0x10, v12                          // wave coordination offset 1
s_set_vgpr_msb 12
v_and_b32 v12, 15, v[vgprSerial-768]                // v12 = v[vgprSerial-768] % 16
s_set_vgpr_msb 3072
v_add_lshl_u32 v8, v12, v8, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v9, v8, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v10, v8, s[sgprStrideD1J]              //  offset 1
v_mul_lo_u32 v[vgprLocalWriteAddr], v8, 256        // glds
v_and_b32 v12, 1, v11                                // v12 = v11 % 2
v_mul_lo_u32 v12, 0x10, v12                          // wave coordination offset 0
s_set_vgpr_msb 12
v_and_b32 v14, 31, v[vgprSerial-768]                // v14 = v[vgprSerial-768] % 32
s_set_vgpr_msb 3072
v_lshrrev_b32 v14, 4, v14                            // 0 = 0 / 16
v_lshlrev_b32 v14, 3, v14                            // thread0 * continuous_output
v_add_lshl_u32 v14, v12, v14, 0                       // coordination 0 = vwA *(wave_id0 + tid0)
v_add_nc_u32 v[vgprLocalWriteAddr], v[vgprLocalWriteAddr], v14          // glds
v_lshrrev_b32 v12, 8, v[vgprLocalWriteAddr]                            // glds: padding 16 per block 256
v_lshl_add_u32 v[vgprLocalWriteAddr], v12, 0x4, v[vgprLocalWriteAddr]  // glds: Add padding amount
s_delay_alu instid0(NO_DEP)
; s_mul_i32 s5, 256, s[sgprWorkGroup1]               // wgp1 * MT1
; s_delay_alu instid0(NO_DEP)
; v_add_nc_u32 v8, s5, v8                            // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
v_add_nc_u32 v11, v10, v14                         // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
s_set_vgpr_msb 192
v_mov_b32 v[vgprFirstTmp-768+0], v[vgprLocalWriteAddr]
v_mov_b32 v[vgprFirstTmp-768+1], v11

label_skip_first_WG_init:
s_set_vgpr_msb 49155
v_mov_b32 v11, v[vgprFirstTmp-768+1]
s_mul_i32 s5, 256, s[sgprWorkGroup0]               // wgp0 * MT0
s_set_vgpr_msb 768
v_add_nc_u32 v11, v11, s5                            // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
; s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
; s_mov_b32 s[sgprSrdC+2], BufferOOB
; s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD
; // Shift num records for gfx125x
; s_and_b32 s5, s[sgprSrdC+2], 127
; s_delay_alu instid0(SALU_CYCLE_1)
; s_lshl_b32 s5, s5, 25
; s_and_b32 s[sgprSrdC+1], s[sgprSrdC+1], 33554431
; s_delay_alu instid0(SALU_CYCLE_1)
; s_or_b32 s[sgprSrdC+1], s[sgprSrdC+1], s5
; s_lshr_b32 s[sgprSrdC+2], s[sgprSrdC+2], 7


s_mul_i32 s14, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
; s_mul_hi_u32 s13, s14, s[sgprStrideC1J]            // ScaleC s14 by Stride
; s_mul_i32 s12, s14, s[sgprStrideC1J]               // ScaleC s14 by Stride
; s_lshl_b64 s[12:13], s[12:13], s[sgprGSULog2BpeC]  // scale by bpe
; s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s12    // add lo to SRD
; s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s13   // add hi to SRD
s_mul_hi_u32 s13, s14, s[sgprStrideD1J]            // ScaleD s14 by Stride
s_mul_i32 s12, s14, s[sgprStrideD1J]               // ScaleD s14 by Stride
s_lshl_b64 s[12:13], s[12:13], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s12    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s13   // add hi to SRD

; s_mul_hi_u32 s13, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
; s_mul_i32 s12, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
; s_lshl_b64 s[12:13], s[12:13], s[sgprGSULog2BpeC]  // scale by bpe
; s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // add lo to SRD
; s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s13       // add hi to SRD
; s_mul_hi_u32 s13, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
; s_mul_i32 s12, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
; s_lshl_b64 s[12:13], s[12:13], s[sgprGSULog2BpeD]  // scale by bpe
; s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // add lo to SRD
; s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s13       // add hi to SRD

; label_GSU_3:
; .set sgprGSULog2BpeC, UNDEF
; .set sgprAddressC, UNDEF // need 26

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
label_GW_B0_E0_1:
/**** Tony Modify ****/
s_add_u32 s[sgprIter], s[sgprIter], 1
; s_delay_alu instid0(SALU_CYCLE_1)
; s_cmp_ge_u32 s[sgprIter], 4
; s_cbranch_scc1 label_glds

label_glds:
/******************************************/
/* Global Write Batch (glds) #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw8); (0,1,0,0:vw8); (0,2,0,0:vw8); (0,3,0,0:vw8); (0,4,0,0:vw8); (0,5,0,0:vw8); (0,6,0,0:vw8); (0,7,0,0:vw8); (1,0,0,0:vw8); (1,1,0,0:vw8); (1,2,0,0:vw8); (1,3,0,0:vw8); (1,4,0,0:vw8); (1,5,0,0:vw8); (1,6,0,0:vw8); (1,7,0,0:vw8); (2,0,0,0:vw8); (2,1,0,0:vw8); (2,2,0,0:vw8); (2,3,0,0:vw8); (2,4,0,0:vw8); (2,5,0,0:vw8); (2,6,0,0:vw8); (2,7,0,0:vw8); (3,0,0,0:vw8); (3,1,0,0:vw8); (3,2,0,0:vw8); (3,3,0,0:vw8); (3,4,0,0:vw8); (3,5,0,0:vw8); (3,6,0,0:vw8); (3,7,0,0:vw8); (4,0,0,0:vw8); (4,1,0,0:vw8); (4,2,0,0:vw8); (4,3,0,0:vw8); (4,4,0,0:vw8); (4,5,0,0:vw8); (4,6,0,0:vw8); (4,7,0,0:vw8); (5,0,0,0:vw8); (5,1,0,0:vw8); (5,2,0,0:vw8); (5,3,0,0:vw8); (5,4,0,0:vw8); (5,5,0,0:vw8); (5,6,0,0:vw8); (5,7,0,0:vw8); (6,0,0,0:vw8); (6,1,0,0:vw8); (6,2,0,0:vw8); (6,3,0,0:vw8); (6,4,0,0:vw8); (6,5,0,0:vw8); (6,6,0,0:vw8); (6,7,0,0:vw8); (7,0,0,0:vw8); (7,1,0,0:vw8); (7,2,0,0:vw8); (7,3,0,0:vw8) */
/******************************************/
s_wait_dscnt 0
s_barrier_signal -1
s_set_vgpr_msb 3
v_mov_b32 v[vgprLocalWriteAddr], v[vgprFirstTmp-768]
s_barrier_wait -1
/* calc coords, apply mask, and issue loads (if necessary) */
/* apply mask, calc new C and issue writes */

s_set_vgpr_msb 768
v_add_nc_u32 v[vgprLocalWriteAddr], v[vgprLocalWriteAddr], 143872 // add first buffer offset
v_prng_b32 v13, v[vgprValuC+-16]               // Pseudo Random Number Generator
s_delay_alu instid0(VALU_DEP_1)

s_setreg_IMM32_b32 hwreg(26,0,2), 2                // set expert mode = 1

v_cvt_scalef32_sr_pk8_fp8_f32 v[24:25], v[vgprValuC+0:vgprValuC+7], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+0:vgprValuC+1], v[vgprValuC+8:vgprValuC+15], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+2:vgprValuC+3], v[vgprValuC+16:vgprValuC+23], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+4:vgprValuC+5], v[vgprValuC+24:vgprValuC+31], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+6:vgprValuC+7], v[vgprValuC+32:vgprValuC+39], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+8:vgprValuC+9], v[vgprValuC+40:vgprValuC+47], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+10:vgprValuC+11], v[vgprValuC+48:vgprValuC+55], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+12:vgprValuC+13], v[vgprValuC+56:vgprValuC+63], v13, s[sgprAlpha]

s_wait_alu depctr_va_vdst(7)
ds_store_b64 v[vgprLocalWriteAddr], v[24:25] offset:0
s_wait_alu depctr_va_vdst(6)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+0:vgprValuC+1] offset:32
s_wait_alu depctr_va_vdst(5)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+2:vgprValuC+3] offset:64
s_wait_alu depctr_va_vdst(4)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+4:vgprValuC+5] offset:96
s_wait_alu depctr_va_vdst(3)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+6:vgprValuC+7] offset:128
s_wait_alu depctr_va_vdst(2)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+8:vgprValuC+9] offset:160
s_wait_alu depctr_va_vdst(1)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+10:vgprValuC+11] offset:192
s_wait_alu depctr_va_vdst(0)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+12:vgprValuC+13] offset:224
s_wait_alu depctr_vm_vsrc(0)
v_add_nc_u32 v[vgprLocalWriteAddr], 32*(16+256), v[vgprLocalWriteAddr]

v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+14:vgprValuC+15], v[vgprValuC+64:vgprValuC+71], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+16:vgprValuC+17], v[vgprValuC+72:vgprValuC+79], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+18:vgprValuC+19], v[vgprValuC+80:vgprValuC+87], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+20:vgprValuC+21], v[vgprValuC+88:vgprValuC+95], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+22:vgprValuC+23], v[vgprValuC+96:vgprValuC+103], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+24:vgprValuC+25], v[vgprValuC+104:vgprValuC+111], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+26:vgprValuC+27], v[vgprValuC+112:vgprValuC+119], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+28:vgprValuC+29], v[vgprValuC+120:vgprValuC+127], v13, s[sgprAlpha]

s_wait_alu depctr_va_vdst(7)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+14:vgprValuC+15] offset:0
s_wait_alu depctr_va_vdst(6)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+16:vgprValuC+17] offset:32
s_wait_alu depctr_va_vdst(5)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+18:vgprValuC+19] offset:64
s_wait_alu depctr_va_vdst(4)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+20:vgprValuC+21] offset:96
s_wait_alu depctr_va_vdst(3)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+22:vgprValuC+23] offset:128
s_wait_alu depctr_va_vdst(2)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+24:vgprValuC+25] offset:160
s_wait_alu depctr_va_vdst(1)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+26:vgprValuC+27] offset:192
s_wait_alu depctr_va_vdst(0)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+28:vgprValuC+29] offset:224
s_wait_alu depctr_vm_vsrc(0)
v_add_nc_u32 v[vgprLocalWriteAddr], 32*(16+256), v[vgprLocalWriteAddr]


v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+30:vgprValuC+31], v[vgprValuC+128:vgprValuC+135], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+32:vgprValuC+33], v[vgprValuC+136:vgprValuC+143], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+34:vgprValuC+35], v[vgprValuC+144:vgprValuC+151], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+36:vgprValuC+37], v[vgprValuC+152:vgprValuC+159], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+38:vgprValuC+39], v[vgprValuC+160:vgprValuC+167], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+40:vgprValuC+41], v[vgprValuC+168:vgprValuC+175], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+42:vgprValuC+43], v[vgprValuC+176:vgprValuC+183], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+44:vgprValuC+45], v[vgprValuC+184:vgprValuC+191], v13, s[sgprAlpha]

s_wait_alu depctr_va_vdst(7)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+30:vgprValuC+31] offset:0
s_wait_alu depctr_va_vdst(6)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+32:vgprValuC+33] offset:32
s_wait_alu depctr_va_vdst(5)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+34:vgprValuC+35] offset:64
s_wait_alu depctr_va_vdst(4)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+36:vgprValuC+37] offset:96
s_wait_alu depctr_va_vdst(3)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+38:vgprValuC+39] offset:128
s_wait_alu depctr_va_vdst(2)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+40:vgprValuC+41] offset:160
s_wait_alu depctr_va_vdst(1)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+42:vgprValuC+43] offset:192
s_wait_alu depctr_va_vdst(0)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+44:vgprValuC+45] offset:224
s_wait_alu depctr_vm_vsrc(0)
v_add_nc_u32 v[vgprLocalWriteAddr], 32*(16+256), v[vgprLocalWriteAddr]


v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+46:vgprValuC+47], v[vgprValuC+192:vgprValuC+199], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+48:vgprValuC+49], v[vgprValuC+200:vgprValuC+207], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+50:vgprValuC+51], v[vgprValuC+208:vgprValuC+215], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+52:vgprValuC+53], v[vgprValuC+216:vgprValuC+223], v13, s[sgprAlpha]
s_set_vgpr_msb 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+54:vgprValuC+55], v[vgprValuC+224-256:vgprValuC+231-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+56:vgprValuC+57], v[vgprValuC+232-256:vgprValuC+239-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+58:vgprValuC+59], v[vgprValuC+240-256:vgprValuC+247-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+60:vgprValuC+61], v[vgprValuC+248-256:vgprValuC+255-256], v13, s[sgprAlpha]

s_set_vgpr_msb 256
s_wait_alu depctr_va_vdst(7)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+46:vgprValuC+47] offset:0
s_wait_alu depctr_va_vdst(6)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+48:vgprValuC+49] offset:32
s_wait_alu depctr_va_vdst(5)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+50:vgprValuC+51] offset:64
s_wait_alu depctr_va_vdst(4)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+52:vgprValuC+53] offset:96
s_wait_alu depctr_va_vdst(3)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+54:vgprValuC+55] offset:128
s_wait_alu depctr_va_vdst(2)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+56:vgprValuC+57] offset:160
s_wait_alu depctr_va_vdst(1)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+58:vgprValuC+59] offset:192
s_wait_alu depctr_va_vdst(0)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+60:vgprValuC+61] offset:224
s_wait_alu depctr_vm_vsrc(0)
v_add_nc_u32 v[vgprLocalWriteAddr], 32*(16+256), v[vgprLocalWriteAddr]

s_set_vgpr_msb 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+62:vgprValuC+63], v[vgprValuC+256-256:vgprValuC+263-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+64:vgprValuC+65], v[vgprValuC+264-256:vgprValuC+271-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+66:vgprValuC+67], v[vgprValuC+272-256:vgprValuC+279-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+68:vgprValuC+69], v[vgprValuC+280-256:vgprValuC+287-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+70:vgprValuC+71], v[vgprValuC+288-256:vgprValuC+295-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+72:vgprValuC+73], v[vgprValuC+296-256:vgprValuC+303-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+74:vgprValuC+75], v[vgprValuC+304-256:vgprValuC+311-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+76:vgprValuC+77], v[vgprValuC+312-256:vgprValuC+319-256], v13, s[sgprAlpha]

s_set_vgpr_msb 256
s_wait_alu depctr_va_vdst(7)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+62:vgprValuC+63] offset:0
s_wait_alu depctr_va_vdst(6)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+64:vgprValuC+65] offset:32
s_wait_alu depctr_va_vdst(5)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+66:vgprValuC+67] offset:64
s_wait_alu depctr_va_vdst(4)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+68:vgprValuC+69] offset:96
s_wait_alu depctr_va_vdst(3)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+70:vgprValuC+71] offset:128
s_wait_alu depctr_va_vdst(2)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+72:vgprValuC+73] offset:160
s_wait_alu depctr_va_vdst(1)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+74:vgprValuC+75] offset:192
s_wait_alu depctr_va_vdst(0)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+76:vgprValuC+77] offset:224
s_wait_alu depctr_vm_vsrc(0)
v_add_nc_u32 v[vgprLocalWriteAddr], 32*(16+256), v[vgprLocalWriteAddr]

s_set_vgpr_msb 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+78:vgprValuC+79], v[vgprValuC+320-256:vgprValuC+327-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+80:vgprValuC+81], v[vgprValuC+328-256:vgprValuC+335-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+82:vgprValuC+83], v[vgprValuC+336-256:vgprValuC+343-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+84:vgprValuC+85], v[vgprValuC+344-256:vgprValuC+351-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+86:vgprValuC+87], v[vgprValuC+352-256:vgprValuC+359-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+88:vgprValuC+89], v[vgprValuC+360-256:vgprValuC+367-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+90:vgprValuC+91], v[vgprValuC+368-256:vgprValuC+375-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+92:vgprValuC+93], v[vgprValuC+376-256:vgprValuC+383-256], v13, s[sgprAlpha]

s_set_vgpr_msb 256
s_wait_alu depctr_va_vdst(7)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+78:vgprValuC+79] offset:0
s_wait_alu depctr_va_vdst(6)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+80:vgprValuC+81] offset:32
s_wait_alu depctr_va_vdst(5)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+82:vgprValuC+83] offset:64
s_wait_alu depctr_va_vdst(4)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+84:vgprValuC+85] offset:96
s_wait_alu depctr_va_vdst(3)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+86:vgprValuC+87] offset:128
s_wait_alu depctr_va_vdst(2)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+88:vgprValuC+89] offset:160
s_wait_alu depctr_va_vdst(1)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+90:vgprValuC+91] offset:192
s_wait_alu depctr_va_vdst(0)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+92:vgprValuC+93] offset:224
s_wait_alu depctr_vm_vsrc(0)
v_add_nc_u32 v[vgprLocalWriteAddr], 32*(16+256), v[vgprLocalWriteAddr]

s_set_vgpr_msb 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+94:vgprValuC+95], v[vgprValuC+384-256:vgprValuC+391-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+96:vgprValuC+97], v[vgprValuC+392-256:vgprValuC+399-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+98:vgprValuC+99], v[vgprValuC+400-256:vgprValuC+407-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+100:vgprValuC+101], v[vgprValuC+408-256:vgprValuC+415-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+102:vgprValuC+103], v[vgprValuC+416-256:vgprValuC+423-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+104:vgprValuC+105], v[vgprValuC+424-256:vgprValuC+431-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+106:vgprValuC+107], v[vgprValuC+432-256:vgprValuC+439-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+108:vgprValuC+109], v[vgprValuC+440-256:vgprValuC+447-256], v13, s[sgprAlpha]

s_set_vgpr_msb 256
s_wait_alu depctr_va_vdst(7)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+94:vgprValuC+95] offset:0
s_wait_alu depctr_va_vdst(6)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+96:vgprValuC+97] offset:32
s_wait_alu depctr_va_vdst(5)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+98:vgprValuC+99] offset:64
s_wait_alu depctr_va_vdst(4)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+100:vgprValuC+101] offset:96
s_wait_alu depctr_va_vdst(3)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+102:vgprValuC+103] offset:128
s_wait_alu depctr_va_vdst(2)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+104:vgprValuC+105] offset:160
s_wait_alu depctr_va_vdst(1)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+106:vgprValuC+107] offset:192
s_wait_alu depctr_va_vdst(0)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+108:vgprValuC+109] offset:224
s_wait_alu depctr_vm_vsrc(0)
v_add_nc_u32 v[vgprLocalWriteAddr], 32*(16+256), v[vgprLocalWriteAddr]

s_set_vgpr_msb 1
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+110:vgprValuC+111], v[vgprValuC+448-256:vgprValuC+455-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+112:vgprValuC+113], v[vgprValuC+456-256:vgprValuC+463-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+114:vgprValuC+115], v[vgprValuC+464-256:vgprValuC+471-256], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+116:vgprValuC+117], v[vgprValuC+472-256:vgprValuC+479-256], v13, s[sgprAlpha]
s_set_vgpr_msb 258
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+118:vgprValuC+119], v[vgprValuC+480-512:vgprValuC+487-512], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+120:vgprValuC+121], v[vgprValuC+488-512:vgprValuC+495-512], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+122:vgprValuC+123], v[vgprValuC+496-512:vgprValuC+503-512], v13, s[sgprAlpha]
v_cvt_scalef32_sr_pk8_fp8_f32 v[vgprValuC+124:vgprValuC+125], v[vgprValuC+504-512:vgprValuC+511-512], v13, s[sgprAlpha]

s_set_vgpr_msb 512
s_wait_alu depctr_va_vdst(7)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+110:vgprValuC+111] offset:0
s_wait_alu depctr_va_vdst(6)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+112:vgprValuC+113] offset:32
s_wait_alu depctr_va_vdst(5)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+114:vgprValuC+115] offset:64
s_wait_alu depctr_va_vdst(4)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+116:vgprValuC+117] offset:96
s_wait_alu depctr_va_vdst(3)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+118:vgprValuC+119] offset:128
s_wait_alu depctr_va_vdst(2)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+120:vgprValuC+121] offset:160
s_wait_alu depctr_va_vdst(1)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+122:vgprValuC+123] offset:192
s_wait_alu depctr_va_vdst(0)
ds_store_b64 v[vgprLocalWriteAddr], v[vgprValuC+124:vgprValuC+125] offset:224
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst

s_setreg_IMM32_b32 hwreg(26,0,2), 0                // set expert mode = 0

// glds global write
s_set_vgpr_msb 12
v_and_b32 v11, 0xF, v[vgprSerial-768]
s_set_vgpr_msb 3075
v_bfe_u32 v12, v[vgprSerial-768], 4, 1
s_set_vgpr_msb 768
v_lshlrev_b32 v11, 4, v11
v_mad_u32_u24 v[vgprGlobalWriteAddr], v12, s[sgprStrideD1J], v11
v_mad_u32_u24 v11, v12, (256+16), v11
v_add_nc_u32 v11, v11, 143872 // add first buffer offset

s_set_vgpr_msb 0
v_mad_u32_u24 v11, s[sgprWaveId], (64*(256+16)), v11 // row0-15
v_add_nc_u32 v12, (16*(256+16)), v11                 // row15-31
v_add_nc_u32 v9, (32*(256+16)), v11                 // row32-47
v_add_nc_u32 v10, (48*(256+16)), v11                // row48-64

// Global write wave offset
s_mul_i32 s92, s[sgprWaveId], 64
v_mad_u32_u24 v[vgprGlobalWriteAddr+0], s[sgprStrideD1J], s92, v[vgprGlobalWriteAddr+0]
s_mul_i32 s92, 256, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_nc_u32 v[vgprGlobalWriteAddr], s92, v[vgprGlobalWriteAddr]

// Sync all waves
s_wait_dscnt(0)
s_barrier_signal -1

// Create global pointers
s_mul_i32 s92, s[sgprStrideD1J], 2
s_sub_u32 s92, s92, 2*(256+16)
v_add_nc_i32 v[vgprGlobalWriteAddr+1], v[vgprGlobalWriteAddr+0], s92
v_add_nc_i32 v[vgprGlobalWriteAddr+2], v[vgprGlobalWriteAddr+1], s92
v_add_nc_i32 v[vgprGlobalWriteAddr+3], v[vgprGlobalWriteAddr+2], s92
v_add_nc_i32 v[vgprGlobalWriteAddr+4], v[vgprGlobalWriteAddr+3], s92
v_add_nc_i32 v[vgprGlobalWriteAddr+5], v[vgprGlobalWriteAddr+4], s92
v_add_nc_i32 v[vgprGlobalWriteAddr+6], v[vgprGlobalWriteAddr+5], s92
v_add_nc_i32 v[vgprGlobalWriteAddr+7], v[vgprGlobalWriteAddr+6], s92
s_barrier_wait -1 

// Rows 0-15
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+0], v11, s[sgprSrdD+0:sgprSrdD+1] offset:0
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+1], v11, s[sgprSrdD+0:sgprSrdD+1] offset:2*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+2], v11, s[sgprSrdD+0:sgprSrdD+1] offset:4*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+3], v11, s[sgprSrdD+0:sgprSrdD+1] offset:6*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+4], v11, s[sgprSrdD+0:sgprSrdD+1] offset:8*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+5], v11, s[sgprSrdD+0:sgprSrdD+1] offset:10*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+6], v11, s[sgprSrdD+0:sgprSrdD+1] offset:12*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+7], v11, s[sgprSrdD+0:sgprSrdD+1] offset:14*(256+16)

s_mul_i32 s92, s[sgprStrideD1J], 16
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s92         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)

// Rows 16-31
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+0], v12, s[sgprSrdD+0:sgprSrdD+1] offset:0
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+1], v12, s[sgprSrdD+0:sgprSrdD+1] offset:2*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+2], v12, s[sgprSrdD+0:sgprSrdD+1] offset:4*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+3], v12, s[sgprSrdD+0:sgprSrdD+1] offset:6*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+4], v12, s[sgprSrdD+0:sgprSrdD+1] offset:8*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+5], v12, s[sgprSrdD+0:sgprSrdD+1] offset:10*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+6], v12, s[sgprSrdD+0:sgprSrdD+1] offset:12*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+7], v12, s[sgprSrdD+0:sgprSrdD+1] offset:14*(256+16)

s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s92         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)

// Rows 32-47
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+0], v9, s[sgprSrdD+0:sgprSrdD+1] offset:0
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+1], v9, s[sgprSrdD+0:sgprSrdD+1] offset:2*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+2], v9, s[sgprSrdD+0:sgprSrdD+1] offset:4*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+3], v9, s[sgprSrdD+0:sgprSrdD+1] offset:6*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+4], v9, s[sgprSrdD+0:sgprSrdD+1] offset:8*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+5], v9, s[sgprSrdD+0:sgprSrdD+1] offset:10*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+6], v9, s[sgprSrdD+0:sgprSrdD+1] offset:12*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+7], v9, s[sgprSrdD+0:sgprSrdD+1] offset:14*(256+16)

s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s92         // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)

// Rows 48-63
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+0], v10, s[sgprSrdD+0:sgprSrdD+1] offset:0
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+1], v10, s[sgprSrdD+0:sgprSrdD+1] offset:2*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+2], v10, s[sgprSrdD+0:sgprSrdD+1] offset:4*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+3], v10, s[sgprSrdD+0:sgprSrdD+1] offset:6*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+4], v10, s[sgprSrdD+0:sgprSrdD+1] offset:8*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+5], v10, s[sgprSrdD+0:sgprSrdD+1] offset:10*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+6], v10, s[sgprSrdD+0:sgprSrdD+1] offset:12*(256+16)
global_store_async_from_lds_b128 v[vgprGlobalWriteAddr+7], v10, s[sgprSrdD+0:sgprSrdD+1] offset:14*(256+16)

/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */
label_GW_B0_E1_N_1:
label_GW_Beta_1:
label_GW_B0_E1_M_1:
label_GW_End_1:
label_KernelEnd:
s_barrier_wait -3 // cluster wait for the last iteration
s_barrier_signal -1
s_barrier_wait -1
s_cmp_ge_u32 s[sgprIter], 4
s_cbranch_scc1 label_ENDPGM

// calculate loop counter
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizeL], 8 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 256

// update WG Idx
s_mov_b32 s[sgprWorkGroup0], s[sgprWorkGroup0Next]
s_mov_b32 s[sgprWorkGroup1], s[sgprWorkGroup1Next]
s_add_u32 s90, s[sgprIter], 1                    // Next iter
s_and_b32 s90, s90, 1                            // check even/odd iter
s_cselect_b32 s92, 1, -1                         // WG0: odd: 1, even: -1
s_cselect_b32 s90, 0, 1                          // WG1: odd: 0, even: 1
s_lshr_b32 s91, s[sgprNumWorkGroups0], 1
s_mul_i32 s92, s91, s92
s_add_i32 s[sgprWorkGroup0Next], s[sgprWorkGroup0Next], s92
s_lshr_b32 s91, s[sgprNumWorkGroups1], 1
s_mul_i32 s90, s91, s90
s_add_i32 s[sgprWorkGroup1Next], s[sgprWorkGroup1Next], s90

// persist: next WG addr
s_set_vgpr_msb 3
v_readfirstlane_b32 s90, v[vgprSerial-768]         // first tId
s_delay_alu instid0(NO_DEP)
s_lshr_b32 s90, s90, 5                             // wId=fTid // wavelen
s_delay_alu instid0(SALU_CYCLE_1)
s_bitcmp1_b32 s90, 0                               // Check parity of wId
s_cbranch_scc1 label_NextAddrB                   // Jump to B if wId is odd

label_NextAddrA:
s_mul_i32 s91, s[sgprWorkGroup0Next], 256
s_mul_i32 s91, s91, s[sgprStrideA0I]
s_add_u32 s[sgprTDMAddrABNext], s[sgprTDMAddrABNoWG], s91
s_addc_u32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNoWG+1], 0
s_or_b32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNext+1], 0x80000000 // set type field to 2(image)
s_mul_i32 s91, s[sgprWorkGroup0Next], 1024                         // * MT(256) * bpe(1.0)
// gl2
s_mov_b32 s91, 0
s_mul_i32 s90, s[sgprStrideA0I], 256               // stride * MT(256) * bpe(1.0)
s_mul_i32 s90, s90, s[sgprWorkGroup0Next]           // *= wgId)
s_set_vgpr_msb 972
v_add_nc_u64 v[vgprGL2PrefetchABNext-768+0:vgprGL2PrefetchABNext-768+1], s[90:91], v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1]  // Add MT offset
s_mul_i32 s90, s[sgprWorkGroup0Next], 1024             // stride * MT(256) * bpe(1)
s_branch label_NextAddrEnd

label_NextAddrB:
s_mul_i32 s91, s[sgprWorkGroup1Next], 256
s_mul_i32 s91, s91, s[sgprStrideB1J]
s_add_u32 s[sgprTDMAddrABNext], s[sgprTDMAddrABNoWG], s91
s_addc_u32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNoWG+1], 0
s_or_b32 s[sgprTDMAddrABNext+1], s[sgprTDMAddrABNext+1], 0x80000000 // set type field to 2(image)
s_mul_i32 s91, s[sgprWorkGroup1Next], 1024                         // * MT(256) * bpe(1.0)
// gl2
s_mov_b32 s91, 0
s_mul_i32 s90, s[sgprStrideB1J], 256               // stride * MT(256) * bpe(1.0)
s_mul_i32 s90, s90, s[sgprWorkGroup1Next]           // *= wgId)
s_set_vgpr_msb 52428
v_add_nc_u64 v[vgprGL2PrefetchABNext-768+0:vgprGL2PrefetchABNext-768+1], s[90:91], v[vgprGL2PrefetchABNoWG-768+0:vgprGL2PrefetchABNoWG-768+1]  // Add MT offset
s_mul_i32 s90, s[sgprWorkGroup1Next], 1024             // stride * MT(256) * bpe(1)

label_NextAddrEnd:
// cluster sync 
s_barrier_signal -1
s_cmp_eq_u32 s[sgprWaveId], 0
s_barrier_wait -1
s_wait_alu depctr_sa_sdst(0)
s_cbranch_scc0 label_Skip_Signal_13
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_13:
// wait for GW to finish
s_wait_asynccnt 0              // last MT GW
s_barrier_wait -3
s_barrier_signal -1
s_mov_b32 s[sgprtdmAGroup0+0], 1
s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_barrier_wait -1
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]

// cluster sync before loop 
s_cmp_eq_u32 s[sgprWaveId], 0
s_wait_alu depctr_sa_sdst(0)
s_cbranch_scc0 label_Skip_Signal_12
s_barrier_signal -3     // Cluster barrier
label_Skip_Signal_12:

s_cmp_eq_u32 s[sgprExitLoopIdx], 3
s_cbranch_scc1 label_ExitLoop3
s_cmp_eq_u32 s[sgprExitLoopIdx], 2
s_cbranch_scc1 label_ExitLoop2
label_ExitLoop1:
; s_getpc_b64 s[90:91]                               // addr of next instr
; s_add_i32 s92, label_Persist_Start_2, 4                   // target branch offset
; s_delay_alu instid0(SALU_CYCLE_1)
; s_abs_i32 s92, s92
; s_delay_alu instid0(SALU_CYCLE_1)
; s_sub_u32 s90, s90, s92
; s_subb_u32 s91, s91, 0
; s_nop 6
; s_delay_alu instid0(SALU_CYCLE_1)
s_nop 0
s_setreg_IMM32_b32 hwreg(26,0,2), 2                // WA: set expert mode = 1 in unrolled loop

// set vgprValuB idx
.set vgprValuB_G0, vgprValuB_Y1+0
.set vgprValuB_G1, vgprValuB_Y2+0
.set vgprValuB_G2, vgprValuB_Y0+0
/* iter 0 */
s_set_vgpr_msb 52238
s_wait_dscnt 40 // MX + half A + half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:128 
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:160
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:192
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:224
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17568
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17600
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17632 
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26304 
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+0-512:vgprValuA_X1_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:128 
ds_load_b128 v[vgprValuA_X1_I0+4-512:vgprValuA_X1_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:160
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+8-512:vgprValuA_X1_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:192 
ds_load_b128 v[vgprValuA_X1_I0+12-512:vgprValuA_X1_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:224
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+16-512:vgprValuA_X1_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8832 
ds_load_b128 v[vgprValuA_X1_I0+20-512:vgprValuA_X1_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+24-512:vgprValuA_X1_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X1_I0+28-512:vgprValuA_X1_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 33294
s_wait_dscnt 40 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+32-512:vgprValuA_X1_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X1_I0+36-512:vgprValuA_X1_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17568
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+40-512:vgprValuA_X1_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17600
ds_load_b128 v[vgprValuA_X1_I0+44-512:vgprValuA_X1_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17632
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+48-512:vgprValuA_X1_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26240 
ds_load_b128 v[vgprValuA_X1_I0+52-512:vgprValuA_X1_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+56-512:vgprValuA_X1_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X1_I0+60-512:vgprValuA_X1_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+64-512:vgprValuA_X1_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34944 
ds_load_b128 v[vgprValuA_X1_I0+68-512:vgprValuA_X1_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34976
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+72-512:vgprValuA_X1_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:35008 
ds_load_b128 v[vgprValuA_X1_I0+76-512:vgprValuA_X1_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:35040
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+80-512:vgprValuA_X1_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43648 
ds_load_b128 v[vgprValuA_X1_I0+84-512:vgprValuA_X1_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+88-512:vgprValuA_X1_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43712
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+92-768:vgprValuA_X1_I0+92-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
ds_load_b128 v[vgprValuA_X1_I0+96-768:vgprValuA_X1_I0+96-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+100-768:vgprValuA_X1_I0+100-768+3], v[vgprLocalReadAddrA+0-512] offset:52384 
ds_load_b128 v[vgprValuA_X1_I0+104-768:vgprValuA_X1_I0+104-768+3], v[vgprLocalReadAddrA+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+108-768:vgprValuA_X1_I0+108-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X1_I0+112-768:vgprValuA_X1_I0+112-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+116-768:vgprValuA_X1_I0+116-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 49758
s_wait_dscnt 46 // another half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+120-768:vgprValuA_X1_I0+120-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+124-768:vgprValuA_X1_I0+124-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+0-768:vgprValuB_G0+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34944  
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+4-768:vgprValuB_G0+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34976
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+8-768:vgprValuB_G0+8-768+3], v[vgprLocalReadAddrB+0-512] offset:35008 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+12-768:vgprValuB_G0+12-768+3], v[vgprLocalReadAddrB+0-512] offset:35040
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+16-768:vgprValuB_G0+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43648
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+20-768:vgprValuB_G0+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43680 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+24-768:vgprValuB_G0+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43712 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258  
ds_load_b128 v[vgprValuB_G0+28-768:vgprValuB_G0+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+32-768:vgprValuB_G0+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+36-768:vgprValuB_G0+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+40-768:vgprValuB_G0+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+44-768:vgprValuB_G0+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+48-768:vgprValuB_G0+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+52-768:vgprValuB_G0+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+56-768:vgprValuB_G0+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch
s_set_vgpr_msb 24067 // global vaddr = src0
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null  scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch
s_set_vgpr_msb 862
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+60-768:vgprValuB_G0+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_cmp_eq_i32 s[sgprIter], 3

s_cselect_b32 s92, 0, 1
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 2

s_cmov_b32 s[sgprtdmAGroup0+0], s92                       // Set TDM as NULL in tail loops
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1]            // update TDM to next AB addr
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 1

s_cmov_b32 s[sgprtdmAGroup0+0], 0
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 24238
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8

s_branch label_Persist_Start_2

label_ExitLoop2:
; s_getpc_b64 s[90:91]                               // addr of next instr
; s_add_i32 s92, label_Persist_Start_3, 4                   // target branch offset
; s_delay_alu instid0(SALU_CYCLE_1)
; s_abs_i32 s92, s92
; s_delay_alu instid0(SALU_CYCLE_1)
; s_sub_u32 s90, s90, s92
; s_subb_u32 s91, s91, 0
; s_nop 6
; s_delay_alu instid0(SALU_CYCLE_1)
s_nop 0
s_setreg_IMM32_b32 hwreg(26,0,2), 2                // WA: set expert mode = 1 in unrolled loop

// set vgprValuB idx
.set vgprValuB_G0, vgprValuB_Y2
.set vgprValuB_G1, vgprValuB_Y0
.set vgprValuB_G2, vgprValuB_Y1
/* iter 0 */
s_set_vgpr_msb 44558
s_wait_dscnt 40 // MX + half A + half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+0-768:vgprValuB_G2+0-768+3], v[vgprLocalReadAddrB+0-512] offset:128 
ds_load_b128 v[vgprValuB_G2+4-768:vgprValuB_G2+4-768+3], v[vgprLocalReadAddrB+0-512] offset:160
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+8-768:vgprValuB_G2+8-768+3], v[vgprLocalReadAddrB+0-512] offset:192
ds_load_b128 v[vgprValuB_G2+12-768:vgprValuB_G2+12-768+3], v[vgprLocalReadAddrB+0-512] offset:224
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+16-768:vgprValuB_G2+16-768+3], v[vgprLocalReadAddrB+0-512] offset:8832
ds_load_b128 v[vgprValuB_G2+20-768:vgprValuB_G2+20-768+3], v[vgprLocalReadAddrB+0-512] offset:8864
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+24-768:vgprValuB_G2+24-768+3], v[vgprLocalReadAddrB+0-512] offset:8896
ds_load_b128 v[vgprValuB_G2+28-768:vgprValuB_G2+28-768+3], v[vgprLocalReadAddrB+0-512] offset:8928
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+32-768:vgprValuB_G2+32-768+3], v[vgprLocalReadAddrB+0-512] offset:17536
ds_load_b128 v[vgprValuB_G2+36-768:vgprValuB_G2+36-768+3], v[vgprLocalReadAddrB+0-512] offset:17568
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+40-768:vgprValuB_G2+40-768+3], v[vgprLocalReadAddrB+0-512] offset:17600
ds_load_b128 v[vgprValuB_G2+44-768:vgprValuB_G2+44-768+3], v[vgprLocalReadAddrB+0-512] offset:17632 
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+48-768:vgprValuB_G2+48-768+3], v[vgprLocalReadAddrB+0-512] offset:26240
ds_load_b128 v[vgprValuB_G2+52-768:vgprValuB_G2+52-768+3], v[vgprLocalReadAddrB+0-512] offset:26272
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3778
ds_load_b128 v[vgprValuB_G2+56-768:vgprValuB_G2+56-768+3], v[vgprLocalReadAddrB+0-512] offset:26304 
ds_load_b128 v[vgprValuB_G2+60-768:vgprValuB_G2+60-768+3], v[vgprLocalReadAddrB+0-512] offset:26336
s_set_vgpr_msb 49678
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+0-512:vgprValuA_X1_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:128 
ds_load_b128 v[vgprValuA_X1_I0+4-512:vgprValuA_X1_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:160
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+8-512:vgprValuA_X1_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:192 
ds_load_b128 v[vgprValuA_X1_I0+12-512:vgprValuA_X1_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:224
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+16-512:vgprValuA_X1_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:8832 
ds_load_b128 v[vgprValuA_X1_I0+20-512:vgprValuA_X1_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:8864
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+24-512:vgprValuA_X1_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:8896
ds_load_b128 v[vgprValuA_X1_I0+28-512:vgprValuA_X1_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:8928
s_set_vgpr_msb 33294
s_wait_dscnt 40 // another half A
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+32-512:vgprValuA_X1_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:17536
ds_load_b128 v[vgprValuA_X1_I0+36-512:vgprValuA_X1_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:17568
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+40-512:vgprValuA_X1_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:17600
ds_load_b128 v[vgprValuA_X1_I0+44-512:vgprValuA_X1_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:17632
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+48-512:vgprValuA_X1_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:26240 
ds_load_b128 v[vgprValuA_X1_I0+52-512:vgprValuA_X1_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:26272
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+0+0+0-768:vgprValuB_G0+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+56-512:vgprValuA_X1_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:26304
ds_load_b128 v[vgprValuA_X1_I0+60-512:vgprValuA_X1_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:26336
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+16+0+0-768:vgprValuB_G0+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3586
s_set_vgpr_msb 526
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+64-512:vgprValuA_X1_I0+64-512+3], v[vgprLocalReadAddrA+0-512] offset:34944 
ds_load_b128 v[vgprValuA_X1_I0+68-512:vgprValuA_X1_I0+68-512+3], v[vgprLocalReadAddrA+0-512] offset:34976
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+72-512:vgprValuA_X1_I0+72-512+3], v[vgprLocalReadAddrA+0-512] offset:35008 
ds_load_b128 v[vgprValuA_X1_I0+76-512:vgprValuA_X1_I0+76-512+3], v[vgprLocalReadAddrA+0-512] offset:35040
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+80-512:vgprValuA_X1_I0+80-512+3], v[vgprLocalReadAddrA+0-512] offset:43648 
ds_load_b128 v[vgprValuA_X1_I0+84-512:vgprValuA_X1_I0+84-512+3], v[vgprLocalReadAddrA+0-512] offset:43680
s_set_vgpr_msb 33294
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+32+0+0-768:vgprValuB_G0+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_a_reuse
s_set_vgpr_msb 3714
ds_load_b128 v[vgprValuA_X1_I0+88-512:vgprValuA_X1_I0+88-512+3], v[vgprLocalReadAddrA+0-512] offset:43712
s_set_vgpr_msb 33374
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+248-256:vgprValuC+248-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+92-768:vgprValuA_X1_I0+92-768+3], v[vgprLocalReadAddrA+0-512] offset:43744
ds_load_b128 v[vgprValuA_X1_I0+96-768:vgprValuA_X1_I0+96-768+3], v[vgprLocalReadAddrA+0-512] offset:52352
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+240-256:vgprValuC+240-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+100-768:vgprValuA_X1_I0+100-768+3], v[vgprLocalReadAddrA+0-512] offset:52384 
ds_load_b128 v[vgprValuA_X1_I0+104-768:vgprValuA_X1_I0+104-768+3], v[vgprLocalReadAddrA+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+232-256:vgprValuC+232-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+108-768:vgprValuA_X1_I0+108-768+3], v[vgprLocalReadAddrA+0-512] offset:52448
ds_load_b128 v[vgprValuA_X1_I0+112-768:vgprValuA_X1_I0+112-768+3], v[vgprLocalReadAddrA+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+224-256:vgprValuC+224-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G0+48+0+0-768:vgprValuB_G0+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuA_X1_I0+116-768:vgprValuA_X1_I0+116-768+3], v[vgprLocalReadAddrA+0-512] offset:61088
s_set_vgpr_msb 49758
s_wait_dscnt 46 // another half B
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_set_vgpr_msb 24158
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+120-768:vgprValuA_X1_I0+120-768+3], v[vgprLocalReadAddrA+0-512] offset:61120
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuA_X1_I0+124-768:vgprValuA_X1_I0+124-768+3], v[vgprLocalReadAddrA+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+0-768:vgprValuB_G0+0-768+3], v[vgprLocalReadAddrB+0-512] offset:34944  
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+4-768:vgprValuB_G0+4-768+3], v[vgprLocalReadAddrB+0-512] offset:34976
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+8-768:vgprValuB_G0+8-768+3], v[vgprLocalReadAddrB+0-512] offset:35008 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+0+0+0-768:vgprValuB_G1+0+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+12-768:vgprValuB_G0+12-768+3], v[vgprLocalReadAddrB+0-512] offset:35040
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+16-768:vgprValuB_G0+16-768+3], v[vgprLocalReadAddrB+0-512] offset:43648
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+20-768:vgprValuB_G0+20-768+3], v[vgprLocalReadAddrB+0-512] offset:43680 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+24-768:vgprValuB_G0+24-768+3], v[vgprLocalReadAddrB+0-512] offset:43712 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258  
ds_load_b128 v[vgprValuB_G0+28-768:vgprValuB_G0+28-768+3], v[vgprLocalReadAddrB+0-512] offset:43744
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+32-768:vgprValuB_G0+32-768+3], v[vgprLocalReadAddrB+0-512] offset:52352 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+36-768:vgprValuB_G0+36-768+3], v[vgprLocalReadAddrB+0-512] offset:52384
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+40-768:vgprValuB_G0+40-768+3], v[vgprLocalReadAddrB+0-512] offset:52416 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+16+0+0-768:vgprValuB_G1+16+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_set_vgpr_msb 24258 
ds_load_b128 v[vgprValuB_G0+44-768:vgprValuB_G0+44-768+3], v[vgprLocalReadAddrB+0-512] offset:52448
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+48-768:vgprValuB_G0+48-768+3], v[vgprLocalReadAddrB+0-512] offset:61056
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+52-768:vgprValuB_G0+52-768+3], v[vgprLocalReadAddrB+0-512] offset:61088
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+56-768:vgprValuB_G0+56-768+3], v[vgprLocalReadAddrB+0-512] offset:61120 
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
// GL2 prefetch
s_set_vgpr_msb 24067 // global vaddr = src0
global_prefetch_b8 v[vgprGL2PrefetchAB-768], null  scope:SCOPE_SE th:TH_LOAD_NT // GL2 Prefetch
s_set_vgpr_msb 862
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 24258
ds_load_b128 v[vgprValuB_G0+60-768:vgprValuB_G0+60-768+3], v[vgprLocalReadAddrB+0-512] offset:61152
s_set_vgpr_msb 49758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_sub_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprTDMGlobalSplitA] // TDM split
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+32+0+0-768:vgprValuB_G1+32+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8
s_cmp_eq_i32 s[sgprIter], 3

s_cselect_b32 s92, 0, 1
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 2

s_cmov_b32 s[sgprtdmAGroup0+0], s92                       // Set TDM as NULL in tail loops
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprTDMAddrABNext:sgprTDMAddrABNext+1]            // update TDM to next AB addr
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_cmp_eq_i32 s[sgprLoopCounterL], 1

s_cmov_b32 s[sgprtdmAGroup0+0], 0
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMSplitA] // TDM split
s_set_vgpr_msb 24238
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+480-512:vgprValuC+480-512+7], v[vgprValuA_X0_I0+64+0+0-512:vgprValuA_X0_I0+64+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse

s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s[sgprTDMAddrSwapA]
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+488-512:vgprValuC+488-512+7], v[vgprValuA_X0_I0+80+0+0-512:vgprValuA_X0_I0+80+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA-512], v[vgprLocalReadSwapAddrA-768], v[vgprLocalReadAddrA-512] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB-512], v[vgprLocalReadSwapAddrB-768], v[vgprLocalReadAddrB-512] // swap Red Blk
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+496-512:vgprValuC+496-512+7], v[vgprValuA_X0_I0+96+0+0-512:vgprValuA_X0_I0+96+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8 matrix_b_reuse
s_set_vgpr_msb 44683
s_set_vgpr_msb 35758
v_wmma_scale_f32_16x16x128_f8f6f4 v[vgprValuC+504-512:vgprValuC+504-512+7], v[vgprValuA_X0_I0+112+0+0-512:vgprValuA_X0_I0+112+0+0-512+15], v[vgprValuB_G1+48+0+0-768:vgprValuB_G1+48+0+0-768+15], 0, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8

s_branch label_Persist_Start_3

label_ExitLoop3:
; s_getpc_b64 s[90:91]                               // addr of next instr
; s_add_i32 s92, label_Persist_Start_1, 4                   // target branch offset
; s_delay_alu instid0(SALU_CYCLE_1)
; s_abs_i32 s92, s92
; s_delay_alu instid0(SALU_CYCLE_1)
; s_sub_u32 s90, s90, s92
; s_subb_u32 s91, s91, 0
; s_nop 6

s_branch label_Persist_Start_1

label_ENDPGM:
s_wait_asynccnt(0)
s_endpgm

label_ASM_End:  /// The end of the kernel
