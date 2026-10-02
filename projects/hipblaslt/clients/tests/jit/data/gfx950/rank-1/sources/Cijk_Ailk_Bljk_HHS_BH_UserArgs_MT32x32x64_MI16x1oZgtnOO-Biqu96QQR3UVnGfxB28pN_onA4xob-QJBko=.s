
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA1_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGMXCC1
.globl Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA1_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGMXCC1
.p2align 8
.type Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA1_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGMXCC1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA1_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGMXCC1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 40 // accvgpr offset
  .amdhsa_next_free_vgpr 44 // vgprs
  .amdhsa_next_free_sgpr 70 // sgprs
  .amdhsa_group_segment_fixed_size 25856 // lds bytes
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
/* Num VGPR   =39 */
/* Num AccVGPR=4 */
/* Num SGPR   =70 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 4 x 1 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 3
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA1_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGMXCC1
    .symbol: 'Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA1_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGMXCC1.kd'
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
      - .name:            A
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideA0
        .size:            4
        .offset:          48
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          52
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      f32
      - .name:            D
        .size:            8
        .offset:          72
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          80
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            batchOffsetD
        .size:            8
        .offset:          104
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          112
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          120
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          128
        .value_kind:      by_value
        .value_type:      u64
    .group_segment_fixed_size:   25856
    .kernarg_segment_align:      8
    .kernarg_segment_size:       136
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 70
    .sgpr_spill_count:           0
    .vgpr_count:                 39
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS1_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR1_PLR1_PKA1_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1_WGMXCC1:
label_ASM_Start:  /// Main body of the asm kernel

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
.set vgprBase, 6
.set vgprLocalWriteAddrA, 2
.set vgprLocalWriteAddrB, 3
.set vgprGlobalReadOffsetA, 0
.set vgprGlobalReadOffsetB, 1
.set vgprLocalReadAddrA, 4
.set vgprLocalReadAddrB, 5
.set vgprSerial, 38

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuA_X0_I0_D0_PACK, vgprBase+8
.set vgprValuB_X0_I0_BASE, vgprBase+16
.set vgprG2LA_BASE, vgprBase+24
.set vgprG2LB_BASE, vgprBase+28
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+4
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+4
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuA_X1_I0_D1, vgprValuA_X0_I0_D0_PACK+4
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
.set sgprAddressA, 28
.set sgprAddressB, 30
.set sgprStridesA, 32
.set sgprStridesB, 34
.set sgprAlpha, 36
.set sgprBeta, 37
.set sgprAddressD, 38
.set sgprAddressC, 40
.set sgprStridesD, 42
.set sgprStridesC, 44
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
.set DepthU, 64
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
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

/* Global Offset B */

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
s_mov_b32 m0, 0x6500                               // LDS clamp at 25856 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

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
v_cvt_f64_u32 v[12:13], s53                        // s49 = s[sgprWorkGroup0] / s53
v_rcp_f64 v[12:13], v[12:13]                       // s49 = s[sgprWorkGroup0] / s53
v_cvt_f64_u32 v[14:15], s[sgprWorkGroup0]          // s49 = s[sgprWorkGroup0] / s53
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s49 = s[sgprWorkGroup0] / s53
v_cvt_u32_f64 v12, v[12:13]                        // s49 = s[sgprWorkGroup0] / s53
v_mul_lo_u32 v13, v12, s53                         // s49 = s[sgprWorkGroup0] / s53
v_sub_u32 v14, s[sgprWorkGroup0], v13              // s49 = s[sgprWorkGroup0] / s53
v_cmpx_ge_u32 exec, v14, s53                       // s49 = s[sgprWorkGroup0] / s53
v_add_u32 v12, v12, 1                              // s49 = s[sgprWorkGroup0] / s53
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v13, v12, s53                         // s49 = s[sgprWorkGroup0] / s53
v_sub_u32 v14, s[sgprWorkGroup0], v13              // s49 = s[sgprWorkGroup0] / s53
v_readfirstlane_b32 s49, v12                       // quotient
v_readfirstlane_b32 s50, v14                       // remainder
s_mul_i32 s49, s49, s53
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s50, s50, s52
/* temp0 = temp0 + temp1 */
s_add_u32 s49, s49, s50
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[12:13], s53                        // s50 = s23 / s53
v_rcp_f64 v[12:13], v[12:13]                       // s50 = s23 / s53
v_cvt_f64_u32 v[14:15], s23                        // s50 = s23 / s53
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s50 = s23 / s53
v_cvt_u32_f64 v12, v[12:13]                        // s50 = s23 / s53
v_mul_lo_u32 v13, v12, s53                         // s50 = s23 / s53
v_sub_u32 v14, s23, v13                            // s50 = s23 / s53
v_cmpx_ge_u32 exec, v14, s53                       // s50 = s23 / s53
v_add_u32 v12, v12, 1                              // s50 = s23 / s53
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s50, v12                       // quotient
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
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [6...36) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...4) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v7, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v6, 15, v7                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v7, 4, v7                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v6, v7, 8, v6                       // 5. K offset: lrKOffset = kIdx * mStride(256); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v10, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v10, 1, v10                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v6, v10, 4, v6                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(16); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v8, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v7, 15, v8                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v7, 6, v7                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v8, 4, v8                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v7, v8, 3, v7                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v9, 7, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v9, 1, v9                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v7, v9, 10, v7                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v8, 6, v[vgprSerial]                 // 8 = Serial / 64
v_lshrrev_b32 v8, 2, v8                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 2048                                // LSU offset: stride = lsuStride(64)*(MT0(32) + PAD0(0))
v_mul_lo_u32 v8, s16, v8                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v8, v6            // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v9, 9, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 512
v_lshl_add_u32 v[vgprLocalReadAddrA], v9, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 512

/* local read addresses: final offsets b */
v_lshrrev_b32 v6, 6, v[vgprSerial]                 // 6 = Serial / 64
v_lshrrev_b32 v6, 2, v6                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v6, s16, v6                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v6, v7            // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v8, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v8, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x1100, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 4 */
/* v7 = A-unroll = serial/LVCA */
v_lshrrev_b32 v7, 2, v[vgprSerial]                 // 7 = Serial / 4
v_and_b32 v6, 3, v[vgprSerial]                     // 6 = Serial % 4
/* tile *= glvw */
v_lshlrev_b32 v6, 3, v6                            // v6 = v6 * 8
v_mov_b32 v10, v7                                  // copy for GlobalSplitU
/* LVCB = 8 */
/* v9 = B-unroll = serial%LVCB */
v_lshrrev_b32 v8, 3, v[vgprSerial]                 // 8 = Serial / 8
v_and_b32 v9, 7, v[vgprSerial]                     // 9 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v9, 3, v9                            // v9 = v9 * 8
v_mov_b32 v11, v9                                  // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v10 */
/* lwaUnrollAssignmentB = v11 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x20, v10    // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v12, 9, v[vgprLocalWriteAddrA]       // padding 32 per block 512
v_lshl_add_u32 v[vgprLocalWriteAddrA], v12, 5, v[vgprLocalWriteAddrA] // padding 32 per block 512

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v8     // lwBL**(DepthU_Compute + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v11, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v12, 7, v[vgprLocalWriteAddrB]       // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v12, 5, v[vgprLocalWriteAddrB] // padding 32 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x1100, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=4352
v_mov_b32 v14, MT0                                 // set MT0 into sgpr
v_mov_b32 v13, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v12, v14                             // v12 = ceil(v13 / v14)
v_rcp_iflag_f32 v12, v12                           // v12 = ceil(v13 / v14)
v_cvt_f32_u32 v15, v13                             // v12 = ceil(v13 / v14)
v_mul_f32 v12, v12, v15                            // v12 = ceil(v13 / v14)
v_cvt_u32_f32 v12, v12                             // v12 = ceil(v13 / v14)
v_mul_u32_u24 v15, v12, v14                        // v12 = ceil(v13 / v14)
v_sub_u32 v15, v13, v15                            // v12 = ceil(v13 / v14)
v_cmp_ne_u32 vcc, v15, 0                           // v12 = ceil(v13 / v14)
v_addc_co_u32 v12, vcc, v12, 0, vcc                // ceil
v_mov_b32 v14, MT1                                 // set MT1 into sgpr
v_mov_b32 v13, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v12     // set back to numWorkGroup0
v_cvt_f32_u32 v12, v14                             // v12 = ceil(v13 / v14)
v_rcp_iflag_f32 v12, v12                           // v12 = ceil(v13 / v14)
v_cvt_f32_u32 v15, v13                             // v12 = ceil(v13 / v14)
v_mul_f32 v12, v12, v15                            // v12 = ceil(v13 / v14)
v_cvt_u32_f32 v12, v12                             // v12 = ceil(v13 / v14)
v_mul_u32_u24 v15, v12, v14                        // v12 = ceil(v13 / v14)
v_sub_u32 v15, v13, v15                            // v12 = ceil(v13 / v14)
v_cmp_ne_u32 vcc, v15, 0                           // v12 = ceil(v13 / v14)
v_addc_co_u32 v12, vcc, v12, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v12     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 60/0 bytes of kern args over preload

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v12, s16                             // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v12, v12                           // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v13, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v12, v12, v13                            // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v12, v12                             // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v13, v12, s16                        // s16 = s[sgprWorkGroup0] / s16
v_sub_u32 v13, s[sgprWorkGroup0], v13              // s16 = s[sgprWorkGroup0] / s16
v_cmpx_eq_u32 exec, v13, s16                       // s16 = s[sgprWorkGroup0] / s16
v_add_u32 v12, 1, v12                              // s16 = s[sgprWorkGroup0] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s16                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s16, v12                       // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v12, s[sgprNumWorkGroups0]           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v12, v12                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v13, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v12, v12, v13                            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v12, v12                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v13, v12, s[sgprNumWorkGroups0]      // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v13, s[sgprWorkGroup0], v13              // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v13, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v12, 1, v12                              // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s16, v12                       // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 88                                  // KernArgAddressOffset
s_mul_i32 s54, s20, 4
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 228
s_mov_b32 s54, 0
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s55, 0
s_load_dwordx4 s[24:27], s[48:49], s54
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
s_and_b32 s53, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s52, s52, s53
s_add_u32 s55, s55, s52
s_cmp_lt_u32 s[sgprWorkGroup0], s55
s_cbranch_scc1 label_FOUND
s_add_u32 s54, s54, s15
s_load_dwordx4 s[24:27], s[48:49], s54
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
s_and_b32 s48, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s52, s52, s48
s_add_u32 s55, s55, s52

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s49, s14, 1
s_sub_u32 s48, s55, s52
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 88                             // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_dwordx2 s[sgprAddressD:sgprAddressD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_dwordx2 s[sgprAddressC:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18
s_load_dwordx2 s[sgprAddressA:sgprAddressA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20
s_load_dwordx2 s[sgprAddressB:sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28
s_load_dwordx2 s[sgprStridesD:sgprStridesD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_dwordx2 s[sgprStridesC:sgprStridesC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38
s_load_dwordx2 s[sgprStridesA:sgprStridesA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[sgprStridesB:sgprStridesB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48
s_load_dword s[sgprAlpha], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
// Read Beta
s_load_dword s37, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
label_LoadExternalStructEnd:
/* init: add vgpr [6...36) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...4) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v7, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v6, 15, v7                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v7, 4, v7                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v6, v7, 8, v6                       // 5. K offset: lrKOffset = kIdx * mStride(256); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v10, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v10, 1, v10                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v6, v10, 4, v6                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(16); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v8, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v7, 15, v8                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v7, 6, v7                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v8, 4, v8                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v7, v8, 3, v7                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v9, 7, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v9, 1, v9                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v7, v9, 10, v7                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(1024); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v8, 6, v[vgprSerial]                 // 8 = Serial / 64
v_lshrrev_b32 v8, 2, v8                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 2048                                // LSU offset: stride = lsuStride(64)*(MT0(32) + PAD0(0))
v_mul_lo_u32 v8, s16, v8                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v8, v6            // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v9, 9, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 512
v_lshl_add_u32 v[vgprLocalReadAddrA], v9, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 512

/* local read addresses: final offsets b */
v_lshrrev_b32 v6, 6, v[vgprSerial]                 // 6 = Serial / 64
v_lshrrev_b32 v6, 2, v6                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v6, s16, v6                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v6, v7            // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v8, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v8, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x1100, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 4 */
/* v7 = A-unroll = serial/LVCA */
v_lshrrev_b32 v7, 2, v[vgprSerial]                 // 7 = Serial / 4
v_and_b32 v6, 3, v[vgprSerial]                     // 6 = Serial % 4
/* tile *= glvw */
v_lshlrev_b32 v6, 3, v6                            // v6 = v6 * 8
v_mov_b32 v10, v7                                  // copy for GlobalSplitU
/* LVCB = 8 */
/* v9 = B-unroll = serial%LVCB */
v_lshrrev_b32 v8, 3, v[vgprSerial]                 // 8 = Serial / 8
v_and_b32 v9, 7, v[vgprSerial]                     // 9 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v9, 3, v9                            // v9 = v9 * 8
v_mov_b32 v11, v9                                  // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v10 */
/* lwaUnrollAssignmentB = v11 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x20, v10    // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v12, 9, v[vgprLocalWriteAddrA]       // padding 32 per block 512
v_lshl_add_u32 v[vgprLocalWriteAddrA], v12, 5, v[vgprLocalWriteAddrA] // padding 32 per block 512

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v8     // lwBL**(DepthU_Compute + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v11, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v12, 7, v[vgprLocalWriteAddrB]       // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v12, 5, v[vgprLocalWriteAddrB] // padding 32 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x1100, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=4352
v_mov_b32 v14, MT0                                 // set MT0 into sgpr
v_mov_b32 v13, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v12, v14                             // v12 = ceil(v13 / v14)
v_rcp_iflag_f32 v12, v12                           // v12 = ceil(v13 / v14)
v_cvt_f32_u32 v15, v13                             // v12 = ceil(v13 / v14)
v_mul_f32 v12, v12, v15                            // v12 = ceil(v13 / v14)
v_cvt_u32_f32 v12, v12                             // v12 = ceil(v13 / v14)
v_mul_u32_u24 v15, v12, v14                        // v12 = ceil(v13 / v14)
v_sub_u32 v15, v13, v15                            // v12 = ceil(v13 / v14)
v_cmp_ne_u32 vcc, v15, 0                           // v12 = ceil(v13 / v14)
v_addc_co_u32 v12, vcc, v12, 0, vcc                // ceil
v_mov_b32 v14, MT1                                 // set MT1 into sgpr
v_mov_b32 v13, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v12     // set back to numWorkGroup0
v_cvt_f32_u32 v12, v14                             // v12 = ceil(v13 / v14)
v_rcp_iflag_f32 v12, v12                           // v12 = ceil(v13 / v14)
v_cvt_f32_u32 v15, v13                             // v12 = ceil(v13 / v14)
v_mul_f32 v12, v12, v15                            // v12 = ceil(v13 / v14)
v_cvt_u32_f32 v12, v12                             // v12 = ceil(v13 / v14)
v_mul_u32_u24 v15, v12, v14                        // v12 = ceil(v13 / v14)
v_sub_u32 v15, v13, v15                            // v12 = ceil(v13 / v14)
v_cmp_ne_u32 vcc, v15, 0                           // v12 = ceil(v13 / v14)
v_addc_co_u32 v12, vcc, v12, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v12     // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 60/0 bytes of kern args over preload

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v12, s16                             // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v12, v12                           // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v13, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v12, v12, v13                            // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v12, v12                             // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v13, v12, s16                        // s16 = s[sgprWorkGroup0] / s16
v_sub_u32 v13, s[sgprWorkGroup0], v13              // s16 = s[sgprWorkGroup0] / s16
v_cmpx_eq_u32 exec, v13, s16                       // s16 = s[sgprWorkGroup0] / s16
v_add_u32 v12, 1, v12                              // s16 = s[sgprWorkGroup0] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s16                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s16, v12                       // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v12, s[sgprNumWorkGroups0]           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v12, v12                           // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v13, s[sgprWorkGroup0]               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v12, v12, v13                            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v12, v12                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v13, v12, s[sgprNumWorkGroups0]      // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v13, s[sgprWorkGroup0], v13              // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v13, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v12, 1, v12                              // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s16, v12                       // quotient
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
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
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

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s16, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
v_cvt_f32_u32 v12, s16                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_rcp_iflag_f32 v12, v12                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_f32_u32 v13, s[sgprWorkGroup1]               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_f32 v12, v12, v13                            // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_u32_f32 v12, v12                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_u32_u24 v13, v12, s16                        // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_sub_u32 v13, s[sgprWorkGroup1], v13              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cmpx_eq_u32 exec, v13, s16                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_add_u32 v12, 1, v12                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mov_b32 v13, 0                                   // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s16
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s16                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s16                        // re-calculate remainder
v_sub_u32 v13, s[sgprWorkGroup1], v13              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v12         // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v13          // remainder
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
v_cvt_f32_u32 v12, s[sgprNumWorkGroups1]           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v12, v12                           // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_f32_u32 v13, s[sgprWorkGroup1]               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v12, v12, v13                            // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_u32_f32 v12, v12                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v13, v12, s[sgprNumWorkGroups1]      // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_sub_u32 v13, s[sgprWorkGroup1], v13              // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmpx_eq_u32 exec, v13, s[sgprNumWorkGroups1]     // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_u32 v12, 1, v12                              // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v13, 0                                   // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprNumWorkGroups1]     // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s[sgprNumWorkGroups1]      // re-calculate remainder
v_sub_u32 v13, s[sgprWorkGroup1], v13              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprGSUSumIdx], v12          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v13         // remainder
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
s_mov_b32 s16, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s16, s16                            // Restore WGM
s_cmp_gt_i32 s16, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s16, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s16, s16                                 // abs(WGM)
v_cvt_f64_u32 v[12:13], s16                        // s17 = s[sgprWorkGroup0] / s16
v_rcp_f64 v[12:13], v[12:13]                       // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[14:15], s[sgprWorkGroup0]          // s17 = s[sgprWorkGroup0] / s16
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s17 = s[sgprWorkGroup0] / s16
v_cvt_u32_f64 v12, v[12:13]                        // s17 = s[sgprWorkGroup0] / s16
v_mul_lo_u32 v13, v12, s16                         // s17 = s[sgprWorkGroup0] / s16
v_sub_u32 v14, s[sgprWorkGroup0], v13              // s17 = s[sgprWorkGroup0] / s16
v_cmpx_ge_u32 exec, v14, s16                       // s17 = s[sgprWorkGroup0] / s16
v_add_u32 v12, v12, 1                              // s17 = s[sgprWorkGroup0] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s17, v12                       // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup0], s20              // WorkGroup0=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s20, s20, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[12:13], s16                        // s18 = s[sgprNumWorkGroups0] / s16
v_rcp_f64 v[12:13], v[12:13]                       // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_f64_u32 v[14:15], s[sgprNumWorkGroups0]      // s18 = s[sgprNumWorkGroups0] / s16
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_u32_f64 v12, v[12:13]                        // s18 = s[sgprNumWorkGroups0] / s16
v_mul_lo_u32 v13, v12, s16                         // s18 = s[sgprNumWorkGroups0] / s16
v_sub_u32 v14, s[sgprNumWorkGroups0], v13          // s18 = s[sgprNumWorkGroups0] / s16
v_cmpx_ge_u32 exec, v14, s16                       // s18 = s[sgprNumWorkGroups0] / s16
v_add_u32 v12, v12, 1                              // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s18, v12                       // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups0], s19          // NumWorkGroups0=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[12:13], s18                        // s[sgprWorkGroup1] = s20 / s18
v_rcp_f64 v[12:13], v[12:13]                       // s[sgprWorkGroup1] = s20 / s18
v_cvt_f64_u32 v[14:15], s20                        // s[sgprWorkGroup1] = s20 / s18
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s[sgprWorkGroup1] = s20 / s18
v_cvt_u32_f64 v12, v[12:13]                        // s[sgprWorkGroup1] = s20 / s18
v_mul_lo_u32 v13, v12, s18                         // s[sgprWorkGroup1] = s20 / s18
v_sub_u32 v14, s20, v13                            // s[sgprWorkGroup1] = s20 / s18
v_cmpx_ge_u32 exec, v14, s18                       // s[sgprWorkGroup1] = s20 / s18
v_add_u32 v12, v12, 1                              // s[sgprWorkGroup1] = s20 / s18
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v13, v12, s18                         // s[sgprWorkGroup1] = s20 / s18
v_sub_u32 v14, s20, v13                            // s[sgprWorkGroup1] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup1], v12         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v14         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s16, s16                                 // WGM
v_cvt_f64_u32 v[12:13], s16                        // s17 = s[sgprWorkGroup1] / s16
v_rcp_f64 v[12:13], v[12:13]                       // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[14:15], s[sgprWorkGroup1]          // s17 = s[sgprWorkGroup1] / s16
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s17 = s[sgprWorkGroup1] / s16
v_cvt_u32_f64 v12, v[12:13]                        // s17 = s[sgprWorkGroup1] / s16
v_mul_lo_u32 v13, v12, s16                         // s17 = s[sgprWorkGroup1] / s16
v_sub_u32 v14, s[sgprWorkGroup1], v13              // s17 = s[sgprWorkGroup1] / s16
v_cmpx_ge_u32 exec, v14, s16                       // s17 = s[sgprWorkGroup1] / s16
v_add_u32 v12, v12, 1                              // s17 = s[sgprWorkGroup1] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s17, v12                       // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup1], s20              // WorkGroup1=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s20, s20, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[12:13], s16                        // s18 = s[sgprNumWorkGroups1] / s16
v_rcp_f64 v[12:13], v[12:13]                       // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_f64_u32 v[14:15], s[sgprNumWorkGroups1]      // s18 = s[sgprNumWorkGroups1] / s16
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_u32_f64 v12, v[12:13]                        // s18 = s[sgprNumWorkGroups1] / s16
v_mul_lo_u32 v13, v12, s16                         // s18 = s[sgprNumWorkGroups1] / s16
v_sub_u32 v14, s[sgprNumWorkGroups1], v13          // s18 = s[sgprNumWorkGroups1] / s16
v_cmpx_ge_u32 exec, v14, s16                       // s18 = s[sgprNumWorkGroups1] / s16
v_add_u32 v12, v12, 1                              // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s18, v12                       // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups1], s19          // NumWorkGroups1=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[12:13], s18                        // s[sgprWorkGroup0] = s20 / s18
v_rcp_f64 v[12:13], v[12:13]                       // s[sgprWorkGroup0] = s20 / s18
v_cvt_f64_u32 v[14:15], s20                        // s[sgprWorkGroup0] = s20 / s18
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s[sgprWorkGroup0] = s20 / s18
v_cvt_u32_f64 v12, v[12:13]                        // s[sgprWorkGroup0] = s20 / s18
v_mul_lo_u32 v13, v12, s18                         // s[sgprWorkGroup0] = s20 / s18
v_sub_u32 v14, s20, v13                            // s[sgprWorkGroup0] = s20 / s18
v_cmpx_ge_u32 exec, v14, s18                       // s[sgprWorkGroup0] = s20 / s18
v_add_u32 v12, v12, 1                              // s[sgprWorkGroup0] = s20 / s18
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v13, v12, s18                         // s[sgprWorkGroup0] = s20 / s18
v_sub_u32 v14, s20, v13                            // s[sgprWorkGroup0] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup0], v12         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v14         // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s20, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s17 // wg1 += blockId * WGM
label_WGM:

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v6 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v8 */

/* global read addresses: unroll assignment a */
/* v7 */

/* global read addresses: unroll assignment b */
/* v9 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */
v_mov_b32 v12, v6                                  // groA0I_0

/* global read addresses: tile offsets b */
v_mov_b32 v13, v8                                  // groB1J_0

/* global read addresses: unroll offsets a */
v_mov_b32 v14, v7                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v15, v9                                  // groBL_0

/* global read addresses: shift a */
s_mul_i32 s16, s[sgprWorkGroup0], 32               // WorkGroup[01] * MT
s_sub_u32 s16, s[sgprSizeI], s16                   // edge = Size0I - WG*MT
s_sub_u32 s16, s16, 8                              // edge -= margin(8)
v_mov_b32 v16, s16                                 // edge vgpr = Size0I- WG*MT - margin(8)
v_min_i32 v12, v16, v12                            // offset = (offset < edge) ? offset(v12) : edge(v16)

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup0], 32            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup0], 32               // WorkGroup[01] * MT
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s17, 64, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 64, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 6 // s[LoopCounterL] = s[sgprSizesSum] / 64
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v16, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v16, v16                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v17, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v16, v16, v17                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v16, v16                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v17, v16, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v17, s[sgprLoopCounterL], v17            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v17, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v16, 1, v16                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v17, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v17, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v16, v16, 1                              // quotient - 1
v_mul_u32_u24 v17, v16, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v17, s[sgprLoopCounterL], v17            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v16       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v17        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 64                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 64                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s20, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s20, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s16, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s16, s16, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s17, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdA:sgprSrdA+1], s[16:17], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_dwordx2 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[18:19], s[18:19], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s18, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s19, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s17, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s19   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s19, s[sgprWorkGroup1], 32            // WorkGroup[01] * MT
s_mul_i32 s18, s[sgprWorkGroup1], 32               // WorkGroup[01] * MT
s_mul_hi_u32 s19, s18, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s18, s18, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s17, 64, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s16, 64, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 6 // s[LoopCounterL] = s[sgprSizesSum] / 64
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v16, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v16, v16                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v17, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v16, v16, v17                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v16, v16                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v17, v16, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v17, s[sgprLoopCounterL], v17            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v17, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v16, 1, v16                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v17, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v17, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v16, v16, 1                              // quotient - 1
v_mul_u32_u24 v17, v16, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v17, s[sgprLoopCounterL], v17            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v16       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v17        // remainder
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s16, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s16, s16, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s16, s16, s17                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s17, s16, 64                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s16, s16, 64                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
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
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s20, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s20, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s16, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s16, s16, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s17, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdB:sgprSrdB+1], s[16:17], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_dwordx2 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[18:19], s[18:19], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s18, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s19, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s17, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s16, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s18, s18, s16                            // accum wg term to tilestart
s_addc_u32 s19, s19, s17                           // accum wg term to tilestart
s_lshl_b64 s[18:19], s[18:19], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s18    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s19   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v16, s[sgprStrideAL], v[14]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc, v[12], v[16+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v16, s[sgprStrideB1J], v[13]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc, v[15], v[16+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsA+0], 128            // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s17, s17, s[sgprGlobalReadIncsA+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cmov_b32 s17, 128                                // DepthU*Bpe if GSUC = 1
s_mul_i32 s[sgprGlobalReadIncsA+0], s17, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsB+0], 128            // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s17, s17, s[sgprGlobalReadIncsB+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s17 // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 6 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 64
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v6, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v6, v6                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v7, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v6, v6, v7                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v6, v6                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v7, s[sgprLoopCounterL], v7              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v7, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v6, 1, v6                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v7, s[sgprGSUSumIdx+1]         // overflow happened in remainder
v_sub_u32 v6, v6, 1                                // quotient - 1
v_mul_u32_u24 v7, v6, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_u32 v7, s[sgprLoopCounterL], v7              // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v6        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v7         // remainder
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
s_cbranch_scc0 label_StaggerUMapping
s_mov_b32 s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping:
s_cmp_eq_u32 s19, 0x2000
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s16, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s19, 0x4000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s19, 0x6000
s_cbranch_scc0 label_StaggerUMapping_3
s_mul_i32 s17, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s16, s16, s17
s_add_u32 s16, s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s19, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s16 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s18 // shift by StaggerUStride

/* addr += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_u32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* addr += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_u32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
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
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

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
s_nop 1                                            // alpha >= numSgprPreload, wait for kern args before
s_and_b32 s66, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s66, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
s_and_b32 s66, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s66, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_RegularSrdInitializationD
s_branch label_GeneralBatchedGemmSrdInitiationD    // General Batched GEMM, Srd initialized to 0
label_RegularSrdInitializationD:  /// Regular SRD initialization for non-General Batched GEMM for D
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_GeneralBatchedGemmSrdInitiationD:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationD_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_and_b32 s66, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s66, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
s_and_b32 s66, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s66, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_RegularSrdInitializationC
s_branch label_GeneralBatchedGemmSrdInitiationC    // General Batched GEMM, Srd initialized to 0
label_RegularSrdInitializationC:  /// Regular SRD initialization for non-General Batched GEMM for C
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_GeneralBatchedGemmSrdInitiationC:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationC_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s68, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s67, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s67, s68, s[sgprStrideC1J]            // ScaleC s68 by Stride
s_mul_i32 s66, s68, s[sgprStrideC1J]               // ScaleC s68 by Stride
s_lshl_b64 s[66:67], s[66:67], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // add hi to SRD
s_and_b32 s67, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s67, s68, s[sgprStrideD1J]            // ScaleD s68 by Stride
s_mul_i32 s66, s68, s[sgprStrideD1J]               // ScaleD s68 by Stride
s_lshl_b64 s[66:67], s[66:67], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi to SRD

s_and_b32 s67, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s67, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_and_b32 s69, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s69, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s67, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s66, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[66:67], s[66:67], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s66, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s66, s66, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s67, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[66:67], s[66:67], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // Offsetting within the Batch Matrix [Higher half of address]
s_load_dwordx2 s[66:67], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s67, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s67, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_and_b32 s69, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s69, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s67, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s66, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[66:67], s[66:67], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s66, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s66, s66, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s67, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[66:67], s[66:67], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // Offsetting within the Batch Matrix [Higher half of address]
s_load_dwordx2 s[66:67], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_and_b32 s66, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s66, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s67, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s66, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s68, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s68, s68, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s69, s68, s[sgprStrideC1J]            // Free1
s_mul_i32 s68, s68, s[sgprStrideC1J]               // Free1
s_add_u32 s66, s66, s68                            // Free1
s_addc_u32 s67, s67, s69                           // Free1
s_sub_u32 s68, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s68, s68, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s69, s68, s[sgprStrideCK]             // Free2
s_mul_i32 s68, s68, s[sgprStrideCK]                // Free2
s_add_u32 s66, s66, s68                            // Free2
s_addc_u32 s67, s67, s69                           // Free2
s_lshl_b64 s[66:67], s[66:67], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...4) from pool */

/* initC: remove ValuA/B vgpr buffer [6...30) from pool */
v_accvgpr_write acc0, 0                            // initC
v_accvgpr_write acc1, 0                            // initC
v_accvgpr_write acc2, 0                            // initC
v_accvgpr_write acc3, 0                            // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */

/* label_PrefetchGlobalLastIterEnd */
s_cbranch_scc0 label_NoBranch_0                    // Only branch on scc1
s_getpc_b64 s[66:67]                               // addr of next instr
s_add_i32 s68, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s66, s66, s68                            // add target branch offset
s_addc_u32 s67, s67, 0                             // add high and carry
s_setpc_b64 s[66:67]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_0:
s_waitcnt vmcnt(0)                                 // wait for global read

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0

/* local write swap a */

/* (EPS=1) local write swap internal offset -> 16384 */

/* local write swap b */

/* (EPS=1) local write swap internal offset -> 16384 */
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
s_waitcnt 0                                        // force waitcnt0
s_barrier                                          // LW to PLR, sync

/* local read prefetch a */
ds_read_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+2], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
/* N/A, lro->1024 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read inc b */
/* N/A, lro->32 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

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

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:1  */
/*  mfmaIndex:0  */
ds_read_u16 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:2176 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+0], v[vgprLocalReadAddrA+0] offset:2240 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:2304 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+1], v[vgprLocalReadAddrA+0] offset:2368 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:2432 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+2], v[vgprLocalReadAddrA+0] offset:2496 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:2560 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+3], v[vgprLocalReadAddrA+0] offset:2624 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:16384 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 16384 sync LDS1
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:16384 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 16384 sync LDS1

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 0 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 0 */

/* local read swap offsets a */

/* local read swap internal offset -> 16384 */

/* local read swap offsets b */

/* local read swap internal offset -> 16384 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(11)                              // wait for prior local read local write old=0, new=11 newLW=2 newLR=9 for iteration == 0
/* pack scheduling: packAIdx:4, packBIdx:0 */
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D1+3] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=1 */

/* iter 1 */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:1  */
/*  mfmaIndex:1  */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt 0                                        // force waitcnt0
s_barrier                                          // PGR, and wait until LW done to sync
ds_read_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:16384 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA+0] offset:16448 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_read_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:16512 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS1
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA+0] offset:16576 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS1
ds_read_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:16640 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0 sync LDS1
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+2], v[vgprLocalReadAddrA+0] offset:16704 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0 sync LDS1
ds_read_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:16768 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0 sync LDS1
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+3], v[vgprLocalReadAddrA+0] offset:16832 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0 sync LDS1
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:16384 // L -> Reg lro=0 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
s_waitcnt lgkmcnt(11)                              // wait for prior local read local write old=0, new=11 newLW=2 newLR=9
/* pack scheduling: packAIdx:4, packBIdx:0 */
v_or_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D1+3] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=1 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=1 */

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

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:1  */
/*  mfmaIndex:0  */
ds_read_u16 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:18560 // L -> Reg lro=1024 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+0], v[vgprLocalReadAddrA+0] offset:18624 // L -> Reg lro=1024 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
ds_read_u16 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:18688 // L -> Reg lro=1024 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS1
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+1], v[vgprLocalReadAddrA+0] offset:18752 // L -> Reg lro=1024 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS1
ds_read_u16 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:18816 // L -> Reg lro=1024 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=1 iui=0 sync LDS1
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+2], v[vgprLocalReadAddrA+0] offset:18880 // L -> Reg lro=1024 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=1 iui=0 sync LDS1
ds_read_u16 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:18944 // L -> Reg lro=1024 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=1 iui=0 sync LDS1
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+3], v[vgprLocalReadAddrA+0] offset:19008 // L -> Reg lro=1024 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=1 iui=0 sync LDS1
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:16448 // L -> Reg lro=32 swapByteOffset=16384 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0

/* local write swap offsets a */

/* (EPS=1) local write swap internal offset -> 16384 */

/* local write swap offsets b */

/* (EPS=1) local write swap internal offset -> 16384 */

/* local read swap offsets a */

/* local read swap internal offset -> 0 */

/* local read swap offsets b */

/* local read swap internal offset -> 0 */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(11)                              // wait for prior local read local write old=0, new=11 newLW=2 newLR=9 for iteration == 0
/* pack scheduling: packAIdx:4, packBIdx:0 */
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D1+3] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=1 */

/* iter 1 */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:1  */
/*  mfmaIndex:1  */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(0)                               // 3wait for local write
s_waitcnt 0                                        // force waitcnt0
s_barrier                                          // PGR, and wait until LW done to sync
ds_read_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+2], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
s_waitcnt lgkmcnt(11)                              // wait for prior local read local write old=0, new=11 newLW=2 newLR=9
/* pack scheduling: packAIdx:4, packBIdx:0 */
v_or_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D1+3] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=1 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=1 */

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
v_xor_b32 v[vgprLocalReadAddrA], 0x4000, v[vgprLocalReadAddrA] // swap Red Blk
v_xor_b32 v[vgprLocalReadAddrB], 0x4000, v[vgprLocalReadAddrB] // swap Red Blk
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */
s_and_b32 s8, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1

/******************************************/
/* Opt. NoLoadLoop - Begin                */
/******************************************/
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_OptNLL_End                    // Branch if Beta is not zero

s_cmp_eq_u32 s[sgprAlpha], 1.0                     // Alpha == 1.0 ?
s_cbranch_scc0 label_OptNLL_End                    // branch if alpha != 1


/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s66, 31, s[sgprSizeI]                    // s66 = s[sgprSizeI] % 32
s_add_u32 s67, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s67                // wg0 >= nwg0-1 ?
s_cselect_b32 s66, s66, 0                          // set rem
s_cmpk_gt_u32 s66, 0                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s66, 31, s[sgprSizeJ]                    // s66 = s[sgprSizeJ] % 32
s_add_u32 s67, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s67                // wg1 >= nwg1-1
s_cselect_b32 s66, s66, 0                          // set rem
s_cmpk_gt_u32 s66, 0                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required

s_and_b32 s67, 63, s[sgprSizesSum+0]               // s67 = s[sgprSizesSum+0] % 64
s_cmp_eq_u32 s67, 0                                // numIterL == 0
s_cbranch_scc0 label_OptNLL_End                    // skip if tail loop required

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:1  */
/*  mfmaIndex:0  */
ds_read_u16 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:2176 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+0], v[vgprLocalReadAddrA+0] offset:2240 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:2304 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+1], v[vgprLocalReadAddrA+0] offset:2368 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:2432 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+2], v[vgprLocalReadAddrA+0] offset:2496 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:2560 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+3], v[vgprLocalReadAddrA+0] offset:2624 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=0 newLR=9 for iteration == 0
/* pack scheduling: packAIdx:4, packBIdx:0 */
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D1+3] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=1 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:1  */
/*  mfmaIndex:1  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
/* pack scheduling: packAIdx:4, packBIdx:0 */
v_or_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D1+3] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=1 */
label_toPGR1end_OptNLL:
/* Stores for OptNLL */
label_Summation_End_OptNLL:
/* endSummation: add vgpr [0...38) to pool */
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // 4 = Serial / 64
v_lshrrev_b32 v5, 1, v4                            // 5 = 4 / 2
v_lshlrev_b32 v1, 4, v5                            // wave coordination offset 1
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v1, v5, v1, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v5, 1, v4                                // v5 = v4 % 2
v_lshlrev_b32 v5, 4, v5                            // wave coordination offset 0
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v0, 4, v0                            // 0 = 0 / 16
v_lshlrev_b32 v0, 2, v0                            // thread0 * continuous_output
v_add_lshl_u32 v0, v5, v0, 0                       // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s8, 32, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_u32 v0, s8, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 32, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v1, s8, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/******************************************/
/* Global Write Elements                  */
/******************************************/
label_GW_B0_OptNLL_MB:
label_GW_B0_FD0_OptNLL_MB:
label_GW_B0_FD0_VW4_OptNLL_MB_Then:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=12 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v11, v3, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc3           // copy acc to vreg[3]

/* apply mask, calc new C and issue writes */
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:

s_endpgm                                           // Kernel End
label_OptNLL_End:
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:1  */
/*  mfmaIndex:0  */
ds_read_u16 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA+0] offset:2176 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+0], v[vgprLocalReadAddrA+0] offset:2240 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA+0] offset:2304 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+1], v[vgprLocalReadAddrA+0] offset:2368 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA+0] offset:2432 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+2], v[vgprLocalReadAddrA+0] offset:2496 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA+0] offset:2560 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X1_I0_D1+3], v[vgprLocalReadAddrA+0] offset:2624 // L -> Reg lro=1024 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_waitcnt lgkmcnt(9)                               // wait for prior local read local write old=0, new=9 newLW=0 newLR=9 for iteration == 0
/* pack scheduling: packAIdx:4, packBIdx:0 */
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D1+3] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=1 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:1  */
/*  mfmaIndex:1  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
/* pack scheduling: packAIdx:4, packBIdx:0 */
v_or_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D1+3] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=1 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [6...30) to pool */

/* Tail: add address/G2L vgpr [30...38) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */
v_and_b32 v[vgprLocalWriteAddrA], 0xf03fff, v[vgprLocalWriteAddrA] // reset to Red

/* local write reset offsets b */
v_and_b32 v[vgprLocalWriteAddrB], 0xf03fff, v[vgprLocalWriteAddrB] // reset to Red
/* Check out VGPR (numG2LA,numG2LB,numG2LMXSA,numG2LMXSB,numG2LMetadata) = (4,4,0,0,0) */
.set vgprG2LA_BASE, 6
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 10
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 63, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 64
s_and_b32 s66, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s67, s[sgprSizesSum], 6                 // s67 = s[sgprSizesSum] / 64
s_and_b32 s68, s[sgprGSU], 0xfff                   // Restore GSU
v_cvt_f32_u32 v14, s68                             // s66 = s67 / s68
v_rcp_iflag_f32 v14, v14                           // s66 = s67 / s68
v_cvt_f32_u32 v15, s67                             // s66 = s67 / s68
v_mul_f32 v14, v14, v15                            // s66 = s67 / s68
v_cvt_u32_f32 v14, v14                             // s66 = s67 / s68
v_mul_u32_u24 v15, v14, s68                        // s66 = s67 / s68
v_sub_u32 v15, s67, v15                            // s66 = s67 / s68
v_cmpx_eq_u32 exec, v15, s68                       // s66 = s67 / s68
v_add_u32 v14, 1, v14                              // s66 = s67 / s68
v_mov_b32 v15, 0                                   // s[sgprGSUSumIdx+1] = s67 % s68
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v15, s68                       // overflow happened in remainder
v_sub_u32 v14, v14, 1                              // quotient - 1
v_mul_u32_u24 v15, v14, s68                        // re-calculate remainder
v_sub_u32 v15, s67, v15                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s66, v14                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v15        // remainder
s_sub_u32 s67, s68, 1                              // GSU-1
s_cmp_eq_u32 s66, 0                                // quotient == 0
s_cselect_b32 s66, s[sgprGSUSumIdx+1], s67         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s66                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
//  removeStagger A
s_sub_i32 s66, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s66, 0
s_cbranch_scc0 label_Negative_2
s_mul_hi_u32 s67, s66, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s66, s66, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_3
label_Negative_2:
s_abs_i32 s66, s66
s_mul_hi_u32 s67, s66, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s66, s66, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s66, s66, 0xffffffff
s_xor_b32 s67, s67, 0xffffffff
s_add_u32 s66, s66, 0x1
s_addc_u32 s67, s67, 0
label_MultiplyDone_3:
s_sub_u32 s66, s66, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s67, s67, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
//  removeStagger B
s_sub_i32 s66, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s66, 0
s_cbranch_scc0 label_Negative_4
s_mul_hi_u32 s67, s66, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s66, s66, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_5
label_Negative_4:
s_abs_i32 s66, s66
s_mul_hi_u32 s67, s66, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s66, s66, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s66, s66, 0xffffffff
s_xor_b32 s67, s67, 0xffffffff
s_add_u32 s66, s66, 0x1
s_addc_u32 s67, s67, 0
label_MultiplyDone_5:
s_sub_u32 s66, s66, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s67, s67, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* Tail global read A */
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* Update M0 for DTLDS */

/* Tail global read B */
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* release sgprs that will not be used */
.set sgprWGM, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF

/* find the last element location for a */

/* find the last element location for b */
// Calculate SizeJ % MacroTile1
s_mul_i32 s11, s[sgprWorkGroup1], 32               // Calculate the remaining dimension along I/J direction.
s_sub_u32 s11, s[sgprSizeJ], s11                   // Calculate the remaining dimension along I/J direction.
s_mul_i32 s11, s11, 2                              // In bytes
s_and_b32 s31, s[sgprSizeL], 63                    // Calculate the remaining dimension along L direction.
s_lshr_b32 s56, s31, 0x6                           // Divided by lsc(64)
s_mul_hi_u32 s29, s11, s31                         // Calculate total number of valid elements.
s_mul_i32 s33, s11, s31                            // Calculate total number of valid elements.
s_cmp_gt_u32 s29, 0
s_cmov_b32 s33, 0xffffffff                         // If valid elements > max(U32), set the value to max
s_sub_u32 s31, s[sgprSizeJ], 1                     // sLoadTileIdx starts from 0
// Calculate SizeJ - 1 % MacroTile1
s_lshr_b32 s11, s31, 5                             // s11 = s31 / 32
s_and_b32 s11, 31, s31                             // s11 = s31 % 32
s_lshr_b32 s11, s11, 0x5                           // Divide lsp to get the load tile index
s_mul_i32 s11, s11, 1                              // Multiply nlc
s_add_i32 s11, s11, s56
s_and_b32 s31, 63, s[sgprSizesSum+0]               // s31 = s[sgprSizesSum+0] % 64
s_and_b32 s31, s31, 7                              // sLoadNum = (SizesSum+0 mod DU) & glvw
s_and_b32 s29, s31, 0x1
s_mov_b32 s34, 0                                   // Set loop count = 0

/* load single element for B */
label_LoadB:
s_cmp_eq_u32 s29, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_MergeB                        // Skip loading B
label_LOAD_B0:
label_LOAD_B0_K1:
s_cmp_ge_u32 s31, 1
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 0 */
buffer_load_short_d16 v18, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B0_K3:
s_cmp_ge_u32 s31, 3
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 2 */
buffer_load_short_d16 v19, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B0_K5:
s_cmp_ge_u32 s31, 5
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 4 */
buffer_load_short_d16 v20, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B0_K7:
s_cmp_ge_u32 s31, 7
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 6 */
buffer_load_short_d16 v21, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeB

/* merge single element for B */
label_MergeB:
s_cmp_eq_u32 s29, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_CheckOtherLoadB               // Skip mergeing B
label_MERGE_B0:
label_MERGE_B0_K1:
s_cmp_ge_u32 s31, 1
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v18     // HasEccHalf: pack
label_MERGE_B0_K3:
s_cmp_ge_u32 s31, 3
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v19     // HasEccHalf: pack
label_MERGE_B0_K5:
s_cmp_ge_u32 s31, 5
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v20     // HasEccHalf: pack
label_MERGE_B0_K7:
s_cmp_ge_u32 s31, 7
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v21     // HasEccHalf: pack
s_branch label_CheckOtherLoadB

/* reload loop for a: check if there's other load range need to be reloaded */
label_CheckOtherLoadA:

/* reload loop for b: check if there's other load range need to be reloaded */
label_CheckOtherLoadB:
s_cmp_eq_u32 s29, 0                                // Noneed to load single element for B?
s_cbranch_scc1 label_TailGlobalLoadEnd
s_add_u32 s34, s34, 1
s_cmp_eq_u32 s34, 1                                // Have reloaded all subtiles?
s_cbranch_scc1 label_TailGlobalLoadEnd
s_sub_i32 s11, s11, 1                              // Check the upper subtile
s_cmp_lt_i32 s11, 0
s_cselect_b32 s40, 1, 0                            // Back to the last subtile
s_add_i32 s11, s11, s40                            // If currently reload the first subtile,                                   check the last subtile next.
label_B0:
v_mov_b32 v14, v[vgprGlobalReadOffsetB+0]
label_CheckAddrB:
v_sub_u32 v14, v14, 16                             // sub prepad
v_add_u32 v15, v14, 15                             // Calculate load range per thread
v_cmp_lt_i32 s[40:41], v14, s33                    // If loading start address < total valid bytes?
v_cmp_ge_i32 s[56:57], v15, s33                    // If loading end address >= total valid bytes?
s_and_b32 s40, s40, s56                            // Find threads which access the last element
s_and_b32 s41, s41, s57                            // Find thread that access the last element
s_add_u32 s40, s40, s41                            // Find thread that access the last element
s_cmp_lg_u32 s40, 0                                // Have threads access the last element?
s_cbranch_scc1 label_LoadB                         // Reload B

/* global read for tail done */
label_TailGlobalLoadEnd:
s_waitcnt vmcnt(0)                                 // 2wait for global read
s_waitcnt 0                                        // force waitcnt0
s_barrier

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
s_waitcnt 0                                        // force waitcnt0
s_barrier                                          // Tail loop LW->LR, sync LDS0
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 6
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+4
.set vgprValuA_X0_I0_D0_PACK, 14
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuA_X1_I0_D1, vgprValuA_X0_I0_D0_PACK+4
.set vgprValuB_X0_I0_BASE, 22
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+4

/* Tail: local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA+0], 0x3fff, v[vgprLocalReadAddrA+0] // reset Red,Blk -> Red

/* Tail: local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB+0], 0x3fff, v[vgprLocalReadAddrB+0] // reset Red,Blk -> Red

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_read_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA+0] offset:192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+2], v[vgprLocalReadAddrA+0] offset:320 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA+0] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+3], v[vgprLocalReadAddrA+0] offset:448 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
/* Adding additional 128 pad since cumulative inc has reached 512 */
s_mov_b32 s8, 2176                                 // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s8, v[vgprLocalReadAddrA+0] // lrA += 2048 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s8, 64                                   // inc
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s8, v[vgprLocalReadAddrB+0] // lrB += 64 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+2] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D1+3] // pack two half Vgpr to one Vgpr
v_and_b32 v30, 63, v[vgprSerial]                   // v30 = v[vgprSerial] % 64
v_lshrrev_b32 v30, 4, v30                          // 30 = 30 / 16
v_lshlrev_b32 v30, 3, v30                          // v30 = v30 * 8
v_add_u32 v31, v30, 0
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], 0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], 0, s[28:29] // set 0 if K_idx >= sizeL
v_add_u32 v31, v31, 4                              // add part of K
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+2], v[vgprValuA_X0_I0+0+0+0+2], 0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0+3], 0, s[28:29] // set 0 if K_idx >= sizeL
v_and_b32 v30, 63, v[vgprSerial]                   // v30 = v[vgprSerial] % 64
v_lshrrev_b32 v30, 4, v30                          // 30 = 30 / 16
v_lshlrev_b32 v30, 3, v30                          // v30 = v30 * 8
v_add_u32 v31, v30, 0
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0, s[28:29] // set 0 if K_idx >= sizeL
v_add_u32 v31, v31, 4                              // add part of K
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], 0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], 0, s[28:29] // set 0 if K_idx >= sizeL
s_and_b32 s30, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s30, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_6    // skip mask
s_and_b32 s30, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s30, 8, s30                              // use shift to fill 0 for outside element
s_lshl_b32 s30, s30, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[32:33], s30, v[vgprValuA_X0_I0+0+0+0+0:vgprValuA_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[34:35], s30, v[vgprValuA_X0_I0+0+0+0+2:vgprValuA_X0_I0+0+0+0+2+1]
v_add_u32 v31, v30, 4                              // add part of K
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v32, s[28:29]
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v33, s[28:29]
v_add_u32 v31, v31, 4                              // add part of K
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+2], v[vgprValuA_X0_I0+0+0+0+2], v34, s[28:29]
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+3], v[vgprValuA_X0_I0+0+0+0+3], v35, s[28:29]
v_lshlrev_b64 v[32:33], s30, v[vgprValuB_X0_I0+0+0+0+0:vgprValuB_X0_I0+0+0+0+0+1]
v_lshlrev_b64 v[34:35], s30, v[vgprValuB_X0_I0+0+0+0+2:vgprValuB_X0_I0+0+0+0+2+1]
v_add_u32 v31, v30, 4                              // add part of K
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v32, s[28:29]
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v33, s[28:29]
v_add_u32 v31, v31, 4                              // add part of K
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+2], v[vgprValuB_X0_I0+0+0+0+2], v34, s[28:29]
v_cmp_ge_i32 s[28:29], v31, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0+3], v35, s[28:29]
label_TailLoop_SkipZeroOutMask_6:
s_nop 1
v_mfma_f32_16x16x32_f16 acc[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+3], acc[0:3] // left value = acc[0+0:3+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_X1_I0, UNDEF
.set vgprValuA_X0_I0_D0_PACK, UNDEF
.set vgprValuA_X0_I0_D1, UNDEF
.set vgprValuA_X1_I0_D1, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_X1_I0, UNDEF

/* Tail: add MISC Vgpr [0...6) to pool */
label_Summation_End_7:
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* shift vector components d0 */
v_mov_b32 v3, s[sgprWorkGroup0]
v_mul_i32_i24 v3, -0x20, v3                        // wg*MT
v_add_co_u32 v3, vcc, s[sgprSizesFree+0], v3       // wgMT = Size - wg*MT
v_mov_b32 v4, 0x20                                 // MT
v_cmp_lt_u32 s[12:13], v3, v4                      // wgMT < MT
v_cndmask_b32 v3, v4, v3, s[12:13]                 // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v5, 6, v[vgprSerial]                 // 5 = Serial / 64
v_and_b32 v5, 1, v5                                // v5 = v5 % 2
v_lshrrev_b32 v6, 4, v3                            // 6 = 3 / 16
v_and_b32 v6, 1, v6                                // v6 = v6 % 2
v_cmp_eq_u32 s[12:13], v6, v5                      // wave_id == block_belong_to_wave?
v_cndmask_b32 v3, v4, v3, s[12:13]                 // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(1)) */
v_lshrrev_b32 v4, 4, v3                            // 4 = 3 / 16
v_lshlrev_b32 v6, 0, v5                            // v6 = v5 * 1
v_sub_u32 v4, v4, v6

/* gbReg: glvw block id */
v_lshrrev_b32 v6, 3, v3                            // 6 = 3 / 8

/* tgbReg: glvw block id */
v_lshrrev_b32 v7, 4, v[vgprSerial]                 // 7 = Serial / 16
v_and_b32 v7, 3, v7                                // v7 = v7 % 4
v_lshlrev_b32 v7, 2, v7                            // v7 = v7 * 4
v_lshrrev_b32 v7, 3, v7                            // 7 = 7 / 8
v_lshlrev_b32 v5, 1, v5                            // v5 = v5 * 2
v_add_co_u32 v7, vcc, v5, v7                       // tgbReg = (tid_coal * continOut) / GLVW
v_sub_u32 v6, v6, v7

/* vwReg: glvw in which vw block? */
v_and_b32 v5, 3, v3                                // permute register between threads
v_lshrrev_b32 v5, 3, v5                            // permute register between threads

/* rReg : reminder of M_size % GlobalReadVectorWidth */
v_and_b32 v7, 7, v3                                // v7 = v3 % 8
v_cmp_eq_u32 vcc, v7, 0x1                          // wgMT%VW == 1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1 // branch to shift d0 r=1
v_cmp_eq_u32 vcc, v7, 0x2                          // wgMT%VW == 2
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2 // branch to shift d0 r=2
v_cmp_eq_u32 vcc, v7, 0x3                          // wgMT%VW == 3
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3 // branch to shift d0 r=3
v_cmp_eq_u32 vcc, v7, 0x4                          // wgMT%VW == 4
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4 // branch to shift d0 r=4
v_cmp_eq_u32 vcc, v7, 0x5                          // wgMT%VW == 5
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5 // branch to shift d0 r=5
v_cmp_eq_u32 vcc, v7, 0x6                          // wgMT%VW == 6
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6 // branch to shift d0 r=6
v_cmp_eq_u32 vcc, v7, 0x7                          // wgMT%VW == 7
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7 // branch to shift d0 r=7

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0

/******************************************/
/* shift d0 r=1                           */
/******************************************/
label_ShiftVectorComponents0_GLVW1:
v_cmp_eq_u32 vcc, v4, 0x0

/* branch to shift d0 r1 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0

/******************************************/
/* shift d0 r=2                           */
/******************************************/
label_ShiftVectorComponents0_GLVW2:
v_cmp_eq_u32 vcc, v4, 0x0

/* branch to shift d0 r2 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0

/******************************************/
/* shift d0 r=3                           */
/******************************************/
label_ShiftVectorComponents0_GLVW3:
v_cmp_eq_u32 vcc, v4, 0x0

/* branch to shift d0 r3 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0

/******************************************/
/* shift d0 r=4                           */
/******************************************/
label_ShiftVectorComponents0_GLVW4:
v_cmp_eq_u32 vcc, v4, 0x0

/* branch to shift d0 r4 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM0

/******************************************/
/* shift d0 r=5                           */
/******************************************/
label_ShiftVectorComponents0_GLVW5:
v_cmp_eq_u32 vcc, v4, 0x0

/* branch to shift d0 r5 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM0

/******************************************/
/* shift d0 r=6                           */
/******************************************/
label_ShiftVectorComponents0_GLVW6:
v_cmp_eq_u32 vcc, v4, 0x0

/* branch to shift d0 r6 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM0

/******************************************/
/* shift d0 r=7                           */
/******************************************/
label_ShiftVectorComponents0_GLVW7:
v_cmp_eq_u32 vcc, v4, 0x0

/* branch to shift d0 r7 mb0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM0

/******************************************/
/* shift d0 r=1 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0:  /// r1 mb0
v_cmp_eq_u32 vcc, v5, 0x0

/* branch to shift d0 r1 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0_VW0

/******************************************/
/* shift d0 r=2 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0:  /// r2 mb0
v_cmp_eq_u32 vcc, v5, 0x0

/* branch to shift d0 r2 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0_VW0

/******************************************/
/* shift d0 r=3 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0:  /// r3 mb0
v_cmp_eq_u32 vcc, v5, 0x0

/* branch to shift d0 r3 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0_VW0

/******************************************/
/* shift d0 r=4 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM0:  /// r4 mb0
v_cmp_eq_u32 vcc, v5, 0x0

/* branch to shift d0 r4 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM0_VW0

/******************************************/
/* shift d0 r=5 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM0:  /// r5 mb0
v_cmp_eq_u32 vcc, v5, 0x0

/* branch to shift d0 r5 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM0_VW0

/******************************************/
/* shift d0 r=6 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM0:  /// r6 mb0
v_cmp_eq_u32 vcc, v5, 0x0

/* branch to shift d0 r6 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM0_VW0

/******************************************/
/* shift d0 r=7 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM0:  /// r7 mb0
v_cmp_eq_u32 vcc, v5, 0x0

/* branch to shift d0 r7 mb0 vw0 */
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM0_VW0

/******************************************/
/* shift d0 r=1 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0_VW0:  /// r1 mb0 vw0
s_mov_b32 s12, 0
v_cmpx_eq_u32 s[12:13], v6, s12                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc3                        // glvw 1 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:64               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
s_mov_b64 s[12:13], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[12:13]                    // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=2 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0_VW0:  /// r2 mb0 vw0
s_mov_b32 s12, 0
v_cmpx_eq_u32 s[12:13], v6, s12                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc2                        // glvw 2 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc3                        // glvw 2 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:64               // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:64               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc1, v8
s_mov_b64 s[12:13], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[12:13]                    // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=3 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0_VW0:  /// r3 mb0 vw0
s_mov_b32 s12, 0
v_cmpx_eq_u32 s[12:13], v6, s12                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc1                        // glvw 3 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc2                        // glvw 3 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc3                        // glvw 3 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:64               // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:64               // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:64               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc1, v8
v_accvgpr_write_b32 acc2, v9
s_mov_b64 s[12:13], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[12:13]                    // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=4 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM0_VW0:  /// r4 mb0 vw0
s_mov_b32 s12, 0
v_cmpx_eq_u32 s[12:13], v6, s12                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 4 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc1                        // glvw 4 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc2                        // glvw 4 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v10, acc3                       // glvw 4 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:64               // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:64               // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:64               // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc1, v8
v_accvgpr_write_b32 acc2, v9
v_accvgpr_write_b32 acc3, v10
s_mov_b64 s[12:13], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[12:13]                    // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=5 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM0_VW0:  /// r5 mb0 vw0
s_mov_b32 s12, 0
v_cmpx_eq_u32 s[12:13], v6, s12                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc3                        // glvw 5 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc0                        // glvw 5 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc1                        // glvw 5 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v10, acc2                       // glvw 5 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:64               // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:64               // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc1, v8
v_accvgpr_write_b32 acc2, v9
v_accvgpr_write_b32 acc3, v10
s_mov_b64 s[12:13], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[12:13]                    // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=6 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM0_VW0:  /// r6 mb0 vw0
s_mov_b32 s12, 0
v_cmpx_eq_u32 s[12:13], v6, s12                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc2                        // glvw 6 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc3                        // glvw 6 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc0                        // glvw 6 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v10, acc1                       // glvw 6 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:64               // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc1, v8
v_accvgpr_write_b32 acc2, v9
v_accvgpr_write_b32 acc3, v10
s_mov_b64 s[12:13], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[12:13]                    // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=7 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM0_VW0:  /// r7 mb0 vw0
s_mov_b32 s12, 0
v_cmpx_eq_u32 s[12:13], v6, s12                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc1                        // glvw 7 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc2                        // glvw 7 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc3                        // glvw 7 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v10, acc0                       // glvw 7 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:64             // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc1, v8
v_accvgpr_write_b32 acc2, v9
v_accvgpr_write_b32 acc3, v10
s_mov_b64 s[12:13], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[12:13]                    // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0

label_ShiftVectorComponents0_GLVW0:  /// end shift0

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // 4 = Serial / 64
v_lshrrev_b32 v5, 1, v4                            // 5 = 4 / 2
v_lshlrev_b32 v1, 4, v5                            // wave coordination offset 1
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v1, v5, v1, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v5, 1, v4                                // v5 = v4 % 2
v_lshlrev_b32 v5, 4, v5                            // wave coordination offset 0
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v0, 4, v0                            // 0 = 0 / 16
v_lshlrev_b32 v0, 2, v0                            // thread0 * continuous_output
v_add_lshl_u32 v0, v5, v0, 0                       // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s8, 32, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_u32 v0, s8, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 32, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v1, s8, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s8, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MB_Then         // jump if edges required
label_GW_B0_FD0_VW4_MB_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=12 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v11, v3, v0, 2                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[16:19], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_MB_NonEdgeEnd:
label_GW_B0_FD0_VW4_MB_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=10 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[28:29], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v11, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[32:33]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_MB_Else:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=28 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[28:29], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v15, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v15, v6, v15, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v16, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v16, v6, v16, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v17, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v17, v6, v17, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v18, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v18, v6, v18, s[32:33]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+14], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v11, v15, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v12, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v13, v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v14, v18, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
s_getpc_b64 s[28:29]                               // addr of next instr
s_add_i32 s30, label_KernelEnd, 4                  // target branch offset
s_add_u32 s28, s28, s30                            // add target branch offset
s_addc_u32 s29, s29, 0                             // add high and carry
s_setpc_b64 s[28:29]                               // branch to label_KernelEnd
label_GSU_4:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=12 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v11, v3, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=10 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[28:29], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v11, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[32:33]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_pk_f16_f32 v12, v[vgprValuC+12], v[vgprValuC+13] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v13, v[vgprValuC+14], v[vgprValuC+15] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[12:13], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW4_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=28 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[28:29], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v15, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v15, v6, v15, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v16, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v16, v6, v16, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v17, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v17, v6, v17, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v18, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v18, v6, v18, s[32:33]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+14], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v15, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v14, v[vgprValuC+14] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v14, v18, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 32 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 31, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_cmpk_gt_u32 s28, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=8 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v12, v2, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
buffer_load_dwordx2 v[14:15], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v11, v3, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 1 - 1 (beta) vscnt(0) (interleaved)
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v14, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v14, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v15, v[vgprValuC+18] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v15, v[vgprValuC+19] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=6 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[28:29], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v11, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[32:33]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[16:17], v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v11, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[32:33]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v16, v[vgprValuC+12] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v16, v[vgprValuC+13] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+14], s[sgprBeta], v17, v[vgprValuC+14] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+15], s[sgprBeta], v17, v[vgprValuC+15] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v12, v[vgprValuC+12], v[vgprValuC+13] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v13, v[vgprValuC+14], v[vgprValuC+15] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[12:13], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW4_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=20 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[28:29], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v16, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v16, v6, v16, s[32:33]               // LDC clip if OOB. offset
buffer_load_short_d16 v15, v16, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v16, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v16, v6, v16, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v18, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v18, v6, v18, s[32:33]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v17, v18, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v18, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v18, v6, v18, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v20, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v20, v6, v20, s[32:33]               // LDC clip if OOB. offset
buffer_load_short_d16 v19, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v20, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v20, v6, v20, s[32:33]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[28:29], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[32:33], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[32:33], s[28:29], s[32:33]             // in0 && in1
v_add_lshl_u32 v22, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v22, v6, v22, s[32:33]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v21, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v22, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v22, v6, v22, s[32:33]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+14], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+11], s[sgprBeta], v15, v[vgprValuC+11] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v17, v[vgprValuC+12] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v18, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v19, v[vgprValuC+13] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v20, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+14], s[sgprBeta], v21, v[vgprValuC+14] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v14, v[vgprValuC+14] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v14, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_End_2:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
