
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT4_4_MXLIBL_MXSFNS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PXCCM0_PAP0_PGL0_PGR2_PLR1_PKA1_RAP0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SKFTR0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSSK_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WASG_WG32_8_1_WGMXCC1_WQS0
.globl Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT4_4_MXLIBL_MXSFNS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PXCCM0_PAP0_PGL0_PGR2_PLR1_PKA1_RAP0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SKFTR0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSSK_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WASG_WG32_8_1_WGMXCC1_WQS0
.p2align 8
.type Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT4_4_MXLIBL_MXSFNS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PXCCM0_PAP0_PGL0_PGR2_PLR1_PKA1_RAP0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SKFTR0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSSK_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WASG_WG32_8_1_WGMXCC1_WQS0,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT4_4_MXLIBL_MXSFNS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PXCCM0_PAP0_PGL0_PGR2_PLR1_PKA1_RAP0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SKFTR0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSSK_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WASG_WG32_8_1_WGMXCC1_WQS0
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 192 // accvgpr offset
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 86 // sgprs
  .amdhsa_group_segment_fixed_size 32768 // lds bytes
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
/* Num VGPR   =192 */
/* Num AccVGPR=64 */
/* Num SGPR   =86 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 4 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=4 */
/* VectorWidthB=4 */
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
  - .name: Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT4_4_MXLIBL_MXSFNS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PXCCM0_PAP0_PGL0_PGR2_PLR1_PKA1_RAP0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SKFTR0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSSK_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WASG_WG32_8_1_WGMXCC1_WQS0
    .symbol: 'Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT4_4_MXLIBL_MXSFNS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PXCCM0_PAP0_PGL0_PGR2_PLR1_PKA1_RAP0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SKFTR0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSSK_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WASG_WG32_8_1_WGMXCC1_WQS0.kd'
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
      - .name:            AddressFlags
        .size:            8
        .offset:          64
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            ItersPerTile
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberItersPerTile
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicShiftItersPerTile
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SKItersPerWG
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skGrid
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            skTiles
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
      - .name:            AddressWS
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            D
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          120
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            batchOffsetD
        .size:            8
        .offset:          144
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          152
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          160
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          168
        .value_kind:      by_value
        .value_type:      u64
    .group_segment_fixed_size:   32768
    .kernarg_segment_align:      8
    .kernarg_segment_size:       176
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 86
    .sgpr_spill_count:           0
    .vgpr_count:                 192
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_HHS_BH_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT4_4_MXLIBL_MXSFNS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PXCCM0_PAP0_PGL0_PGR2_PLR1_PKA1_RAP0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SKFTR0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSSK_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS64_WASG_WG32_8_1_WGMXCC1_WQS0:
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
.set vgprBase, 12
.set vgprLocalWriteAddrA, 8
.set vgprLocalWriteAddrB, 9
.set vgprGlobalReadOffsetA, 0
.set vgprGlobalReadOffsetB, 4
.set vgprLocalReadAddrA, 10
.set vgprLocalReadAddrB, 11
.set vgprSerial, 124

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuA_X0_I0_D0_PACK, vgprBase+8
.set vgprValuB_X0_I0_BASE, vgprBase+40
.set vgprValuB_X0_I0_D0_PACK, vgprBase+48
.set vgprG2LA_BASE, vgprBase+80
.set vgprG2LB_BASE, vgprBase+96
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X2_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X3_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X2_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X3_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuA_X0_I0_D0, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+2
.set vgprValuA_X0_I0_D2, vgprValuA_X0_I0_D0_PACK+4
.set vgprValuA_X0_I0_D3, vgprValuA_X0_I0_D0_PACK+6
.set vgprValuA_X1_I0_D0, vgprValuA_X0_I0_D0_PACK+8
.set vgprValuA_X1_I0_D1, vgprValuA_X0_I0_D0_PACK+10
.set vgprValuA_X1_I0_D2, vgprValuA_X0_I0_D0_PACK+12
.set vgprValuA_X1_I0_D3, vgprValuA_X0_I0_D0_PACK+14
.set vgprValuA_X2_I0_D0, vgprValuA_X0_I0_D0_PACK+16
.set vgprValuA_X2_I0_D1, vgprValuA_X0_I0_D0_PACK+18
.set vgprValuA_X2_I0_D2, vgprValuA_X0_I0_D0_PACK+20
.set vgprValuA_X2_I0_D3, vgprValuA_X0_I0_D0_PACK+22
.set vgprValuA_X3_I0_D0, vgprValuA_X0_I0_D0_PACK+24
.set vgprValuA_X3_I0_D1, vgprValuA_X0_I0_D0_PACK+26
.set vgprValuA_X3_I0_D2, vgprValuA_X0_I0_D0_PACK+28
.set vgprValuA_X3_I0_D3, vgprValuA_X0_I0_D0_PACK+30
.set vgprValuB_X0_I0_D0, vgprValuB_X0_I0_D0_PACK+0
.set vgprValuB_X0_I0_D1, vgprValuB_X0_I0_D0_PACK+2
.set vgprValuB_X0_I0_D2, vgprValuB_X0_I0_D0_PACK+4
.set vgprValuB_X0_I0_D3, vgprValuB_X0_I0_D0_PACK+6
.set vgprValuB_X1_I0_D0, vgprValuB_X0_I0_D0_PACK+8
.set vgprValuB_X1_I0_D1, vgprValuB_X0_I0_D0_PACK+10
.set vgprValuB_X1_I0_D2, vgprValuB_X0_I0_D0_PACK+12
.set vgprValuB_X1_I0_D3, vgprValuB_X0_I0_D0_PACK+14
.set vgprValuB_X2_I0_D0, vgprValuB_X0_I0_D0_PACK+16
.set vgprValuB_X2_I0_D1, vgprValuB_X0_I0_D0_PACK+18
.set vgprValuB_X2_I0_D2, vgprValuB_X0_I0_D0_PACK+20
.set vgprValuB_X2_I0_D3, vgprValuB_X0_I0_D0_PACK+22
.set vgprValuB_X3_I0_D0, vgprValuB_X0_I0_D0_PACK+24
.set vgprValuB_X3_I0_D1, vgprValuB_X0_I0_D0_PACK+26
.set vgprValuB_X3_I0_D2, vgprValuB_X0_I0_D0_PACK+28
.set vgprValuB_X3_I0_D3, vgprValuB_X0_I0_D0_PACK+30
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
.set sgprAddressA, 24
.set sgprAddressB, 26
.set sgprStridesA, 28
.set sgprStridesB, 30
.set sgprAddressFlags, 32
.set sgprItersPerTile, 34
.set sgprMagicNumberItersPerTile, 35
.set sgprMagicShiftItersPerTile, 36
.set sgprSKItersPerWG, 37
.set sgprskGrid, 38
.set sgprskTiles, 39
.set sgprAlpha, 40
.set sgprBeta, 41
.set sgprAddressWS, 42
.set sgprAddressD, 44
.set sgprAddressC, 46
.set sgprStridesD, 48
.set sgprStridesC, 50
.set sgprSrdWS, 52
.set sgprStreamKLocalEnd, 56
.set sgprStreamKLocalStart, 57
.set sgprPersistentIterationEnd, 58
.set sgprPersistentIteration, 59
.set sgprPersistentWorkGroupIndex, 60

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
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 128
.set MT1, 128
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
s_mov_b32 m0, 0x8000                               // LDS clamp at 32768 bytes
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
v_cvt_f64_u32 v[12:13], s67                        // s63 = s[sgprWorkGroup0] / s67
v_rcp_f64 v[12:13], v[12:13]                       // s63 = s[sgprWorkGroup0] / s67
v_cvt_f64_u32 v[14:15], s[sgprWorkGroup0]          // s63 = s[sgprWorkGroup0] / s67
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s63 = s[sgprWorkGroup0] / s67
v_cvt_u32_f64 v12, v[12:13]                        // s63 = s[sgprWorkGroup0] / s67
v_mul_lo_u32 v13, v12, s67                         // s63 = s[sgprWorkGroup0] / s67
v_sub_u32 v14, s[sgprWorkGroup0], v13              // s63 = s[sgprWorkGroup0] / s67
v_cmpx_ge_u32 exec, v14, s67                       // s63 = s[sgprWorkGroup0] / s67
v_add_u32 v12, v12, 1                              // s63 = s[sgprWorkGroup0] / s67
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v13, v12, s67                         // s63 = s[sgprWorkGroup0] / s67
v_sub_u32 v14, s[sgprWorkGroup0], v13              // s63 = s[sgprWorkGroup0] / s67
v_readfirstlane_b32 s63, v12                       // quotient
v_readfirstlane_b32 s64, v14                       // remainder
s_mul_i32 s63, s63, s67
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s64, s64, s66
/* temp0 = temp0 + temp1 */
s_add_u32 s63, s63, s64
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[12:13], s67                        // s64 = s19 / s67
v_rcp_f64 v[12:13], v[12:13]                       // s64 = s19 / s67
v_cvt_f64_u32 v[14:15], s19                        // s64 = s19 / s67
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s64 = s19 / s67
v_cvt_u32_f64 v12, v[12:13]                        // s64 = s19 / s67
v_mul_lo_u32 v13, v12, s67                         // s64 = s19 / s67
v_sub_u32 v14, s19, v13                            // s64 = s19 / s67
v_cmpx_ge_u32 exec, v14, s67                       // s64 = s19 / s67
v_add_u32 v12, v12, 1                              // s64 = s19 / s67
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s64, v12                       // quotient
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
/* init: add vgpr [12...104) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...64) to pool */
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
s_waitcnt lgkmcnt(0)                               // wait for 100/0 bytes of kern args over preload
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s12, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s12, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s11, 128                                 // KernArgAddressOffset
s_mul_i32 s68, s16, 4
s_mov_b64 s[62:63], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s11, 252
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
s_lshr_b32 s67, s21, 7                             // s67 = s21 / 128
s_and_b32 s64, 127, s21                            // s64 = s21 % 128
s_addc_u32 s67, s67, 0
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
s_lshr_b32 s67, s21, 7                             // s67 = s21 / 128
s_and_b32 s64, 127, s21                            // s64 = s21 % 128
s_addc_u32 s67, s67, 0
s_mul_i32 s66, s66, s67
s_mul_i32 s66, s66, s22
s_add_u32 s69, s69, s66

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s63, s10, 1
s_sub_u32 s62, s69, s66
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s62
/* Check if custom structure pointer is null */
s_and_b32 s12, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s12, 2                                // ArgType == 2 ?
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
s_mul_i32 s63, s63, 252
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s63
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
s_load_dword s41, s[sgprKernArgAddress:sgprKernArgAddress+1], 136 // 136
label_LoadExternalStructEnd:
/* init: add vgpr [12...104) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...64) to pool */
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
s_waitcnt lgkmcnt(0)                               // wait for 100/0 bytes of kern args over preload

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

label_MultiGemmEnd:
.set sgprSrdA, 64
.set sgprSrdB, 68
.set sgprShadowLimitA, 62
.set sgprShadowLimitB, 72
.set sgprStaggerUIter, 61
.set sgprWrapUA, 74
.set sgprWrapUB, 76
.set sgprGlobalReadIncsA, 78
.set sgprGlobalReadIncsB, 79
.set sgprPackKForV0, 80
.set sgprPackKForV1, 81
s_mov_b32 s[sgprPackKForV0], 0x05040100
s_mov_b32 s[sgprPackKForV1], 0x07060302
s_and_b32 s12, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s12, 3                                // ArgType == 3 for General Batched GEMM
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
s_mov_b32 s[sgprPersistentWorkGroupIndex], s[sgprWorkGroup0] // Save mapped persistent rank
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitInit                  // Jump to single kernel init
v_cvt_f32_u32 v12, s[sgprSkSplit]                  // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_rcp_iflag_f32 v12, v12                           // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cvt_f32_u32 v13, s[sgprPersistentWorkGroupIndex] // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_f32 v12, v12, v13                            // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cvt_u32_f32 v12, v12                             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mul_u32_u24 v13, v12, s[sgprSkSplit]             // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_sub_u32 v13, s[sgprPersistentWorkGroupIndex], v13 // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_cmpx_eq_u32 exec, v13, s[sgprSkSplit]            // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_add_u32 v12, 1, v12                              // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
v_mov_b32 v13, 0                                   // TileIdx = SKIdx // WGsPerTile, PartialIdx = SKIdx % WGsPerTile
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprSkSplit]            // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s[sgprSkSplit]             // re-calculate remainder
v_sub_u32 v13, s[sgprPersistentWorkGroupIndex], v13 // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s12, v12                       // quotient
v_readfirstlane_b32 s13, v13                       // remainder
s_mul_i32 s14, s[sgprSkSplit], s[sgprSKItersPerWG]
s_sub_u32 s14, s[sgprItersPerTile], s14            // extraIters = itersPerTile - SkSplit * skItersPerWG
s_mul_i32 s[sgprPersistentIteration], s13, s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_cmp_lt_u32 s13, s14                              // Check if WG gets an extra iteration
s_cbranch_scc1 label_SK_HasExtra                   // Has extra iter
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s14 // This WG does not have an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_branch label_SK_DoneExtra                        // Done init for parallel reduction
label_SK_HasExtra:
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s13 // This WG has an extra iteration
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], 1 // StreamK ending iteration (case: after extra iters)
label_SK_DoneExtra:
s_mul_i32 s12, s12, s[sgprItersPerTile]            // Tile offset = tilesIdx * itersPerTile
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // Offset to correct tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s12 // Offset to correct tile
s_mov_b32 s[sgprSkPartialIdx], s13                 // Save partial idx for SrdD calculation
s_branch label_SK_InitDone                         // Done init for parallel reduction
label_SK_SplitInit:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprItersPerTile] // DP starting iteration (case: DP work to do)
s_mul_i32 s12, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s12, s12, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s12, s12, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_mov_b32 s[sgprPersistentIterationEnd], s12       // DP ending iteration (case: only DP work to do)
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_cmp_lt_u32 s12, s[sgprPersistentIterationEnd]    // Check if there are DP tiles to do
s_cbranch_scc1 label_SK_InitDone                   // Done init
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s13, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s12, s12, s13                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global first-E mapping
s_cbranch_scc0 label_SK_GlobalExtraIters           // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_AssignNoTiles              // no SK tiles -> global mapping
v_cvt_f32_u32 v12, s[sgprskTiles]                  // F = skGrid / skTiles, rem = skGrid % skTiles
v_rcp_iflag_f32 v12, v12                           // F = skGrid / skTiles, rem = skGrid % skTiles
v_cvt_f32_u32 v13, s[sgprskGrid]                   // F = skGrid / skTiles, rem = skGrid % skTiles
v_mul_f32 v12, v12, v13                            // F = skGrid / skTiles, rem = skGrid % skTiles
v_cvt_u32_f32 v12, v12                             // F = skGrid / skTiles, rem = skGrid % skTiles
v_mul_u32_u24 v13, v12, s[sgprskTiles]             // F = skGrid / skTiles, rem = skGrid % skTiles
v_sub_u32 v13, s[sgprskGrid], v13                  // F = skGrid / skTiles, rem = skGrid % skTiles
v_cmpx_eq_u32 exec, v13, s[sgprskTiles]            // F = skGrid / skTiles, rem = skGrid % skTiles
v_add_u32 v12, 1, v12                              // F = skGrid / skTiles, rem = skGrid % skTiles
v_mov_b32 v13, 0                                   // F = skGrid / skTiles, rem = skGrid % skTiles
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprskTiles]            // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s[sgprskTiles]             // re-calculate remainder
v_sub_u32 v13, s[sgprskGrid], v13                  // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v12                       // quotient
v_readfirstlane_b32 s14, v13                       // remainder
s_cmp_eq_u32 s14, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters          // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters                 // ragged -> global mapping
label_SK_AssignNoTiles:
label_SK_GlobalExtraIters:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s14, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s13, s[sgprPersistentWorkGroupIndex], s14 // StreamK starting iteration (case: before extra iters)
s_add_u32 s14, s13, s14                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s12  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s13, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s14, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone                  // skip per-tile path
label_SK_PerTileExtraIters:
s_mov_b32 s12, s13                                 // F = skGrid / skTiles
v_cvt_f32_u32 v12, s12                             // q = w/F, s = w%F
v_rcp_iflag_f32 v12, v12                           // q = w/F, s = w%F
v_cvt_f32_u32 v13, s[sgprPersistentWorkGroupIndex] // q = w/F, s = w%F
v_mul_f32 v12, v12, v13                            // q = w/F, s = w%F
v_cvt_u32_f32 v12, v12                             // q = w/F, s = w%F
v_mul_u32_u24 v13, v12, s12                        // q = w/F, s = w%F
v_sub_u32 v13, s[sgprPersistentWorkGroupIndex], v13 // q = w/F, s = w%F
v_cmpx_eq_u32 exec, v13, s12                       // q = w/F, s = w%F
v_add_u32 v12, 1, v12                              // q = w/F, s = w%F
v_mov_b32 v13, 0                                   // q = w/F, s = w%F
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s12                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s12                        // re-calculate remainder
v_sub_u32 v13, s[sgprPersistentWorkGroupIndex], v13 // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v12                       // quotient
v_readfirstlane_b32 s14, v13                       // remainder
s_mul_i32 s[sgprPersistentIteration], s13, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s13, s14, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s13 // q*I + s*W
s_mul_i32 s13, s12, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s13, s[sgprItersPerTile], s13            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s12, s14, s13                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s14, s13                              // s < remI?
s_cselect_b32 s12, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s12 // end = start + W + (s < remI)
label_SK_AssignItersDone:
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile] // Total SK iters
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s12 // Cap ending iter at total SK iters
label_SK_InitDone:
s_mul_i32 s12, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s12, s12, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s12, s12, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_cmp_lt_u32 s[sgprPersistentIteration], s12       // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_0                    // Only branch on scc0
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s14, label_KernelEnd, 4                  // target branch offset
s_add_u32 s12, s12, s14                            // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_setpc_b64 s[12:13]                               // branch to label_KernelEnd
label_NoBranch_0:

/******************************************/
/* Persistent Loop Start                  */
/******************************************/
label_PersistentLoopStart:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
/* StreamK calculate tile idx and map to WG */
s_mul_hi_u32 s13, s[sgprPersistentIteration], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s14, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s12, s[sgprPersistentIteration], s14     // s_magic mul, div alg 2
s_add_u32 s12, s12, s13
s_and_b32 s14, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s12, s12, s14                           // sMagicDiv Alg 2
s_mul_i32 s13, s12, s[sgprItersPerTile]            // Tile start iteration
s_add_u32 s14, s13, s[sgprItersPerTile]            // Tile end iteration
s_sub_u32 s[sgprStreamKLocalStart], s[sgprPersistentIteration], s13 // Local iteration start
s_min_u32 s[sgprStreamKLocalEnd], s[sgprPersistentIterationEnd], s14 // 1. (Local) iteration end (SK tile)
s_sub_u32 s[sgprStreamKLocalEnd], s[sgprStreamKLocalEnd], s13 // 2. Local iteration end (SK tile)
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitUpdate                // Jump to single kernel update
s_mov_b32 s13, s[sgprPersistentIterationEnd]       // Parallel reduction, work contained to single partial tile
s_branch label_SK_UpdateDone                       // Done update for parallel reduction
label_SK_SplitUpdate:
s_mul_i32 s15, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s15, s15, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_sub_u32 s15, s15, s[sgprskTiles]                 // dpTiles = totalTiles - skTiles
s_mul_i32 s15, s15, s[sgprItersPerTile]            // dpSectionSize = dpTiles * ItersPerTile
s_mul_i32 s13, s[sgprskGrid], s[sgprItersPerTile]  // DP iterations shift
s_add_u32 s13, s13, s[sgprPersistentIteration]     // Add DP shift
s_cmp_lt_u32 s13, s15                              // Check if still in DP section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s13, s14                                 // SK iterations shift
s_cmp_le_u32 s15, s[sgprPersistentIteration]       // Check if continuing in SK section
s_cbranch_scc1 label_SK_UpdateDone                 // Done update
s_mov_b32 s16, s15                                 // park dpSectionSize
s_mov_b32 s17, s12                                 // park current tile idx
s_mul_i32 s12, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s13, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s12, s12, s13                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global first-E mapping
s_cbranch_scc0 label_SK_GlobalExtraIters_1         // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_AssignNoTiles_1            // no SK tiles -> global mapping
v_cvt_f32_u32 v12, s[sgprskTiles]                  // F = skGrid / skTiles, rem = skGrid % skTiles
v_rcp_iflag_f32 v12, v12                           // F = skGrid / skTiles, rem = skGrid % skTiles
v_cvt_f32_u32 v13, s[sgprskGrid]                   // F = skGrid / skTiles, rem = skGrid % skTiles
v_mul_f32 v12, v12, v13                            // F = skGrid / skTiles, rem = skGrid % skTiles
v_cvt_u32_f32 v12, v12                             // F = skGrid / skTiles, rem = skGrid % skTiles
v_mul_u32_u24 v13, v12, s[sgprskTiles]             // F = skGrid / skTiles, rem = skGrid % skTiles
v_sub_u32 v13, s[sgprskGrid], v13                  // F = skGrid / skTiles, rem = skGrid % skTiles
v_cmpx_eq_u32 exec, v13, s[sgprskTiles]            // F = skGrid / skTiles, rem = skGrid % skTiles
v_add_u32 v12, 1, v12                              // F = skGrid / skTiles, rem = skGrid % skTiles
v_mov_b32 v13, 0                                   // F = skGrid / skTiles, rem = skGrid % skTiles
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprskTiles]            // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s[sgprskTiles]             // re-calculate remainder
v_sub_u32 v13, s[sgprskGrid], v13                  // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v12                       // quotient
v_readfirstlane_b32 s14, v13                       // remainder
s_cmp_eq_u32 s14, 0                                // skGrid % skTiles == 0?
s_cbranch_scc1 label_SK_PerTileExtraIters_1        // all-partial -> per-tile extras
s_branch label_SK_GlobalExtraIters_1               // ragged -> global mapping
label_SK_AssignNoTiles_1:
label_SK_GlobalExtraIters_1:
s_mul_i32 s[sgprPersistentIteration], s[sgprPersistentWorkGroupIndex], s[sgprSKItersPerWG] // StreamK starting iteration (case: after extra iters)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // Add extra iters
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // StreamK ending iteration (case: after extra iters)
s_add_u32 s14, s[sgprSKItersPerWG], 1              // Spread out extra iterations
s_mul_i32 s13, s[sgprPersistentWorkGroupIndex], s14 // StreamK starting iteration (case: before extra iters)
s_add_u32 s14, s13, s14                            // StreamK ending iteration (case: before extra iters)
s_cmp_lt_u32 s[sgprPersistentWorkGroupIndex], s12  // Check if lane gets an extra iteration
s_cselect_b32 s[sgprPersistentIteration], s13, s[sgprPersistentIteration] // Set start iter
s_cselect_b32 s[sgprPersistentIterationEnd], s14, s[sgprPersistentIterationEnd] // Set end iter
s_branch label_SK_AssignItersDone_1                // skip per-tile path
label_SK_PerTileExtraIters_1:
s_mov_b32 s12, s13                                 // F = skGrid / skTiles
v_cvt_f32_u32 v12, s12                             // q = w/F, s = w%F
v_rcp_iflag_f32 v12, v12                           // q = w/F, s = w%F
v_cvt_f32_u32 v13, s[sgprPersistentWorkGroupIndex] // q = w/F, s = w%F
v_mul_f32 v12, v12, v13                            // q = w/F, s = w%F
v_cvt_u32_f32 v12, v12                             // q = w/F, s = w%F
v_mul_u32_u24 v13, v12, s12                        // q = w/F, s = w%F
v_sub_u32 v13, s[sgprPersistentWorkGroupIndex], v13 // q = w/F, s = w%F
v_cmpx_eq_u32 exec, v13, s12                       // q = w/F, s = w%F
v_add_u32 v12, 1, v12                              // q = w/F, s = w%F
v_mov_b32 v13, 0                                   // q = w/F, s = w%F
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s12                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s12                        // re-calculate remainder
v_sub_u32 v13, s[sgprPersistentWorkGroupIndex], v13 // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v12                       // quotient
v_readfirstlane_b32 s14, v13                       // remainder
s_mul_i32 s[sgprPersistentIteration], s13, s[sgprItersPerTile] // q * ItersPerTile
s_mul_i32 s13, s14, s[sgprSKItersPerWG]            // s * SKItersPerWG
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s13 // q*I + s*W
s_mul_i32 s13, s12, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s13, s[sgprItersPerTile], s13            // remI = ItersPerTile - F*SKItersPerWG
s_min_u32 s12, s14, s13                            // min(s, remI)
s_add_u32 s[sgprPersistentIteration], s[sgprPersistentIteration], s12 // start = q*I + s*W + min(s, remI)
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIteration], s[sgprSKItersPerWG] // start + SKItersPerWG
s_cmp_lt_u32 s14, s13                              // s < remI?
s_cselect_b32 s12, 1, 0                            // extra iter within tile
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s12 // end = start + W + (s < remI)
label_SK_AssignItersDone_1:
s_mov_b32 s12, s17                                 // restore current tile idx
s_add_u32 s13, s[sgprPersistentIteration], s16     // Offset to start of SK section
s_add_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s16 // Offset to start of SK section
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // totalTiles = nwg0 * nwg1
s_mul_i32 s16, s16, s[sgprSizesFree+2]             // totalTiles *= batch dim 0
s_mul_i32 s16, s16, s[sgprItersPerTile]            // totalIters = totalTiles * itersPerTile
s_min_u32 s[sgprPersistentIterationEnd], s[sgprPersistentIterationEnd], s16 // Cap ending iter at total SK iters
s_cmp_lt_u32 s[sgprPersistentIteration], s16       // Make sure there's work to do
s_cbranch_scc1 label_NoBranch_2                    // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_KernelEnd, 4                  // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_KernelEnd
label_NoBranch_2:
label_SK_UpdateDone:
s_mov_b32 s[sgprPersistentIteration], s13          // Store current iteration
/* Map persistent tile index to wg0/1/2 */
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // Total tiles
v_cvt_f32_u32 v12, s13                             // TileID // nWG0*nWG1
v_rcp_iflag_f32 v12, v12                           // TileID // nWG0*nWG1
v_cvt_f32_u32 v13, s12                             // TileID // nWG0*nWG1
v_mul_f32 v12, v12, v13                            // TileID // nWG0*nWG1
v_cvt_u32_f32 v12, v12                             // TileID // nWG0*nWG1
v_mul_u32_u24 v13, v12, s13                        // TileID // nWG0*nWG1
v_sub_u32 v13, s12, v13                            // TileID // nWG0*nWG1
v_cmpx_eq_u32 exec, v13, s13                       // TileID // nWG0*nWG1
v_add_u32 v12, 1, v12                              // TileID // nWG0*nWG1
v_mov_b32 v13, 0                                   // TileID // nWG0*nWG1
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s13                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s13                        // re-calculate remainder
v_sub_u32 v13, s12, v13                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup2], v12         // quotient
v_readfirstlane_b32 s14, v13                       // remainder
v_cvt_f32_u32 v12, s[sgprNumWorkGroups0]           // TileID // nWG0
v_rcp_iflag_f32 v12, v12                           // TileID // nWG0
v_cvt_f32_u32 v13, s14                             // TileID // nWG0
v_mul_f32 v12, v12, v13                            // TileID // nWG0
v_cvt_u32_f32 v12, v12                             // TileID // nWG0
v_mul_u32_u24 v13, v12, s[sgprNumWorkGroups0]      // TileID // nWG0
v_sub_u32 v13, s14, v13                            // TileID // nWG0
v_cmpx_eq_u32 exec, v13, s[sgprNumWorkGroups0]     // TileID // nWG0
v_add_u32 v12, 1, v12                              // TileID // nWG0
v_mov_b32 v13, 0                                   // TileID // nWG0
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s[sgprNumWorkGroups0]      // re-calculate remainder
v_sub_u32 v13, s14, v13                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v12         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v13         // remainder

v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck                  // branch if s[Alpha] != 0
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_4                    // Only branch on scc0
s_getpc_b64 s[16:17]                               // addr of next instr
s_add_i32 s18, label_PersistentLoopClose, 4        // target branch offset
s_add_u32 s16, s16, s18                            // add target branch offset
s_addc_u32 s17, s17, 0                             // add high and carry
s_setpc_b64 s[16:17]                               // branch to label_PersistentLoopClose
label_NoBranch_4:
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
v_cvt_f64_u32 v[12:13], s12                        // s13 = s[sgprWorkGroup0] / s12
v_rcp_f64 v[12:13], v[12:13]                       // s13 = s[sgprWorkGroup0] / s12
v_cvt_f64_u32 v[14:15], s[sgprWorkGroup0]          // s13 = s[sgprWorkGroup0] / s12
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s13 = s[sgprWorkGroup0] / s12
v_cvt_u32_f64 v12, v[12:13]                        // s13 = s[sgprWorkGroup0] / s12
v_mul_lo_u32 v13, v12, s12                         // s13 = s[sgprWorkGroup0] / s12
v_sub_u32 v14, s[sgprWorkGroup0], v13              // s13 = s[sgprWorkGroup0] / s12
v_cmpx_ge_u32 exec, v14, s12                       // s13 = s[sgprWorkGroup0] / s12
v_add_u32 v12, v12, 1                              // s13 = s[sgprWorkGroup0] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v12                       // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup0], s16              // WorkGroup0=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s16, s16, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[12:13], s12                        // s14 = s[sgprNumWorkGroups0] / s12
v_rcp_f64 v[12:13], v[12:13]                       // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_f64_u32 v[14:15], s[sgprNumWorkGroups0]      // s14 = s[sgprNumWorkGroups0] / s12
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s14 = s[sgprNumWorkGroups0] / s12
v_cvt_u32_f64 v12, v[12:13]                        // s14 = s[sgprNumWorkGroups0] / s12
v_mul_lo_u32 v13, v12, s12                         // s14 = s[sgprNumWorkGroups0] / s12
v_sub_u32 v14, s[sgprNumWorkGroups0], v13          // s14 = s[sgprNumWorkGroups0] / s12
v_cmpx_ge_u32 exec, v14, s12                       // s14 = s[sgprNumWorkGroups0] / s12
v_add_u32 v12, v12, 1                              // s14 = s[sgprNumWorkGroups0] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s14, v12                       // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups0], s15          // NumWorkGroups0=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[12:13], s14                        // s[sgprWorkGroup1] = s16 / s14
v_rcp_f64 v[12:13], v[12:13]                       // s[sgprWorkGroup1] = s16 / s14
v_cvt_f64_u32 v[14:15], s16                        // s[sgprWorkGroup1] = s16 / s14
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s[sgprWorkGroup1] = s16 / s14
v_cvt_u32_f64 v12, v[12:13]                        // s[sgprWorkGroup1] = s16 / s14
v_mul_lo_u32 v13, v12, s14                         // s[sgprWorkGroup1] = s16 / s14
v_sub_u32 v14, s16, v13                            // s[sgprWorkGroup1] = s16 / s14
v_cmpx_ge_u32 exec, v14, s14                       // s[sgprWorkGroup1] = s16 / s14
v_add_u32 v12, v12, 1                              // s[sgprWorkGroup1] = s16 / s14
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v13, v12, s14                         // s[sgprWorkGroup1] = s16 / s14
v_sub_u32 v14, s16, v13                            // s[sgprWorkGroup1] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup1], v12         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v14         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s14 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s16, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s13, s13, s12                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s13 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s12, s12                                 // WGM
v_cvt_f64_u32 v[12:13], s12                        // s13 = s[sgprWorkGroup1] / s12
v_rcp_f64 v[12:13], v[12:13]                       // s13 = s[sgprWorkGroup1] / s12
v_cvt_f64_u32 v[14:15], s[sgprWorkGroup1]          // s13 = s[sgprWorkGroup1] / s12
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s13 = s[sgprWorkGroup1] / s12
v_cvt_u32_f64 v12, v[12:13]                        // s13 = s[sgprWorkGroup1] / s12
v_mul_lo_u32 v13, v12, s12                         // s13 = s[sgprWorkGroup1] / s12
v_sub_u32 v14, s[sgprWorkGroup1], v13              // s13 = s[sgprWorkGroup1] / s12
v_cmpx_ge_u32 exec, v14, s12                       // s13 = s[sgprWorkGroup1] / s12
v_add_u32 v12, v12, 1                              // s13 = s[sgprWorkGroup1] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s13, v12                       // quotient
s_mul_i32 s16, s13, s12                            // quotient * non-magic divisor
s_sub_u32 s16, s[sgprWorkGroup1], s16              // WorkGroup1=remainder
s_mul_i32 s16, s16, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s16, s16, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[12:13], s12                        // s14 = s[sgprNumWorkGroups1] / s12
v_rcp_f64 v[12:13], v[12:13]                       // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_f64_u32 v[14:15], s[sgprNumWorkGroups1]      // s14 = s[sgprNumWorkGroups1] / s12
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s14 = s[sgprNumWorkGroups1] / s12
v_cvt_u32_f64 v12, v[12:13]                        // s14 = s[sgprNumWorkGroups1] / s12
v_mul_lo_u32 v13, v12, s12                         // s14 = s[sgprNumWorkGroups1] / s12
v_sub_u32 v14, s[sgprNumWorkGroups1], v13          // s14 = s[sgprNumWorkGroups1] / s12
v_cmpx_ge_u32 exec, v14, s12                       // s14 = s[sgprNumWorkGroups1] / s12
v_add_u32 v12, v12, 1                              // s14 = s[sgprNumWorkGroups1] / s12
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s14, v12                       // quotient
s_mul_i32 s15, s12, s14                            // quotient * non-magic divisor
s_sub_u32 s15, s[sgprNumWorkGroups1], s15          // NumWorkGroups1=remainder
s_cmp_eq_u32 s15, 0                                // remainder == 0 ?
s_cmov_b32 s15, s12                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s13, s14                              // blockId >= numFullBlocks ?
s_cselect_b32 s14, s15, s12
v_cvt_f64_u32 v[12:13], s14                        // s[sgprWorkGroup0] = s16 / s14
v_rcp_f64 v[12:13], v[12:13]                       // s[sgprWorkGroup0] = s16 / s14
v_cvt_f64_u32 v[14:15], s16                        // s[sgprWorkGroup0] = s16 / s14
v_mul_f64 v[12:13], v[12:13], v[14:15]             // s[sgprWorkGroup0] = s16 / s14
v_cvt_u32_f64 v12, v[12:13]                        // s[sgprWorkGroup0] = s16 / s14
v_mul_lo_u32 v13, v12, s14                         // s[sgprWorkGroup0] = s16 / s14
v_sub_u32 v14, s16, v13                            // s[sgprWorkGroup0] = s16 / s14
v_cmpx_ge_u32 exec, v14, s14                       // s[sgprWorkGroup0] = s16 / s14
v_add_u32 v12, v12, 1                              // s[sgprWorkGroup0] = s16 / s14
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v13, v12, s14                         // s[sgprWorkGroup0] = s16 / s14
v_sub_u32 v14, s16, v13                            // s[sgprWorkGroup0] = s16 / s14
v_readfirstlane_b32 s[sgprWorkGroup0], v12         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v14         // remainder
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
v_and_b32 v13, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v12, 15, v13                             // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v12, 2, v12                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v13, 4, v13                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v12, v13, 9, v12                    // 5. K offset: lrKOffset = kIdx * mStride(512); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v16, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v16, 1, v16                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v12, v16, 6, v12                    // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(64); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v14, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v13, 15, v14                             // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v13, 2, v13                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(4)
v_lshrrev_b32 v14, 4, v14                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v13, v14, 9, v13                    // 5. K offset: lrKOffset = kIdx * mStride(512); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v15, 7, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v15, 1, v15                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v13, v15, 6, v13                    // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(64); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v14, 6, v[vgprSerial]                // 14 = Serial / 64
v_lshrrev_b32 v14, 2, v14                          // LSU offset: Get LSU wave_id
s_mov_b32 s12, 8192                                // LSU offset: stride = lsuStride(64)*(MT0(128) + PAD0(0))
v_mul_lo_u32 v14, s12, v14                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v14, v12          // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)

/* local read addresses: final offsets b */
v_lshrrev_b32 v12, 6, v[vgprSerial]                // 12 = Serial / 64
v_lshrrev_b32 v12, 2, v12                          // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64)*(MT1(128) + PAD1(0)) (dup assign opt.)
v_mul_lo_u32 v12, s12, v12                         // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v12, v13          // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x4000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v13 = A-unroll = serial/LVCA */
v_lshrrev_b32 v13, 4, v[vgprSerial]                // 13 = Serial / 16
v_and_b32 v12, 15, v[vgprSerial]                   // 12 = Serial % 16
/* tile *= glvw */
v_lshlrev_b32 v12, 3, v12                          // v12 = v12 * 8
v_mov_b32 v16, v13                                 // copy for GlobalSplitU
/* LVCB = 8 */
/* v15 = B-unroll = serial%LVCB */
v_lshrrev_b32 v14, 3, v[vgprSerial]                // 14 = Serial / 8
v_and_b32 v15, 7, v[vgprSerial]                    // 15 = Serial % 8
/* unroll *= glvw */
v_lshlrev_b32 v15, 3, v15                          // v15 = v15 * 8
v_mov_b32 v17, v15                                 // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v16 */
/* lwaUnrollAssignmentB = v17 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v16    // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v12, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v17    // lwBL**(MTB + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v14, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(MT1J+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x4000, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=16384

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v12 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v14 */

/* global read addresses: unroll assignment a */
/* v13 */

/* global read addresses: unroll assignment b */
/* v15 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */
v_mov_b32 v18, v12                                 // groA0I_0

/* global read addresses: tile offsets b */
v_mov_b32 v19, v14                                 // groB1J_0
v_add_co_u32 v20, vcc, 32, v19                     // groB1J_1 += LSPB
v_add_co_u32 v21, vcc, 32, v20                     // groB1J_2 += LSPB
v_add_co_u32 v22, vcc, 32, v21                     // groB1J_3 += LSPB

/* global read addresses: unroll offsets a */
v_mov_b32 v23, v13                                 // groAL_0
v_add_co_u32 v24, vcc, 16, v23                     // groAL_1 + LSPA
v_add_co_u32 v25, vcc, 16, v24                     // groAL_2 + LSPA
v_add_co_u32 v26, vcc, 16, v25                     // groAL_3 + LSPA

/* global read addresses: unroll offsets b */
v_mov_b32 v27, v15                                 // groBL_0

/* global read addresses: shift a */
s_mul_i32 s12, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_sub_u32 s12, s[sgprSizeI], s12                   // edge = Size0I - WG*MT
s_sub_u32 s12, s12, 8                              // edge -= margin(8)
v_mov_b32 v28, s12                                 // edge vgpr = Size0I- WG*MT - margin(8)
v_min_i32 v18, v28, v18                            // offset = (offset < edge) ? offset(v18) : edge(v28)

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mul_i32 s12, s[sgprStreamKLocalStart], 64        // StreamK tile start offset
s_mul_hi_u32 s13, s12, s[sgprStrideAL]             // StreamK tile start offset
s_mul_i32 s12, s12, s[sgprStrideAL]                // StreamK tile start offset
s_add_u32 s14, s14, s12                            // accum GsuOffset term to tilestart
s_addc_u32 s15, s15, s13                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s12, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s13, constStrideA0I, s12              // stride x (size-1)
s_mul_i32 s12, constStrideA0I, s12                 // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s12, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s13, s[sgprStrideAL], s12             // stride x (size-1)
s_mul_i32 s12, s[sgprStrideAL], s12                // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s14 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s15 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s12, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s12, s12, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s13, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdA:sgprSrdA+1], s[12:13], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_dwordx2 s[12:13], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s14, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s15, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s13, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s15   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s15, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s14, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
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
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s12, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s12, s12, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s13, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdB:sgprSrdB+1], s[12:13], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_dwordx2 s[12:13], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x98 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s12        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s13       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s14, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s15, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s13, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s12, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s14, s14, s12                            // accum wg term to tilestart
s_addc_u32 s15, s15, s13                           // accum wg term to tilestart
s_lshl_b64 s[14:15], s[14:15], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s14    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s15   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v28, s[sgprStrideAL], v[23]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc, v[18], v[28+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v28, s[sgprStrideAL], v[24]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc, v[18], v[28+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v28, s[sgprStrideAL], v[25]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc, v[18], v[28+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetA+2+0], 0x8, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v28, s[sgprStrideAL], v[26]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc, v[18], v[28+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetA+3+0], 0x8, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v23, s[sgprStrideB1J], v[19]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc, v[27], v[23+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mul_lo_u32 v23, s[sgprStrideB1J], v[20]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc, v[27], v[23+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1] //  (multiple bpe)
v_mul_lo_u32 v23, s[sgprStrideB1J], v[21]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc, v[27], v[23+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2] //  (multiple bpe)
v_mul_lo_u32 v23, s[sgprStrideB1J], v[22]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc, v[27], v[23+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
s_mul_i32 s[sgprGlobalReadIncsA+0], 128, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_mov_b32 s[sgprGlobalReadIncsB+0], 128            // incrB (unrollIdx)
/* declare loop num iterations */
s_sub_u32 s[sgprLoopCounterL], s[sgprStreamKLocalEnd], s[sgprStreamKLocalStart] // StreamK loop counter = localEnd - localStart
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_SKAlphaCheck_1                // branch if s[Alpha] != 0
s_mov_b32 s[sgprLoopCounterL], 0                   // Skip iterations
label_SKAlphaCheck_1:
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
s_cbranch_scc0 label_StaggerUMapping
s_mov_b32 s12, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping:
s_cmp_eq_u32 s15, 0x2000
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s12, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s15, 0x4000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s12, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s15, 0x6000
s_cbranch_scc0 label_StaggerUMapping_3
s_mul_i32 s13, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s12, s12, s13
s_add_u32 s12, s12, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
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

/* addr += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_u32 s13, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s12, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s12        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s13       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s12 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s13 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* addr += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_u32 s13, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s12, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
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
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0

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
s_nop 1                                            // alpha >= numSgprPreload, wait for kern args before
s_and_b32 s52, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s52, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_RegularSrdInitializationD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmSrdInitiationD // Parallel Reduction for General Batched GEMM, Srd initialized to workspace
label_RegularSrdInitializationD:  /// Regular SRD initialization for non-General Batched GEMM for D
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_GeneralBatchedGemmSrdInitiationD:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationD_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_and_b32 s52, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s52, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_RegularSrdInitializationC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmSrdInitiationC // Parallel Reduction for General Batched GEMM, Srd initialized to workspace
label_RegularSrdInitializationC:  /// Regular SRD initialization for non-General Batched GEMM for C
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_GeneralBatchedGemmSrdInitiationC:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationC_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s52, 1
s_mov_b32 s53, 1
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_BPEDone                       // If synchronizer, use regular output BPE
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_BPEDone                       // If split == 1, use reguler output BPE
s_mov_b32 s52, 1
s_mov_b32 s53, 2
label_BPEDone:

s_mul_i32 s84, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s83, s84, s[sgprStrideC1J]            // ScaleC s84 by Stride
s_mul_i32 s82, s84, s[sgprStrideC1J]               // ScaleC s84 by Stride
s_lshl_b64 s[82:83], s[82:83], s52                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // add hi to SRD
s_mul_hi_u32 s83, s84, s[sgprStrideD1J]            // ScaleD s84 by Stride
s_mul_i32 s82, s84, s[sgprStrideD1J]               // ScaleD s84 by Stride
s_lshl_b64 s[82:83], s[82:83], s53                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // add hi to SRD

s_and_b32 s54, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s54, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s83, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s82, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[82:83], s[82:83], s52                 // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s82, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s82, s82, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s83, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[82:83], s[82:83], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // Offsetting within the Batch Matrix [Higher half of address]
s_load_dwordx2 s[82:83], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s54, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s54, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s83, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s82, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[82:83], s[82:83], s53                 // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s82, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s82, s82, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s83, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[82:83], s[82:83], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // Offsetting within the Batch Matrix [Higher half of address]
s_load_dwordx2 s[82:83], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_SK_SplitSrd                   // Skip this block if using single-kernel stream-k fixup
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_SK_SplitSrd                   // branch if split == 1
// Split Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * SplitIdx * bpe%s
s_mul_hi_u32 s83, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_mul_i32 s82, s[sgprSizesFree+0], s[sgprSkPartialIdx] // Free0
s_sub_u32 s84, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s84, s84, s[sgprSkPartialIdx]            // Free1
s_mul_hi_u32 s85, s84, s[sgprStrideC1J]            // Free1
s_mul_i32 s84, s84, s[sgprStrideC1J]               // Free1
s_add_u32 s82, s82, s84                            // Free1
s_addc_u32 s83, s83, s85                           // Free1
s_sub_u32 s84, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s84, s84, s[sgprSkPartialIdx]            // Free2
s_mul_hi_u32 s85, s84, s[sgprStrideCK]             // Free2
s_mul_i32 s84, s84, s[sgprStrideCK]                // Free2
s_add_u32 s82, s82, s84                            // Free2
s_addc_u32 s83, s83, s85                           // Free2
s_lshl_b64 s[82:83], s[82:83], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83       // add hi GSU offset to SRD
label_SK_SplitSrd:

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...64) from pool */

/* initC: remove ValuA/B vgpr buffer [12...92) from pool */
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
v_accvgpr_write acc16, 0                           // initC
v_accvgpr_write acc17, 0                           // initC
v_accvgpr_write acc18, 0                           // initC
v_accvgpr_write acc19, 0                           // initC
v_accvgpr_write acc20, 0                           // initC
v_accvgpr_write acc21, 0                           // initC
v_accvgpr_write acc22, 0                           // initC
v_accvgpr_write acc23, 0                           // initC
v_accvgpr_write acc24, 0                           // initC
v_accvgpr_write acc25, 0                           // initC
v_accvgpr_write acc26, 0                           // initC
v_accvgpr_write acc27, 0                           // initC
v_accvgpr_write acc28, 0                           // initC
v_accvgpr_write acc29, 0                           // initC
v_accvgpr_write acc30, 0                           // initC
v_accvgpr_write acc31, 0                           // initC
v_accvgpr_write acc32, 0                           // initC
v_accvgpr_write acc33, 0                           // initC
v_accvgpr_write acc34, 0                           // initC
v_accvgpr_write acc35, 0                           // initC
v_accvgpr_write acc36, 0                           // initC
v_accvgpr_write acc37, 0                           // initC
v_accvgpr_write acc38, 0                           // initC
v_accvgpr_write acc39, 0                           // initC
v_accvgpr_write acc40, 0                           // initC
v_accvgpr_write acc41, 0                           // initC
v_accvgpr_write acc42, 0                           // initC
v_accvgpr_write acc43, 0                           // initC
v_accvgpr_write acc44, 0                           // initC
v_accvgpr_write acc45, 0                           // initC
v_accvgpr_write acc46, 0                           // initC
v_accvgpr_write acc47, 0                           // initC
v_accvgpr_write acc48, 0                           // initC
v_accvgpr_write acc49, 0                           // initC
v_accvgpr_write acc50, 0                           // initC
v_accvgpr_write acc51, 0                           // initC
v_accvgpr_write acc52, 0                           // initC
v_accvgpr_write acc53, 0                           // initC
v_accvgpr_write acc54, 0                           // initC
v_accvgpr_write acc55, 0                           // initC
v_accvgpr_write acc56, 0                           // initC
v_accvgpr_write acc57, 0                           // initC
v_accvgpr_write acc58, 0                           // initC
v_accvgpr_write acc59, 0                           // initC
v_accvgpr_write acc60, 0                           // initC
v_accvgpr_write acc61, 0                           // initC
v_accvgpr_write acc62, 0                           // initC
v_accvgpr_write acc63, 0                           // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */

/* label_PrefetchGlobalLastIterEnd */
s_cbranch_scc0 label_NoBranch_6                    // Only branch on scc1
s_getpc_b64 s[52:53]                               // addr of next instr
s_add_i32 s54, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s52, s52, s54                            // add target branch offset
s_addc_u32 s53, s53, 0                             // add high and carry
s_setpc_b64 s[52:53]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_6:
s_waitcnt vmcnt(0)                                 // wait for global read
s_barrier                                          // For stream-k / persistent loop

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4096 sync LDS0
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8192 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8192 sync LDS0
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:12288 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12288 sync LDS0

/* local write b */
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:256 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 256 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:512 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:768 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 768 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2] offset:1024 // lwoB_0_4_0_0 = (4 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1024 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+2] offset:1280 // lwoB_0_5_0_0 = (5 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1280 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+3] offset:1536 // lwoB_0_6_0_0 = (6 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1536 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+3] offset:1792 // lwoB_0_7_0_0 = (7 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1792 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4] offset:64 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 64 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+4] offset:320 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 320 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+5] offset:576 // lwoB_0_2_1_0 = (2 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+5] offset:832 // lwoB_0_3_1_0 = (3 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 832 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6] offset:1088 // lwoB_0_4_1_0 = (4 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1088 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+6] offset:1344 // lwoB_0_5_1_0 = (5 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1344 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+7] offset:1600 // lwoB_0_6_1_0 = (6 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1600 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+7] offset:1856 // lwoB_0_7_1_0 = (7 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1856 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8] offset:128 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 128 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+8] offset:384 // lwoB_0_1_2_0 = (1 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 384 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+9] offset:640 // lwoB_0_2_2_0 = (2 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 640 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+9] offset:896 // lwoB_0_3_2_0 = (3 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 896 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10] offset:1152 // lwoB_0_4_2_0 = (4 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1152 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+10] offset:1408 // lwoB_0_5_2_0 = (5 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1408 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+11] offset:1664 // lwoB_0_6_2_0 = (6 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1664 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+11] offset:1920 // lwoB_0_7_2_0 = (7 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1920 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12] offset:192 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 192 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+12] offset:448 // lwoB_0_1_3_0 = (1 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 448 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+13] offset:704 // lwoB_0_2_3_0 = (2 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 704 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+13] offset:960 // lwoB_0_3_3_0 = (3 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 960 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14] offset:1216 // lwoB_0_4_3_0 = (4 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1216 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+14] offset:1472 // lwoB_0_5_3_0 = (5 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1472 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+15] offset:1728 // lwoB_0_6_3_0 = (6 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1728 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+15] offset:1984 // lwoB_0_7_3_0 = (7 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1984 sync LDS0

/* local write swap a */

/* local write swap b */
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
label_skipPGR2_2:
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
// Skip force waitcnt0
s_barrier                                          // LW to PLR, sync

/* local read prefetch a */
ds_read_b64 v[vgprValuA_X0_I0_D0+0:vgprValuA_X0_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D1+0:vgprValuA_X0_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D2+0:vgprValuA_X0_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D3+0:vgprValuA_X0_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read prefetch b */
ds_read_b64 v[vgprValuB_X0_I0_D0+0:vgprValuB_X0_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D1+0:vgprValuB_X0_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D2+0:vgprValuB_X0_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D3+0:vgprValuB_X0_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
/* N/A, lro->2048 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read inc b */
/* N/A, lro->2048 */
/* localReadDoCntA 1 localReadDoCntMXSA 0 localReadDoCntB 1 localReadDoCntMXSB 0 localReadDoCntM 0 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL == 1 (PGR>=2, not Suppress: single-loop -> toPGR1)
s_cbranch_scc1 label_toPGR1                        // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:17 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0_D1+0], v[vgprValuB_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0_D3+0], v[vgprValuB_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b64 v[vgprValuA_X1_I0_D0+0:vgprValuA_X1_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:4096 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X1_I0_D1+0:vgprValuA_X1_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:4352 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s52, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s53, s[sgprWrapUA+1], 0              // incUpper <- ?
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0_D1+1], v[vgprValuA_X0_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+5], v[vgprValuA_X0_I0_D3+1], v[vgprValuA_X0_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0_D1+1], v[vgprValuA_X0_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0_D3+1], v[vgprValuA_X0_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b64 v[vgprValuA_X1_I0_D2+0:vgprValuA_X1_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X1_I0_D3+0:vgprValuA_X1_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:4864 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s52        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s53       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s52 // limit -= inc)
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0_D1+0], v[vgprValuB_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0_D3+0], v[vgprValuB_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0_D1+1], v[vgprValuB_X0_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+5], v[vgprValuB_X0_I0_D3+1], v[vgprValuB_X0_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b64 v[vgprValuB_X1_I0_D0+0:vgprValuB_X1_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:4096 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X1_I0_D1+0:vgprValuB_X1_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:4352 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s53 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0_D1+1], v[vgprValuB_X0_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0_D3+1], v[vgprValuB_X0_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b64 v[vgprValuB_X1_I0_D2+0:vgprValuB_X1_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X1_I0_D3+0:vgprValuB_X1_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:4864 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s52, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s53, s[sgprWrapUB+1], 0              // incUpper <- ?
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X2_I0_D0+0:vgprValuA_X2_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:8192 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X2_I0_D1+0:vgprValuA_X2_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:8448 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s52        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s53       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s52 // limit -= inc)
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X2_I0_D2+0:vgprValuA_X2_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:8704 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X2_I0_D3+0:vgprValuA_X2_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:8960 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0 sync LDS0
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s53 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X2_I0_D0+0:vgprValuB_X2_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:8192 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X2_I0_D1+0:vgprValuB_X2_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:8448 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X2_I0_D2+0:vgprValuB_X2_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:8704 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X2_I0_D3+0:vgprValuB_X2_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:8960 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X3_I0_D0+0:vgprValuA_X3_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:12288 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X3_I0_D1+0:vgprValuA_X3_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:12544 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X3_I0_D2+0:vgprValuA_X3_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:12800 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X3_I0_D3+0:vgprValuA_X3_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:13056 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X3_I0_D0+0:vgprValuB_X3_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:12288 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X3_I0_D1+0:vgprValuB_X3_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:12544 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X3_I0_D2+0:vgprValuB_X3_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:12800 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X3_I0_D3+0:vgprValuB_X3_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:13056 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=4 */

/* iter 1 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:17 */
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(15)                              // wait for prior local read local write old=8, new=8 newLW=0 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0_D1+0], v[vgprValuB_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+1], v[vgprValuB_X1_I0_D3+0], v[vgprValuB_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:17  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0_D1+1], v[vgprValuA_X1_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+5], v[vgprValuA_X1_I0_D3+1], v[vgprValuA_X1_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0_D1+1], v[vgprValuA_X1_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+7], v[vgprValuA_X1_I0_D3+1], v[vgprValuA_X1_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:18  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0_D1+0], v[vgprValuB_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+3], v[vgprValuB_X1_I0_D3+0], v[vgprValuB_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0_D1+1], v[vgprValuB_X1_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+5], v[vgprValuB_X1_I0_D3+1], v[vgprValuB_X1_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:19  */
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0_D1+1], v[vgprValuB_X1_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+7], v[vgprValuB_X1_I0_D3+1], v[vgprValuB_X1_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:20  */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:21  */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:22  */
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:23  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4096 sync LDS0
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:24  */
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:25  */
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:26  */
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:27  */
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:28  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8192 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8192 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:29  */
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:30  */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:31  */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=4 */

/* iter 2 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:17 */
/*  mfmaIndex:32  */
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X2_I0+0], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+1], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+0], v[vgprValuB_X2_I0_D1+0], v[vgprValuB_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+1], v[vgprValuB_X2_I0_D3+0], v[vgprValuB_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+2], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+3], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:33  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:12288 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12288 sync LDS0
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X2_I0+4], v[vgprValuA_X2_I0_D1+1], v[vgprValuA_X2_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+5], v[vgprValuA_X2_I0_D3+1], v[vgprValuA_X2_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+6], v[vgprValuA_X2_I0_D1+1], v[vgprValuA_X2_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+7], v[vgprValuA_X2_I0_D3+1], v[vgprValuA_X2_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:34  */
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X2_I0+2], v[vgprValuB_X2_I0_D1+0], v[vgprValuB_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+3], v[vgprValuB_X2_I0_D3+0], v[vgprValuB_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+4], v[vgprValuB_X2_I0_D1+1], v[vgprValuB_X2_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+5], v[vgprValuB_X2_I0_D3+1], v[vgprValuB_X2_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:35  */
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X2_I0+6], v[vgprValuB_X2_I0_D1+1], v[vgprValuB_X2_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+7], v[vgprValuB_X2_I0_D3+1], v[vgprValuB_X2_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:36  */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:37  */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:38  */
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:39  */
/* sched write - iter 2 writesPerItem=8 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:256 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 256 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:512 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:768 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 768 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2] offset:1024 // lwoB_0_4_0_0 = (4 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1024 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+2] offset:1280 // lwoB_0_5_0_0 = (5 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1280 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+3] offset:1536 // lwoB_0_6_0_0 = (6 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1536 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+3] offset:1792 // lwoB_0_7_0_0 = (7 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1792 sync LDS0
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:40  */
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:41  */
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:42  */
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:43  */
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:44  */
/* sched write - iter 2 writesPerItem=8 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4] offset:64 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 64 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+4] offset:320 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 320 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+5] offset:576 // lwoB_0_2_1_0 = (2 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+5] offset:832 // lwoB_0_3_1_0 = (3 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 832 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6] offset:1088 // lwoB_0_4_1_0 = (4 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1088 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+6] offset:1344 // lwoB_0_5_1_0 = (5 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1344 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+7] offset:1600 // lwoB_0_6_1_0 = (6 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1600 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+7] offset:1856 // lwoB_0_7_1_0 = (7 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1856 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:45  */
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:46  */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:47  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]

/* iter 3 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:17 */
/*  mfmaIndex:48  */
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X3_I0+0], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+1], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+0], v[vgprValuB_X3_I0_D1+0], v[vgprValuB_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+1], v[vgprValuB_X3_I0_D3+0], v[vgprValuB_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+2], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+3], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:49  */
/* sched write - iter 3 writesPerItem=8 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8] offset:128 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 128 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+8] offset:384 // lwoB_0_1_2_0 = (1 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 384 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+9] offset:640 // lwoB_0_2_2_0 = (2 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 640 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+9] offset:896 // lwoB_0_3_2_0 = (3 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 896 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10] offset:1152 // lwoB_0_4_2_0 = (4 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1152 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+10] offset:1408 // lwoB_0_5_2_0 = (5 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1408 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+11] offset:1664 // lwoB_0_6_2_0 = (6 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1664 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+11] offset:1920 // lwoB_0_7_2_0 = (7 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1920 sync LDS0
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X3_I0+4], v[vgprValuA_X3_I0_D1+1], v[vgprValuA_X3_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+5], v[vgprValuA_X3_I0_D3+1], v[vgprValuA_X3_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+6], v[vgprValuA_X3_I0_D1+1], v[vgprValuA_X3_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+7], v[vgprValuA_X3_I0_D3+1], v[vgprValuA_X3_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:50  */
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X3_I0+2], v[vgprValuB_X3_I0_D1+0], v[vgprValuB_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+3], v[vgprValuB_X3_I0_D3+0], v[vgprValuB_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+4], v[vgprValuB_X3_I0_D1+1], v[vgprValuB_X3_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+5], v[vgprValuB_X3_I0_D3+1], v[vgprValuB_X3_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:51  */
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X3_I0+6], v[vgprValuB_X3_I0_D1+1], v[vgprValuB_X3_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+7], v[vgprValuB_X3_I0_D3+1], v[vgprValuB_X3_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:52  */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:53  */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:54  */
/* sched write - iter 3 writesPerItem=8 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12] offset:192 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 192 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+12] offset:448 // lwoB_0_1_3_0 = (1 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 448 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+13] offset:704 // lwoB_0_2_3_0 = (2 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 704 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+13] offset:960 // lwoB_0_3_3_0 = (3 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 960 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14] offset:1216 // lwoB_0_4_3_0 = (4 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1216 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+14] offset:1472 // lwoB_0_5_3_0 = (5 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1472 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+15] offset:1728 // lwoB_0_6_3_0 = (6 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1728 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+15] offset:1984 // lwoB_0_7_3_0 = (7 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1984 sync LDS0
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:55  */
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:56  */
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:57  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier                                          // PGR, and wait until LW done to sync
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:58  */
ds_read_b64 v[vgprValuA_X0_I0_D0+0:vgprValuA_X0_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D1+0:vgprValuA_X0_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:59  */
ds_read_b64 v[vgprValuA_X0_I0_D2+0:vgprValuA_X0_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D3+0:vgprValuA_X0_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:60  */
ds_read_b64 v[vgprValuB_X0_I0_D0+0:vgprValuB_X0_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D1+0:vgprValuB_X0_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:61  */
ds_read_b64 v[vgprValuB_X0_I0_D2+0:vgprValuB_X0_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D3+0:vgprValuB_X0_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:62  */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:63  */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]

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

/* iter 0 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:17 */
/*  mfmaIndex:0  */

/* Global Read IncA */

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s52, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s53, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s52        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s53       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s52 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s53 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* Global Read IncB */

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s52, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s53, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s52        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s53       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s52 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s53 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0_D1+0], v[vgprValuB_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0_D3+0], v[vgprValuB_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b64 v[vgprValuA_X1_I0_D0+0:vgprValuA_X1_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:4096 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X1_I0_D1+0:vgprValuA_X1_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:4352 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0_D1+1], v[vgprValuA_X0_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+5], v[vgprValuA_X0_I0_D3+1], v[vgprValuA_X0_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0_D1+1], v[vgprValuA_X0_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0_D3+1], v[vgprValuA_X0_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b64 v[vgprValuA_X1_I0_D2+0:vgprValuA_X1_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X1_I0_D3+0:vgprValuA_X1_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:4864 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0_D1+0], v[vgprValuB_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0_D3+0], v[vgprValuB_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0_D1+1], v[vgprValuB_X0_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+5], v[vgprValuB_X0_I0_D3+1], v[vgprValuB_X0_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b64 v[vgprValuB_X1_I0_D0+0:vgprValuB_X1_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:4096 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X1_I0_D1+0:vgprValuB_X1_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:4352 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0_D1+1], v[vgprValuB_X0_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0_D3+1], v[vgprValuB_X0_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b64 v[vgprValuB_X1_I0_D2+0:vgprValuB_X1_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X1_I0_D3+0:vgprValuB_X1_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:4864 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X2_I0_D0+0:vgprValuA_X2_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:8192 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X2_I0_D1+0:vgprValuA_X2_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:8448 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X2_I0_D2+0:vgprValuA_X2_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:8704 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X2_I0_D3+0:vgprValuA_X2_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:8960 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X2_I0_D0+0:vgprValuB_X2_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:8192 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X2_I0_D1+0:vgprValuB_X2_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:8448 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X2_I0_D2+0:vgprValuB_X2_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:8704 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X2_I0_D3+0:vgprValuB_X2_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:8960 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X3_I0_D0+0:vgprValuA_X3_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:12288 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X3_I0_D1+0:vgprValuA_X3_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:12544 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X3_I0_D2+0:vgprValuA_X3_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:12800 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X3_I0_D3+0:vgprValuA_X3_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:13056 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X3_I0_D0+0:vgprValuB_X3_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:12288 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X3_I0_D1+0:vgprValuB_X3_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:12544 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X3_I0_D2+0:vgprValuB_X3_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:12800 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X3_I0_D3+0:vgprValuB_X3_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:13056 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=4 */

/* iter 1 */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:17 */
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(15)                              // wait for prior local read local write old=8, new=8 newLW=0 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0_D1+0], v[vgprValuB_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+1], v[vgprValuB_X1_I0_D3+0], v[vgprValuB_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:17  */
/* schedule remaining localreads for one buffer scheduling */
/* localReadsVacancy: latencyLeft 5 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0_D1+1], v[vgprValuA_X1_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+5], v[vgprValuA_X1_I0_D3+1], v[vgprValuA_X1_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0_D1+1], v[vgprValuA_X1_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+7], v[vgprValuA_X1_I0_D3+1], v[vgprValuA_X1_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:18  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0_D1+0], v[vgprValuB_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+3], v[vgprValuB_X1_I0_D3+0], v[vgprValuB_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0_D1+1], v[vgprValuB_X1_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+5], v[vgprValuB_X1_I0_D3+1], v[vgprValuB_X1_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:19  */
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0_D1+1], v[vgprValuB_X1_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+7], v[vgprValuB_X1_I0_D3+1], v[vgprValuB_X1_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:20  */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:21  */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:22  */
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:23  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4096 sync LDS0
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:24  */
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:25  */
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:26  */
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:27  */
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:28  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8192 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8192 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:29  */
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:30  */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:31  */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=4 */

/* iter 2 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:17 */
/*  mfmaIndex:32  */
s_waitcnt lgkmcnt(11)                              // wait for prior local read local write old=8, new=11 newLW=3 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X2_I0+0], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+1], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+0], v[vgprValuB_X2_I0_D1+0], v[vgprValuB_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+1], v[vgprValuB_X2_I0_D3+0], v[vgprValuB_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+2], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+3], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:33  */
/* sched write - iter 2 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:12288 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12288 sync LDS0
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X2_I0+4], v[vgprValuA_X2_I0_D1+1], v[vgprValuA_X2_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+5], v[vgprValuA_X2_I0_D3+1], v[vgprValuA_X2_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+6], v[vgprValuA_X2_I0_D1+1], v[vgprValuA_X2_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+7], v[vgprValuA_X2_I0_D3+1], v[vgprValuA_X2_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:34  */
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X2_I0+2], v[vgprValuB_X2_I0_D1+0], v[vgprValuB_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+3], v[vgprValuB_X2_I0_D3+0], v[vgprValuB_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+4], v[vgprValuB_X2_I0_D1+1], v[vgprValuB_X2_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+5], v[vgprValuB_X2_I0_D3+1], v[vgprValuB_X2_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:35  */
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X2_I0+6], v[vgprValuB_X2_I0_D1+1], v[vgprValuB_X2_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+7], v[vgprValuB_X2_I0_D3+1], v[vgprValuB_X2_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:36  */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:37  */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:38  */
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:39  */
/* sched write - iter 2 writesPerItem=8 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:256 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 256 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:512 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:768 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 768 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2] offset:1024 // lwoB_0_4_0_0 = (4 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1024 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+2] offset:1280 // lwoB_0_5_0_0 = (5 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1280 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+3] offset:1536 // lwoB_0_6_0_0 = (6 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1536 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+3] offset:1792 // lwoB_0_7_0_0 = (7 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1792 sync LDS0
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:40  */
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:41  */
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:42  */
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:43  */
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:44  */
/* sched write - iter 2 writesPerItem=8 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4] offset:64 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 64 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+4] offset:320 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 320 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+5] offset:576 // lwoB_0_2_1_0 = (2 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+5] offset:832 // lwoB_0_3_1_0 = (3 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 832 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6] offset:1088 // lwoB_0_4_1_0 = (4 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1088 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+6] offset:1344 // lwoB_0_5_1_0 = (5 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1344 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+7] offset:1600 // lwoB_0_6_1_0 = (6 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1600 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+7] offset:1856 // lwoB_0_7_1_0 = (7 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1856 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:45  */
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:46  */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:47  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=4 */

/* iter 3 (swap and reset local write pointers iteration)  */
/*  grEndMfmaIndex:6, lwStartMfmaIndex:18, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:17 */
/*  mfmaIndex:48  */
s_waitcnt lgkmcnt(15)                              // wait for prior local read local write old=0, new=17 newLW=17 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X3_I0+0], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+1], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+0], v[vgprValuB_X3_I0_D1+0], v[vgprValuB_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+1], v[vgprValuB_X3_I0_D3+0], v[vgprValuB_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+2], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+3], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:49  */
/* sched write - iter 3 writesPerItem=8 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8] offset:128 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 128 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+8] offset:384 // lwoB_0_1_2_0 = (1 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 384 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+9] offset:640 // lwoB_0_2_2_0 = (2 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 640 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+9] offset:896 // lwoB_0_3_2_0 = (3 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 896 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10] offset:1152 // lwoB_0_4_2_0 = (4 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1152 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+10] offset:1408 // lwoB_0_5_2_0 = (5 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1408 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+11] offset:1664 // lwoB_0_6_2_0 = (6 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1664 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+11] offset:1920 // lwoB_0_7_2_0 = (7 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1920 sync LDS0
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X3_I0+4], v[vgprValuA_X3_I0_D1+1], v[vgprValuA_X3_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+5], v[vgprValuA_X3_I0_D3+1], v[vgprValuA_X3_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+6], v[vgprValuA_X3_I0_D1+1], v[vgprValuA_X3_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+7], v[vgprValuA_X3_I0_D3+1], v[vgprValuA_X3_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:50  */
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X3_I0+2], v[vgprValuB_X3_I0_D1+0], v[vgprValuB_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+3], v[vgprValuB_X3_I0_D3+0], v[vgprValuB_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+4], v[vgprValuB_X3_I0_D1+1], v[vgprValuB_X3_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+5], v[vgprValuB_X3_I0_D3+1], v[vgprValuB_X3_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:51  */
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X3_I0+6], v[vgprValuB_X3_I0_D1+1], v[vgprValuB_X3_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+7], v[vgprValuB_X3_I0_D3+1], v[vgprValuB_X3_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:52  */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:53  */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:54  */
/* sched write - iter 3 writesPerItem=8 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12] offset:192 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 192 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+12] offset:448 // lwoB_0_1_3_0 = (1 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 448 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+13] offset:704 // lwoB_0_2_3_0 = (2 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 704 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+13] offset:960 // lwoB_0_3_3_0 = (3 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 960 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14] offset:1216 // lwoB_0_4_3_0 = (4 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1216 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+14] offset:1472 // lwoB_0_5_3_0 = (5 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1472 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+15] offset:1728 // lwoB_0_6_3_0 = (6 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1728 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+15] offset:1984 // lwoB_0_7_3_0 = (7 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1984 sync LDS0
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:55  */

/* local write swap offsets a */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:56  */
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:57  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier                                          // noLoadLoop sync
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:58  */
ds_read_b64 v[vgprValuA_X0_I0_D0+0:vgprValuA_X0_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D1+0:vgprValuA_X0_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:59  */
ds_read_b64 v[vgprValuA_X0_I0_D2+0:vgprValuA_X0_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D3+0:vgprValuA_X0_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:60  */
ds_read_b64 v[vgprValuB_X0_I0_D0+0:vgprValuB_X0_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D1+0:vgprValuB_X0_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:61  */
ds_read_b64 v[vgprValuB_X0_I0_D2+0:vgprValuB_X0_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D3+0:vgprValuB_X0_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:62  */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:63  */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=4 */
label_toPGR1:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:55, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:54 */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0_D1+0], v[vgprValuB_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0_D3+0], v[vgprValuB_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b64 v[vgprValuA_X1_I0_D0+0:vgprValuA_X1_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:4096 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X1_I0_D1+0:vgprValuA_X1_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:4352 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0_D1+1], v[vgprValuA_X0_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+5], v[vgprValuA_X0_I0_D3+1], v[vgprValuA_X0_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0_D1+1], v[vgprValuA_X0_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0_D3+1], v[vgprValuA_X0_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b64 v[vgprValuA_X1_I0_D2+0:vgprValuA_X1_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:4608 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X1_I0_D3+0:vgprValuA_X1_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:4864 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0_D1+0], v[vgprValuB_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0_D3+0], v[vgprValuB_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0_D1+1], v[vgprValuB_X0_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+5], v[vgprValuB_X0_I0_D3+1], v[vgprValuB_X0_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:3  */
ds_read_b64 v[vgprValuB_X1_I0_D0+0:vgprValuB_X1_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:4096 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X1_I0_D1+0:vgprValuB_X1_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:4352 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0_D1+1], v[vgprValuB_X0_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0_D3+1], v[vgprValuB_X0_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:4  */
ds_read_b64 v[vgprValuB_X1_I0_D2+0:vgprValuB_X1_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:4608 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X1_I0_D3+0:vgprValuB_X1_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:4864 // L -> Reg lro=2048 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0 sync LDS0
/* localReadsVacancy: latencyLeft 1 */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X2_I0_D0+0:vgprValuA_X2_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:8192 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X2_I0_D1+0:vgprValuA_X2_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:8448 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X2_I0_D2+0:vgprValuA_X2_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:8704 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X2_I0_D3+0:vgprValuA_X2_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:8960 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X2_I0_D0+0:vgprValuB_X2_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:8192 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X2_I0_D1+0:vgprValuB_X2_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:8448 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:8  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X2_I0_D2+0:vgprValuB_X2_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:8704 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X2_I0_D3+0:vgprValuB_X2_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:8960 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X3_I0_D0+0:vgprValuA_X3_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:12288 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X3_I0_D1+0:vgprValuA_X3_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:12544 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuA_X3_I0_D2+0:vgprValuA_X3_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:12800 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X3_I0_D3+0:vgprValuA_X3_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:13056 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X3_I0_D0+0:vgprValuB_X3_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:12288 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X3_I0_D1+0:vgprValuB_X3_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:12544 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 5 */
ds_read_b64 v[vgprValuB_X3_I0_D2+0:vgprValuB_X3_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:12800 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X3_I0_D3+0:vgprValuB_X3_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:13056 // L -> Reg lro=6144 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0 sync LDS0
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=4 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:55, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:54 */
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(15)                              // wait for prior local read local write old=8, new=8 newLW=0 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0_D1+0], v[vgprValuB_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+1], v[vgprValuB_X1_I0_D3+0], v[vgprValuB_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:17  */
/* localReadsVacancy: latencyLeft 5 */
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X1_I0+4], v[vgprValuA_X1_I0_D1+1], v[vgprValuA_X1_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+5], v[vgprValuA_X1_I0_D3+1], v[vgprValuA_X1_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+6], v[vgprValuA_X1_I0_D1+1], v[vgprValuA_X1_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+7], v[vgprValuA_X1_I0_D3+1], v[vgprValuA_X1_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:18  */
/* localReadsVacancy: latencyLeft 5 */
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0_D1+0], v[vgprValuB_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+3], v[vgprValuB_X1_I0_D3+0], v[vgprValuB_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0_D1+1], v[vgprValuB_X1_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X1_I0+5], v[vgprValuB_X1_I0_D3+1], v[vgprValuB_X1_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:19  */
/* localReadsVacancy: latencyLeft 5 */
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0_D1+1], v[vgprValuB_X1_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X1_I0+7], v[vgprValuB_X1_I0_D3+1], v[vgprValuB_X1_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:20  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:21  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:22  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:23  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:24  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:25  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:26  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:27  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:28  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:29  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:30  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+4+0+0:vgprValuA_X1_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:31  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], v[vgprValuA_X1_I0+6+0+0:vgprValuA_X1_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=2 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=2 skipReadsIterB=1 readsPerIterB=4 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:55, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:54 */
/*  mfmaIndex:32  */
/* localReadsVacancy: latencyLeft 5 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=8, new=8 newLW=0 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X2_I0+0], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+1], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+0], v[vgprValuB_X2_I0_D1+0], v[vgprValuB_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+1], v[vgprValuB_X2_I0_D3+0], v[vgprValuB_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+2], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+3], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:33  */
/* localReadsVacancy: latencyLeft 5 */
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X2_I0+4], v[vgprValuA_X2_I0_D1+1], v[vgprValuA_X2_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+5], v[vgprValuA_X2_I0_D3+1], v[vgprValuA_X2_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+6], v[vgprValuA_X2_I0_D1+1], v[vgprValuA_X2_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+7], v[vgprValuA_X2_I0_D3+1], v[vgprValuA_X2_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:34  */
/* localReadsVacancy: latencyLeft 5 */
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X2_I0+2], v[vgprValuB_X2_I0_D1+0], v[vgprValuB_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+3], v[vgprValuB_X2_I0_D3+0], v[vgprValuB_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+4], v[vgprValuB_X2_I0_D1+1], v[vgprValuB_X2_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X2_I0+5], v[vgprValuB_X2_I0_D3+1], v[vgprValuB_X2_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:35  */
/* localReadsVacancy: latencyLeft 5 */
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X2_I0+6], v[vgprValuB_X2_I0_D1+1], v[vgprValuB_X2_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X2_I0+7], v[vgprValuB_X2_I0_D3+1], v[vgprValuB_X2_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:36  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:37  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:38  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:39  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X2_I0+2+0+0:vgprValuB_X2_I0+2+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:40  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:41  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:42  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:43  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:44  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:45  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:46  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+4+0+0:vgprValuA_X2_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:47  */
/* localReadsVacancy: latencyLeft 5 */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X2_I0+6+0+0:vgprValuB_X2_I0+6+0+0+1], v[vgprValuA_X2_I0+6+0+0:vgprValuA_X2_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=4 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=4 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:55, lwEndMfmaIndex:55  */
/*  numMfmaForLR:6, syncPlrMfmaIndex:57 , sync1LdsMfmaIndex:54 */
/*  mfmaIndex:48  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_perm_b32 v[vgprValuA_X3_I0+0], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+1], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+0], v[vgprValuB_X3_I0_D1+0], v[vgprValuB_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+1], v[vgprValuB_X3_I0_D3+0], v[vgprValuB_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+2], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+3], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:49  */
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_perm_b32 v[vgprValuA_X3_I0+4], v[vgprValuA_X3_I0_D1+1], v[vgprValuA_X3_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+5], v[vgprValuA_X3_I0_D3+1], v[vgprValuA_X3_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+6], v[vgprValuA_X3_I0_D1+1], v[vgprValuA_X3_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+7], v[vgprValuA_X3_I0_D3+1], v[vgprValuA_X3_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
/*  mfmaIndex:50  */
/* pack scheduling: packAIdx:6, packBIdx:2 */
v_perm_b32 v[vgprValuB_X3_I0+2], v[vgprValuB_X3_I0_D1+0], v[vgprValuB_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+3], v[vgprValuB_X3_I0_D3+0], v[vgprValuB_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+4], v[vgprValuB_X3_I0_D1+1], v[vgprValuB_X3_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X3_I0+5], v[vgprValuB_X3_I0_D3+1], v[vgprValuB_X3_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
/*  mfmaIndex:51  */
/* pack scheduling: packAIdx:8, packBIdx:2 */
v_perm_b32 v[vgprValuB_X3_I0+6], v[vgprValuB_X3_I0_D1+1], v[vgprValuB_X3_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X3_I0+7], v[vgprValuB_X3_I0_D3+1], v[vgprValuB_X3_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X3_I0+0+0+0:vgprValuB_X3_I0+0+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
/*  mfmaIndex:52  */
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
/*  mfmaIndex:53  */
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
/*  mfmaIndex:54  */
/* schedule remaining localreads for one buffer scheduling */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
/*  mfmaIndex:55  */
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X3_I0+2+0+0:vgprValuB_X3_I0+2+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
/*  mfmaIndex:56  */
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
/*  mfmaIndex:57  */
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
/*  mfmaIndex:58  */
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
/*  mfmaIndex:59  */
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X3_I0+4+0+0:vgprValuB_X3_I0+4+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
/*  mfmaIndex:60  */
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
/*  mfmaIndex:61  */
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
/*  mfmaIndex:62  */
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+4+0+0:vgprValuA_X3_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
/*  mfmaIndex:63  */
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X3_I0+6+0+0:vgprValuB_X3_I0+6+0+0+1], v[vgprValuA_X3_I0+6+0+0:vgprValuA_X3_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=4 */
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [12...92) to pool */

/* Tail: add address/G2L vgpr [92...124) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */

/* local write reset offsets b */
/* Check out VGPR (numG2LA,numG2LB,numG2LMXSA,numG2LMXSB,numG2LMetadata) = (16,16,0,0,0) */
.set vgprG2LA_BASE, 12
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 28
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 63, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 64
s_cmp_lt_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // Check if WG processes final iteration of tile
s_cmov_b32 s[sgprLoopCounterL], 0                  // This WG not completing tile
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
//  removeStagger A
s_sub_i32 s82, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s82, 0
s_cbranch_scc0 label_Negative_8
s_mul_hi_u32 s83, s82, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s82, s82, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_9
label_Negative_8:
s_abs_i32 s82, s82
s_mul_hi_u32 s83, s82, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s82, s82, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s82, s82, 0xffffffff
s_xor_b32 s83, s83, 0xffffffff
s_add_u32 s82, s82, 0x1
s_addc_u32 s83, s83, 0
label_MultiplyDone_9:
s_sub_u32 s82, s82, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s83, s83, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s82        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s83       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s82 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s83 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
//  removeStagger B
s_sub_i32 s82, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s82, 0
s_cbranch_scc0 label_Negative_10
s_mul_hi_u32 s83, s82, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s82, s82, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_11
label_Negative_10:
s_abs_i32 s82, s82
s_mul_hi_u32 s83, s82, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s82, s82, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s82, s82, 0xffffffff
s_xor_b32 s83, s83, 0xffffffff
s_add_u32 s82, s82, 0x1
s_addc_u32 s83, s83, 0
label_MultiplyDone_11:
s_sub_u32 s82, s82, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s83, s83, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s82        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s83       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s82 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s83 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* Tail global read A */
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* Update M0 for DTLDS */

/* Tail global read B */
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0

/* release sgprs that will not be used */
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
s_mul_i32 s62, s[sgprWorkGroup1], 128              // Calculate the remaining dimension along I/J direction.
s_sub_u32 s62, s[sgprSizeJ], s62                   // Calculate the remaining dimension along I/J direction.
s_mul_i32 s62, s62, 2                              // In bytes
s_and_b32 s74, s[sgprSizeL], 63                    // Calculate the remaining dimension along L direction.
s_lshr_b32 s84, s74, 0x6                           // Divided by lsc(64)
s_mul_hi_u32 s72, s62, s74                         // Calculate total number of valid elements.
s_mul_i32 s76, s62, s74                            // Calculate total number of valid elements.
s_cmp_gt_u32 s72, 0
s_cmov_b32 s76, 0xffffffff                         // If valid elements > max(U32), set the value to max
s_sub_u32 s74, s[sgprSizeJ], 1                     // sLoadTileIdx starts from 0
// Calculate SizeJ - 1 % MacroTile1
s_lshr_b32 s62, s74, 7                             // s62 = s74 / 128
s_and_b32 s62, 127, s74                            // s62 = s74 % 128
s_lshr_b32 s62, s62, 0x5                           // Divide lsp to get the load tile index
s_mul_i32 s62, s62, 1                              // Multiply nlc
s_add_i32 s62, s62, s84
s_and_b32 s74, 63, s[sgprSizesSum+0]               // s74 = s[sgprSizesSum+0] % 64
s_and_b32 s74, s74, 7                              // sLoadNum = (SizesSum+0 mod DU) & glvw
s_and_b32 s72, s74, 0x1
s_mov_b32 s77, 0                                   // Set loop count = 0

/* load single element for B */
label_LoadB:
s_cmp_eq_u32 s72, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_MergeB                        // Skip loading B
s_cmp_eq_u32 s62, 3
s_cbranch_scc1 label_LOAD_B3
s_cmp_eq_u32 s62, 2
s_cbranch_scc1 label_LOAD_B2
s_cmp_eq_u32 s62, 1
s_cbranch_scc1 label_LOAD_B1
label_LOAD_B0:
label_LOAD_B0_K1:
s_cmp_ge_u32 s74, 1
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 0 */
buffer_load_short_d16 v48, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B0_K3:
s_cmp_ge_u32 s74, 3
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 2 */
buffer_load_short_d16 v49, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B0_K5:
s_cmp_ge_u32 s74, 5
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 4 */
buffer_load_short_d16 v50, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B0_K7:
s_cmp_ge_u32 s74, 7
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 6 */
buffer_load_short_d16 v51, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeB
label_LOAD_B1:
label_LOAD_B1_K1:
s_cmp_ge_u32 s74, 1
s_cbranch_scc0 label_MergeB
/* g2l=4, load component 0 */
buffer_load_short_d16 v48, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B1_K3:
s_cmp_ge_u32 s74, 3
s_cbranch_scc0 label_MergeB
/* g2l=4, load component 2 */
buffer_load_short_d16 v49, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B1_K5:
s_cmp_ge_u32 s74, 5
s_cbranch_scc0 label_MergeB
/* g2l=4, load component 4 */
buffer_load_short_d16 v50, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B1_K7:
s_cmp_ge_u32 s74, 7
s_cbranch_scc0 label_MergeB
/* g2l=4, load component 6 */
buffer_load_short_d16 v51, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeB
label_LOAD_B2:
label_LOAD_B2_K1:
s_cmp_ge_u32 s74, 1
s_cbranch_scc0 label_MergeB
/* g2l=8, load component 0 */
buffer_load_short_d16 v48, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B2_K3:
s_cmp_ge_u32 s74, 3
s_cbranch_scc0 label_MergeB
/* g2l=8, load component 2 */
buffer_load_short_d16 v49, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B2_K5:
s_cmp_ge_u32 s74, 5
s_cbranch_scc0 label_MergeB
/* g2l=8, load component 4 */
buffer_load_short_d16 v50, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B2_K7:
s_cmp_ge_u32 s74, 7
s_cbranch_scc0 label_MergeB
/* g2l=8, load component 6 */
buffer_load_short_d16 v51, v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeB
label_LOAD_B3:
label_LOAD_B3_K1:
s_cmp_ge_u32 s74, 1
s_cbranch_scc0 label_MergeB
/* g2l=12, load component 0 */
buffer_load_short_d16 v48, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B3_K3:
s_cmp_ge_u32 s74, 3
s_cbranch_scc0 label_MergeB
/* g2l=12, load component 2 */
buffer_load_short_d16 v49, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
label_LOAD_B3_K5:
s_cmp_ge_u32 s74, 5
s_cbranch_scc0 label_MergeB
/* g2l=12, load component 4 */
buffer_load_short_d16 v50, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
label_LOAD_B3_K7:
s_cmp_ge_u32 s74, 7
s_cbranch_scc0 label_MergeB
/* g2l=12, load component 6 */
buffer_load_short_d16 v51, v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
s_branch label_MergeB

/* merge single element for B */
label_MergeB:
s_cmp_eq_u32 s72, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_CheckOtherLoadB               // Skip mergeing B
s_cmp_eq_u32 s62, 3
s_cbranch_scc1 label_MERGE_B3
s_cmp_eq_u32 s62, 2
s_cbranch_scc1 label_MERGE_B2
s_cmp_eq_u32 s62, 1
s_cbranch_scc1 label_MERGE_B1
label_MERGE_B0:
label_MERGE_B0_K1:
s_cmp_ge_u32 s74, 1
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v48     // HasEccHalf: pack
label_MERGE_B0_K3:
s_cmp_ge_u32 s74, 3
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v49     // HasEccHalf: pack
label_MERGE_B0_K5:
s_cmp_ge_u32 s74, 5
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v50     // HasEccHalf: pack
label_MERGE_B0_K7:
s_cmp_ge_u32 s74, 7
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v51     // HasEccHalf: pack
s_branch label_CheckOtherLoadB
label_MERGE_B1:
label_MERGE_B1_K1:
s_cmp_ge_u32 s74, 1
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+0], v[vgprG2LB+4+0], v48     // HasEccHalf: pack
label_MERGE_B1_K3:
s_cmp_ge_u32 s74, 3
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+1], v[vgprG2LB+4+1], v49     // HasEccHalf: pack
label_MERGE_B1_K5:
s_cmp_ge_u32 s74, 5
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+2], v[vgprG2LB+4+2], v50     // HasEccHalf: pack
label_MERGE_B1_K7:
s_cmp_ge_u32 s74, 7
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+3], v[vgprG2LB+4+3], v51     // HasEccHalf: pack
s_branch label_CheckOtherLoadB
label_MERGE_B2:
label_MERGE_B2_K1:
s_cmp_ge_u32 s74, 1
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+0], v[vgprG2LB+8+0], v48     // HasEccHalf: pack
label_MERGE_B2_K3:
s_cmp_ge_u32 s74, 3
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+1], v[vgprG2LB+8+1], v49     // HasEccHalf: pack
label_MERGE_B2_K5:
s_cmp_ge_u32 s74, 5
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+2], v[vgprG2LB+8+2], v50     // HasEccHalf: pack
label_MERGE_B2_K7:
s_cmp_ge_u32 s74, 7
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+8+3], v[vgprG2LB+8+3], v51     // HasEccHalf: pack
s_branch label_CheckOtherLoadB
label_MERGE_B3:
label_MERGE_B3_K1:
s_cmp_ge_u32 s74, 1
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+0], v[vgprG2LB+12+0], v48   // HasEccHalf: pack
label_MERGE_B3_K3:
s_cmp_ge_u32 s74, 3
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+1], v[vgprG2LB+12+1], v49   // HasEccHalf: pack
label_MERGE_B3_K5:
s_cmp_ge_u32 s74, 5
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+2], v[vgprG2LB+12+2], v50   // HasEccHalf: pack
label_MERGE_B3_K7:
s_cmp_ge_u32 s74, 7
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+12+3], v[vgprG2LB+12+3], v51   // HasEccHalf: pack
s_branch label_CheckOtherLoadB

/* reload loop for a: check if there's other load range need to be reloaded */
label_CheckOtherLoadA:

/* reload loop for b: check if there's other load range need to be reloaded */
label_CheckOtherLoadB:
s_cmp_eq_u32 s72, 0                                // Noneed to load single element for B?
s_cbranch_scc1 label_TailGlobalLoadEnd
s_add_u32 s77, s77, 1
s_cmp_eq_u32 s77, 4                                // Have reloaded all subtiles?
s_cbranch_scc1 label_TailGlobalLoadEnd
s_sub_i32 s62, s62, 1                              // Check the upper subtile
s_cmp_lt_i32 s62, 0
s_cselect_b32 s82, 4, 0                            // Back to the last subtile
s_add_i32 s62, s62, s82                            // If currently reload the first subtile,                                   check the last subtile next.
s_cmp_eq_u32 s62, 3
s_cbranch_scc1 label_B3
s_cmp_eq_u32 s62, 2
s_cbranch_scc1 label_B2
s_cmp_eq_u32 s62, 1
s_cbranch_scc1 label_B1
label_B0:
v_mov_b32 v44, v[vgprGlobalReadOffsetB+0]
s_branch label_CheckAddrB
label_B1:
v_mov_b32 v44, v[vgprGlobalReadOffsetB+1]
s_branch label_CheckAddrB
label_B2:
v_mov_b32 v44, v[vgprGlobalReadOffsetB+2]
s_branch label_CheckAddrB
label_B3:
v_mov_b32 v44, v[vgprGlobalReadOffsetB+3]
label_CheckAddrB:
v_sub_u32 v44, v44, 16                             // sub prepad
v_add_u32 v45, v44, 15                             // Calculate load range per thread
v_cmp_lt_i32 s[82:83], v44, s76                    // If loading start address < total valid bytes?
v_cmp_ge_i32 s[84:85], v45, s76                    // If loading end address >= total valid bytes?
s_and_b32 s82, s82, s84                            // Find threads which access the last element
s_and_b32 s83, s83, s85                            // Find thread that access the last element
s_add_u32 s82, s82, s83                            // Find thread that access the last element
s_cmp_lg_u32 s82, 0                                // Have threads access the last element?
s_cbranch_scc1 label_LoadB                         // Reload B

/* global read for tail done */
label_TailGlobalLoadEnd:
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip force waitcnt0
s_barrier

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4096 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4096 sync LDS0
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+8:vgprG2LA+8+3] offset:8192 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8192 sync LDS0
ds_write_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+12:vgprG2LA+12+3] offset:12288 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12288 sync LDS0

/* local write b */
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:256 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 256 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:512 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:768 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 768 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+2] offset:1024 // lwoB_0_4_0_0 = (4 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1024 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+2] offset:1280 // lwoB_0_5_0_0 = (5 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1280 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+3] offset:1536 // lwoB_0_6_0_0 = (6 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1536 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+3] offset:1792 // lwoB_0_7_0_0 = (7 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 1792 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4] offset:64 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 64 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+4] offset:320 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 320 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+5] offset:576 // lwoB_0_2_1_0 = (2 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+5] offset:832 // lwoB_0_3_1_0 = (3 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 832 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+6] offset:1088 // lwoB_0_4_1_0 = (4 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1088 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+6] offset:1344 // lwoB_0_5_1_0 = (5 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1344 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+7] offset:1600 // lwoB_0_6_1_0 = (6 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1600 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+7] offset:1856 // lwoB_0_7_1_0 = (7 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 1856 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8] offset:128 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 128 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+8] offset:384 // lwoB_0_1_2_0 = (1 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 384 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+9] offset:640 // lwoB_0_2_2_0 = (2 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 640 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+9] offset:896 // lwoB_0_3_2_0 = (3 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 896 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+10] offset:1152 // lwoB_0_4_2_0 = (4 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1152 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+10] offset:1408 // lwoB_0_5_2_0 = (5 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1408 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+11] offset:1664 // lwoB_0_6_2_0 = (6 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1664 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+11] offset:1920 // lwoB_0_7_2_0 = (7 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 1920 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12] offset:192 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 192 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+12] offset:448 // lwoB_0_1_3_0 = (1 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 448 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+13] offset:704 // lwoB_0_2_3_0 = (2 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 704 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+13] offset:960 // lwoB_0_3_3_0 = (3 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 960 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+14] offset:1216 // lwoB_0_4_3_0 = (4 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1216 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+14] offset:1472 // lwoB_0_5_3_0 = (5 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1472 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+15] offset:1728 // lwoB_0_6_3_0 = (6 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1728 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+15] offset:1984 // lwoB_0_7_3_0 = (7 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 1984 sync LDS0

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip force waitcnt0
s_barrier                                          // Tail loop LW->LR, sync LDS0
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 12
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X2_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X3_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X0_I0_D0_PACK, 20
.set vgprValuA_X0_I0_D0, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+2
.set vgprValuA_X0_I0_D2, vgprValuA_X0_I0_D0_PACK+4
.set vgprValuA_X0_I0_D3, vgprValuA_X0_I0_D0_PACK+6
.set vgprValuA_X1_I0_D0, vgprValuA_X0_I0_D0_PACK+8
.set vgprValuA_X1_I0_D1, vgprValuA_X0_I0_D0_PACK+10
.set vgprValuA_X1_I0_D2, vgprValuA_X0_I0_D0_PACK+12
.set vgprValuA_X1_I0_D3, vgprValuA_X0_I0_D0_PACK+14
.set vgprValuA_X2_I0_D0, vgprValuA_X0_I0_D0_PACK+16
.set vgprValuA_X2_I0_D1, vgprValuA_X0_I0_D0_PACK+18
.set vgprValuA_X2_I0_D2, vgprValuA_X0_I0_D0_PACK+20
.set vgprValuA_X2_I0_D3, vgprValuA_X0_I0_D0_PACK+22
.set vgprValuA_X3_I0_D0, vgprValuA_X0_I0_D0_PACK+24
.set vgprValuA_X3_I0_D1, vgprValuA_X0_I0_D0_PACK+26
.set vgprValuA_X3_I0_D2, vgprValuA_X0_I0_D0_PACK+28
.set vgprValuA_X3_I0_D3, vgprValuA_X0_I0_D0_PACK+30
.set vgprValuB_X0_I0_BASE, 52
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X2_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X3_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X0_I0_D0_PACK, 60
.set vgprValuB_X0_I0_D0, vgprValuB_X0_I0_D0_PACK+0
.set vgprValuB_X0_I0_D1, vgprValuB_X0_I0_D0_PACK+2
.set vgprValuB_X0_I0_D2, vgprValuB_X0_I0_D0_PACK+4
.set vgprValuB_X0_I0_D3, vgprValuB_X0_I0_D0_PACK+6
.set vgprValuB_X1_I0_D0, vgprValuB_X0_I0_D0_PACK+8
.set vgprValuB_X1_I0_D1, vgprValuB_X0_I0_D0_PACK+10
.set vgprValuB_X1_I0_D2, vgprValuB_X0_I0_D0_PACK+12
.set vgprValuB_X1_I0_D3, vgprValuB_X0_I0_D0_PACK+14
.set vgprValuB_X2_I0_D0, vgprValuB_X0_I0_D0_PACK+16
.set vgprValuB_X2_I0_D1, vgprValuB_X0_I0_D0_PACK+18
.set vgprValuB_X2_I0_D2, vgprValuB_X0_I0_D0_PACK+20
.set vgprValuB_X2_I0_D3, vgprValuB_X0_I0_D0_PACK+22
.set vgprValuB_X3_I0_D0, vgprValuB_X0_I0_D0_PACK+24
.set vgprValuB_X3_I0_D1, vgprValuB_X0_I0_D0_PACK+26
.set vgprValuB_X3_I0_D2, vgprValuB_X0_I0_D0_PACK+28
.set vgprValuB_X3_I0_D3, vgprValuB_X0_I0_D0_PACK+30

/* Tail: local read reset offsets a */

/* Tail: local read reset offsets b */

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_read_b64 v[vgprValuA_X0_I0_D0+0:vgprValuA_X0_I0_D0+0+1], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D1+0:vgprValuA_X0_I0_D1+0+1], v[vgprLocalReadAddrA+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D2+0:vgprValuA_X0_I0_D2+0+1], v[vgprLocalReadAddrA+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuA_X0_I0_D3+0:vgprValuA_X0_I0_D3+0+1], v[vgprLocalReadAddrA+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_read_b64 v[vgprValuB_X0_I0_D0+0:vgprValuB_X0_I0_D0+0+1], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D1+0:vgprValuB_X0_I0_D1+0+1], v[vgprLocalReadAddrB+0] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D2+0:vgprValuB_X0_I0_D2+0+1], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_b64 v[vgprValuB_X0_I0_D3+0:vgprValuB_X0_I0_D3+0+1], v[vgprLocalReadAddrB+0] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
s_mov_b32 s61, 4096                                // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s61, v[vgprLocalReadAddrA+0] // lrA += 4096 ((MT+PAD)*bpeDS)

/* local read inc b */
                                                   // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s61, v[vgprLocalReadAddrB+0] // lrB += 4096 ((MT+PAD)*bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+4], v[vgprValuA_X0_I0_D1+1], v[vgprValuA_X0_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+5], v[vgprValuA_X0_I0_D3+1], v[vgprValuA_X0_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+6], v[vgprValuA_X0_I0_D1+1], v[vgprValuA_X0_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+7], v[vgprValuA_X0_I0_D3+1], v[vgprValuA_X0_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0_D1+0], v[vgprValuB_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0_D3+0], v[vgprValuB_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0_D1+0], v[vgprValuB_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0_D3+0], v[vgprValuB_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0_D1+1], v[vgprValuB_X0_I0_D0+1], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+5], v[vgprValuB_X0_I0_D3+1], v[vgprValuB_X0_I0_D2+1], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0_D1+1], v[vgprValuB_X0_I0_D0+1], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0_D3+1], v[vgprValuB_X0_I0_D2+1], s[sgprPackKForV1] // select K=23 for vector=1
v_and_b32 v92, 63, v[vgprSerial]                   // v92 = v[vgprSerial] % 64
v_lshrrev_b32 v92, 4, v92                          // 92 = 92 / 16
v_lshlrev_b32 v92, 2, v92                          // v92 = v92 * 4
v_cmp_ge_i32 s[62:63], v92, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+2+0+0+0], v[vgprValuA_X0_I0+2+0+0+0], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+0], v[vgprValuA_X0_I0+4+0+0+0], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+6+0+0+0], v[vgprValuA_X0_I0+6+0+0+0], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0+1], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0+1], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+6+0+0+1], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+0], v[vgprValuB_X0_I0+2+0+0+0], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+6+0+0+0], v[vgprValuB_X0_I0+6+0+0+0], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0+1], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], 0, s[62:63] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0+1], 0, s[62:63] // set 0 if K_idx >= sizeL
v_sub_u32 v93, s[sgprLoopCounterL], v92            // get distance between size and k index
v_cmp_lt_i32 s[62:63], v93, 4                      // set partial 0 if distance less than input per thread
s_and_b32 s61, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s61, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_12   // skip mask
s_and_b32 s61, s[sgprLoopCounterL], 3              // get inputs for edge thread
s_sub_u32 s61, 4, s61                              // use shift to fill 0 for outside element
s_lshl_b32 s61, s61, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[94:95], s61, v[vgprValuA_X0_I0+0+0+0+0:vgprValuA_X0_I0+0+0+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v94, s[62:63]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v95, s[62:63]
v_lshlrev_b64 v[94:95], s61, v[vgprValuA_X0_I0+2+0+0+0:vgprValuA_X0_I0+2+0+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+2+0+0+0], v[vgprValuA_X0_I0+2+0+0+0], v94, s[62:63]
v_cndmask_b32 v[vgprValuA_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0+1], v95, s[62:63]
v_lshlrev_b64 v[94:95], s61, v[vgprValuA_X0_I0+4+0+0+0:vgprValuA_X0_I0+4+0+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+0], v[vgprValuA_X0_I0+4+0+0+0], v94, s[62:63]
v_cndmask_b32 v[vgprValuA_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0+1], v95, s[62:63]
v_lshlrev_b64 v[94:95], s61, v[vgprValuA_X0_I0+6+0+0+0:vgprValuA_X0_I0+6+0+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+6+0+0+0], v[vgprValuA_X0_I0+6+0+0+0], v94, s[62:63]
v_cndmask_b32 v[vgprValuA_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+6+0+0+1], v95, s[62:63]
v_lshlrev_b64 v[94:95], s61, v[vgprValuB_X0_I0+0+0+0+0:vgprValuB_X0_I0+0+0+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v94, s[62:63]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v95, s[62:63]
v_lshlrev_b64 v[94:95], s61, v[vgprValuB_X0_I0+2+0+0+0:vgprValuB_X0_I0+2+0+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+0], v[vgprValuB_X0_I0+2+0+0+0], v94, s[62:63]
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0+1], v95, s[62:63]
v_lshlrev_b64 v[94:95], s61, v[vgprValuB_X0_I0+4+0+0+0:vgprValuB_X0_I0+4+0+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], v94, s[62:63]
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], v95, s[62:63]
v_lshlrev_b64 v[94:95], s61, v[vgprValuB_X0_I0+6+0+0+0:vgprValuB_X0_I0+6+0+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+6+0+0+0], v[vgprValuB_X0_I0+6+0+0+0], v94, s[62:63]
v_cndmask_b32 v[vgprValuB_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0+1], v95, s[62:63]
label_TailLoop_SkipZeroOutMask_12:
s_nop 1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
v_mfma_f32_16x16x16_f16 acc[8:11], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[8:11] // left value = acc[8+0:11+0]
v_mfma_f32_16x16x16_f16 acc[12:15], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[12:15] // left value = acc[12+0:15+0]
v_mfma_f32_16x16x16_f16 acc[16:19], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[16:19] // left value = acc[16+0:19+0]
v_mfma_f32_16x16x16_f16 acc[20:23], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[20:23] // left value = acc[20+0:23+0]
v_mfma_f32_16x16x16_f16 acc[24:27], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[24:27] // left value = acc[24+0:27+0]
v_mfma_f32_16x16x16_f16 acc[28:31], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[28:31] // left value = acc[28+0:31+0]
v_mfma_f32_16x16x16_f16 acc[32:35], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[32:35] // left value = acc[32+0:35+0]
v_mfma_f32_16x16x16_f16 acc[36:39], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[36:39] // left value = acc[36+0:39+0]
v_mfma_f32_16x16x16_f16 acc[40:43], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[40:43] // left value = acc[40+0:43+0]
v_mfma_f32_16x16x16_f16 acc[44:47], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[44:47] // left value = acc[44+0:47+0]
v_mfma_f32_16x16x16_f16 acc[48:51], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[48:51] // left value = acc[48+0:51+0]
v_mfma_f32_16x16x16_f16 acc[52:55], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[52:55] // left value = acc[52+0:55+0]
v_mfma_f32_16x16x16_f16 acc[56:59], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+4+0+0:vgprValuA_X0_I0+4+0+0+1], acc[56:59] // left value = acc[56+0:59+0]
v_mfma_f32_16x16x16_f16 acc[60:63], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], v[vgprValuA_X0_I0+6+0+0:vgprValuA_X0_I0+6+0+0+1], acc[60:63] // left value = acc[60+0:63+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x10 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x10 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
s_mov_b32 s61, 256                                 // tailloop lds offset
s_mul_i32 s61, s[sgprOrigLoopCounter], s61         // scale by mul
v_sub_u32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA], s61 // remove lro damage
s_mov_b32 s61, 256                                 // tailloop lds offset
s_mul_i32 s61, s[sgprOrigLoopCounter], s61         // scale by mul
v_sub_u32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB], s61 // remove lro damage
label_SkipTailLoopL:
.set vgprValuA_X0_I0_BASE, UNDEF
.set vgprValuA_X0_I0, UNDEF
.set vgprValuA_X1_I0, UNDEF
.set vgprValuA_X2_I0, UNDEF
.set vgprValuA_X3_I0, UNDEF
.set vgprValuA_X0_I0_D0_PACK, UNDEF
.set vgprValuA_X0_I0_D0, UNDEF
.set vgprValuA_X0_I0_D1, UNDEF
.set vgprValuA_X0_I0_D2, UNDEF
.set vgprValuA_X0_I0_D3, UNDEF
.set vgprValuA_X1_I0_D0, UNDEF
.set vgprValuA_X1_I0_D1, UNDEF
.set vgprValuA_X1_I0_D2, UNDEF
.set vgprValuA_X1_I0_D3, UNDEF
.set vgprValuA_X2_I0_D0, UNDEF
.set vgprValuA_X2_I0_D1, UNDEF
.set vgprValuA_X2_I0_D2, UNDEF
.set vgprValuA_X2_I0_D3, UNDEF
.set vgprValuA_X3_I0_D0, UNDEF
.set vgprValuA_X3_I0_D1, UNDEF
.set vgprValuA_X3_I0_D2, UNDEF
.set vgprValuA_X3_I0_D3, UNDEF
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_X1_I0, UNDEF
.set vgprValuB_X2_I0, UNDEF
.set vgprValuB_X3_I0, UNDEF
.set vgprValuB_X0_I0_D0_PACK, UNDEF
.set vgprValuB_X0_I0_D0, UNDEF
.set vgprValuB_X0_I0_D1, UNDEF
.set vgprValuB_X0_I0_D2, UNDEF
.set vgprValuB_X0_I0_D3, UNDEF
.set vgprValuB_X1_I0_D0, UNDEF
.set vgprValuB_X1_I0_D1, UNDEF
.set vgprValuB_X1_I0_D2, UNDEF
.set vgprValuB_X1_I0_D3, UNDEF
.set vgprValuB_X2_I0_D0, UNDEF
.set vgprValuB_X2_I0_D1, UNDEF
.set vgprValuB_X2_I0_D2, UNDEF
.set vgprValuB_X2_I0_D3, UNDEF
.set vgprValuB_X3_I0_D0, UNDEF
.set vgprValuB_X3_I0_D1, UNDEF
.set vgprValuB_X3_I0_D2, UNDEF
.set vgprValuB_X3_I0_D3, UNDEF

/* Tail: add MISC Vgpr [0...12) to pool */
label_Summation_End_13:
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* shift vector components d0 */
v_mov_b32 v3, s[sgprWorkGroup0]
v_mul_i32_i24 v3, -0x80, v3                        // wg*MT
v_add_co_u32 v3, vcc, s[sgprSizesFree+0], v3       // wgMT = Size - wg*MT
v_mov_b32 v4, 0x80                                 // MT
v_cmp_lt_u32 s[8:9], v3, v4                        // wgMT < MT
v_cndmask_b32 v3, v4, v3, s[8:9]                   // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v5, 6, v[vgprSerial]                 // 5 = Serial / 64
v_and_b32 v5, 1, v5                                // v5 = v5 % 2
v_lshrrev_b32 v6, 6, v3                            // 6 = 3 / 64
v_and_b32 v6, 1, v6                                // v6 = v6 % 2
v_cmp_eq_u32 s[8:9], v6, v5                        // wave_id == block_belong_to_wave?
v_cndmask_b32 v3, v4, v3, s[8:9]                   // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(4)) */
v_lshrrev_b32 v4, 6, v3                            // 4 = 3 / 64
v_lshlrev_b32 v6, 0, v5                            // v6 = v5 * 1
v_sub_u32 v4, v4, v6

/* gbReg: glvw block id */
v_lshrrev_b32 v6, 3, v3                            // 6 = 3 / 8

/* tgbReg: glvw block id */
v_lshrrev_b32 v7, 0, v[vgprSerial]                 // 7 = Serial / 1
v_and_b32 v7, 15, v7                               // v7 = v7 % 16
v_lshlrev_b32 v7, 2, v7                            // v7 = v7 * 4
v_lshrrev_b32 v7, 3, v7                            // 7 = 7 / 8
v_lshlrev_b32 v5, 3, v5                            // v5 = v5 * 8
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
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v6, s8                       // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc12                       // glvw 1 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_read_b32 v7, acc28                       // glvw 1 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v7
v_accvgpr_read_b32 v7, acc44                       // glvw 1 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v7
v_accvgpr_read_b32 v7, acc60                       // glvw 1 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v7
v_accvgpr_read_b32 v7, acc13                       // glvw 1 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_read_b32 v7, acc29                       // glvw 1 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v7
v_accvgpr_read_b32 v7, acc45                       // glvw 1 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v7
v_accvgpr_read_b32 v7, acc61                       // glvw 1 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v7
v_accvgpr_read_b32 v7, acc14                       // glvw 1 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_read_b32 v7, acc30                       // glvw 1 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v7
v_accvgpr_read_b32 v7, acc46                       // glvw 1 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v7
v_accvgpr_read_b32 v7, acc62                       // glvw 1 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v7
v_accvgpr_read_b32 v7, acc15                       // glvw 1 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
v_accvgpr_read_b32 v7, acc31                       // glvw 1 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v7
v_accvgpr_read_b32 v7, acc47                       // glvw 1 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v7
v_accvgpr_read_b32 v7, acc63                       // glvw 1 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v7
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=2 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0_VW0:  /// r2 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v6, s8                       // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc8                        // glvw 2 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc12                       // glvw 2 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc4, v8
v_accvgpr_read_b32 v7, acc24                       // glvw 2 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v8, acc28                       // glvw 2 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v7
v_accvgpr_write_b32 acc20, v8
v_accvgpr_read_b32 v7, acc40                       // glvw 2 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v8, acc44                       // glvw 2 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v7
v_accvgpr_write_b32 acc36, v8
v_accvgpr_read_b32 v7, acc56                       // glvw 2 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v8, acc60                       // glvw 2 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v7
v_accvgpr_write_b32 acc52, v8
v_accvgpr_read_b32 v7, acc9                        // glvw 2 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v8, acc13                       // glvw 2 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_write_b32 acc5, v8
v_accvgpr_read_b32 v7, acc25                       // glvw 2 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v8, acc29                       // glvw 2 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v7
v_accvgpr_write_b32 acc21, v8
v_accvgpr_read_b32 v7, acc41                       // glvw 2 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v8, acc45                       // glvw 2 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v7
v_accvgpr_write_b32 acc37, v8
v_accvgpr_read_b32 v7, acc57                       // glvw 2 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v8, acc61                       // glvw 2 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v7
v_accvgpr_write_b32 acc53, v8
v_accvgpr_read_b32 v7, acc10                       // glvw 2 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v8, acc14                       // glvw 2 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_write_b32 acc6, v8
v_accvgpr_read_b32 v7, acc26                       // glvw 2 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v8, acc30                       // glvw 2 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v7
v_accvgpr_write_b32 acc22, v8
v_accvgpr_read_b32 v7, acc42                       // glvw 2 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v8, acc46                       // glvw 2 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v7
v_accvgpr_write_b32 acc38, v8
v_accvgpr_read_b32 v7, acc58                       // glvw 2 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v8, acc62                       // glvw 2 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v7
v_accvgpr_write_b32 acc54, v8
v_accvgpr_read_b32 v7, acc11                       // glvw 2 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v8, acc15                       // glvw 2 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
v_accvgpr_write_b32 acc7, v8
v_accvgpr_read_b32 v7, acc27                       // glvw 2 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v8, acc31                       // glvw 2 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v7
v_accvgpr_write_b32 acc23, v8
v_accvgpr_read_b32 v7, acc43                       // glvw 2 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v8, acc47                       // glvw 2 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v7
v_accvgpr_write_b32 acc39, v8
v_accvgpr_read_b32 v7, acc59                       // glvw 2 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v8, acc63                       // glvw 2 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v7
v_accvgpr_write_b32 acc55, v8
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=3 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0_VW0:  /// r3 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v6, s8                       // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc4                        // glvw 3 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc8                        // glvw 3 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc12                       // glvw 3 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc4, v8
v_accvgpr_write_b32 acc8, v9
v_accvgpr_read_b32 v7, acc20                       // glvw 3 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v8, acc24                       // glvw 3 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v9, acc28                       // glvw 3 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v7
v_accvgpr_write_b32 acc20, v8
v_accvgpr_write_b32 acc24, v9
v_accvgpr_read_b32 v7, acc36                       // glvw 3 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v8, acc40                       // glvw 3 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v9, acc44                       // glvw 3 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v7
v_accvgpr_write_b32 acc36, v8
v_accvgpr_write_b32 acc40, v9
v_accvgpr_read_b32 v7, acc52                       // glvw 3 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v8, acc56                       // glvw 3 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v9, acc60                       // glvw 3 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v7
v_accvgpr_write_b32 acc52, v8
v_accvgpr_write_b32 acc56, v9
v_accvgpr_read_b32 v7, acc5                        // glvw 3 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v8, acc9                        // glvw 3 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v9, acc13                       // glvw 3 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_write_b32 acc5, v8
v_accvgpr_write_b32 acc9, v9
v_accvgpr_read_b32 v7, acc21                       // glvw 3 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v8, acc25                       // glvw 3 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v9, acc29                       // glvw 3 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v7
v_accvgpr_write_b32 acc21, v8
v_accvgpr_write_b32 acc25, v9
v_accvgpr_read_b32 v7, acc37                       // glvw 3 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v8, acc41                       // glvw 3 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v9, acc45                       // glvw 3 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v7
v_accvgpr_write_b32 acc37, v8
v_accvgpr_write_b32 acc41, v9
v_accvgpr_read_b32 v7, acc53                       // glvw 3 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v8, acc57                       // glvw 3 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v9, acc61                       // glvw 3 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v7
v_accvgpr_write_b32 acc53, v8
v_accvgpr_write_b32 acc57, v9
v_accvgpr_read_b32 v7, acc6                        // glvw 3 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v8, acc10                       // glvw 3 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v9, acc14                       // glvw 3 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_write_b32 acc6, v8
v_accvgpr_write_b32 acc10, v9
v_accvgpr_read_b32 v7, acc22                       // glvw 3 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v8, acc26                       // glvw 3 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v9, acc30                       // glvw 3 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v7
v_accvgpr_write_b32 acc22, v8
v_accvgpr_write_b32 acc26, v9
v_accvgpr_read_b32 v7, acc38                       // glvw 3 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v8, acc42                       // glvw 3 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v9, acc46                       // glvw 3 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v7
v_accvgpr_write_b32 acc38, v8
v_accvgpr_write_b32 acc42, v9
v_accvgpr_read_b32 v7, acc54                       // glvw 3 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v8, acc58                       // glvw 3 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v9, acc62                       // glvw 3 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v7
v_accvgpr_write_b32 acc54, v8
v_accvgpr_write_b32 acc58, v9
v_accvgpr_read_b32 v7, acc7                        // glvw 3 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v8, acc11                       // glvw 3 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v9, acc15                       // glvw 3 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
v_accvgpr_write_b32 acc7, v8
v_accvgpr_write_b32 acc11, v9
v_accvgpr_read_b32 v7, acc23                       // glvw 3 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v8, acc27                       // glvw 3 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v9, acc31                       // glvw 3 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v7
v_accvgpr_write_b32 acc23, v8
v_accvgpr_write_b32 acc27, v9
v_accvgpr_read_b32 v7, acc39                       // glvw 3 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v8, acc43                       // glvw 3 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v9, acc47                       // glvw 3 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v7
v_accvgpr_write_b32 acc39, v8
v_accvgpr_write_b32 acc43, v9
v_accvgpr_read_b32 v7, acc55                       // glvw 3 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v8, acc59                       // glvw 3 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v9, acc63                       // glvw 3 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v7
v_accvgpr_write_b32 acc55, v8
v_accvgpr_write_b32 acc59, v9
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=4 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM0_VW0:  /// r4 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v6, s8                       // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 4 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc4                        // glvw 4 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc8                        // glvw 4 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v10, acc12                      // glvw 4 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc4, v8
v_accvgpr_write_b32 acc8, v9
v_accvgpr_write_b32 acc12, v10
v_accvgpr_read_b32 v7, acc16                       // glvw 4 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v8, acc20                       // glvw 4 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v9, acc24                       // glvw 4 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v10, acc28                      // glvw 4 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v7
v_accvgpr_write_b32 acc20, v8
v_accvgpr_write_b32 acc24, v9
v_accvgpr_write_b32 acc28, v10
v_accvgpr_read_b32 v7, acc32                       // glvw 4 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v8, acc36                       // glvw 4 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v9, acc40                       // glvw 4 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v10, acc44                      // glvw 4 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v7
v_accvgpr_write_b32 acc36, v8
v_accvgpr_write_b32 acc40, v9
v_accvgpr_write_b32 acc44, v10
v_accvgpr_read_b32 v7, acc48                       // glvw 4 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v8, acc52                       // glvw 4 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v9, acc56                       // glvw 4 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v10, acc60                      // glvw 4 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v7
v_accvgpr_write_b32 acc52, v8
v_accvgpr_write_b32 acc56, v9
v_accvgpr_write_b32 acc60, v10
v_accvgpr_read_b32 v7, acc1                        // glvw 4 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v8, acc5                        // glvw 4 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v9, acc9                        // glvw 4 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v10, acc13                      // glvw 4 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_write_b32 acc5, v8
v_accvgpr_write_b32 acc9, v9
v_accvgpr_write_b32 acc13, v10
v_accvgpr_read_b32 v7, acc17                       // glvw 4 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v8, acc21                       // glvw 4 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v9, acc25                       // glvw 4 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v10, acc29                      // glvw 4 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v7
v_accvgpr_write_b32 acc21, v8
v_accvgpr_write_b32 acc25, v9
v_accvgpr_write_b32 acc29, v10
v_accvgpr_read_b32 v7, acc33                       // glvw 4 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v8, acc37                       // glvw 4 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v9, acc41                       // glvw 4 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v10, acc45                      // glvw 4 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v7
v_accvgpr_write_b32 acc37, v8
v_accvgpr_write_b32 acc41, v9
v_accvgpr_write_b32 acc45, v10
v_accvgpr_read_b32 v7, acc49                       // glvw 4 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v8, acc53                       // glvw 4 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v9, acc57                       // glvw 4 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v10, acc61                      // glvw 4 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v7
v_accvgpr_write_b32 acc53, v8
v_accvgpr_write_b32 acc57, v9
v_accvgpr_write_b32 acc61, v10
v_accvgpr_read_b32 v7, acc2                        // glvw 4 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v8, acc6                        // glvw 4 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v9, acc10                       // glvw 4 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v10, acc14                      // glvw 4 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_write_b32 acc6, v8
v_accvgpr_write_b32 acc10, v9
v_accvgpr_write_b32 acc14, v10
v_accvgpr_read_b32 v7, acc18                       // glvw 4 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v8, acc22                       // glvw 4 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v9, acc26                       // glvw 4 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v10, acc30                      // glvw 4 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v7
v_accvgpr_write_b32 acc22, v8
v_accvgpr_write_b32 acc26, v9
v_accvgpr_write_b32 acc30, v10
v_accvgpr_read_b32 v7, acc34                       // glvw 4 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v8, acc38                       // glvw 4 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v9, acc42                       // glvw 4 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v10, acc46                      // glvw 4 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v7
v_accvgpr_write_b32 acc38, v8
v_accvgpr_write_b32 acc42, v9
v_accvgpr_write_b32 acc46, v10
v_accvgpr_read_b32 v7, acc50                       // glvw 4 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v8, acc54                       // glvw 4 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v9, acc58                       // glvw 4 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v10, acc62                      // glvw 4 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v7
v_accvgpr_write_b32 acc54, v8
v_accvgpr_write_b32 acc58, v9
v_accvgpr_write_b32 acc62, v10
v_accvgpr_read_b32 v7, acc3                        // glvw 4 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v8, acc7                        // glvw 4 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v9, acc11                       // glvw 4 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v10, acc15                      // glvw 4 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
v_accvgpr_write_b32 acc7, v8
v_accvgpr_write_b32 acc11, v9
v_accvgpr_write_b32 acc15, v10
v_accvgpr_read_b32 v7, acc19                       // glvw 4 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v8, acc23                       // glvw 4 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v9, acc27                       // glvw 4 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v10, acc31                      // glvw 4 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v7
v_accvgpr_write_b32 acc23, v8
v_accvgpr_write_b32 acc27, v9
v_accvgpr_write_b32 acc31, v10
v_accvgpr_read_b32 v7, acc35                       // glvw 4 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v8, acc39                       // glvw 4 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v9, acc43                       // glvw 4 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v10, acc47                      // glvw 4 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v7
v_accvgpr_write_b32 acc39, v8
v_accvgpr_write_b32 acc43, v9
v_accvgpr_write_b32 acc47, v10
v_accvgpr_read_b32 v7, acc51                       // glvw 4 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v8, acc55                       // glvw 4 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v9, acc59                       // glvw 4 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v10, acc63                      // glvw 4 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v7
v_accvgpr_write_b32 acc55, v8
v_accvgpr_write_b32 acc59, v9
v_accvgpr_write_b32 acc63, v10
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=5 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM0_VW0:  /// r5 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v6, s8                       // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc12                       // glvw 5 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc0                        // glvw 5 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc4                        // glvw 5 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v10, acc8                       // glvw 5 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc4, v8
v_accvgpr_write_b32 acc8, v9
v_accvgpr_write_b32 acc12, v10
v_accvgpr_read_b32 v7, acc28                       // glvw 5 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v8, acc16                       // glvw 5 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v9, acc20                       // glvw 5 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v10, acc24                      // glvw 5 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v7
v_accvgpr_write_b32 acc20, v8
v_accvgpr_write_b32 acc24, v9
v_accvgpr_write_b32 acc28, v10
v_accvgpr_read_b32 v7, acc44                       // glvw 5 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v8, acc32                       // glvw 5 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v9, acc36                       // glvw 5 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v10, acc40                      // glvw 5 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v7
v_accvgpr_write_b32 acc36, v8
v_accvgpr_write_b32 acc40, v9
v_accvgpr_write_b32 acc44, v10
v_accvgpr_read_b32 v7, acc60                       // glvw 5 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v8, acc48                       // glvw 5 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v9, acc52                       // glvw 5 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v10, acc56                      // glvw 5 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v7
v_accvgpr_write_b32 acc52, v8
v_accvgpr_write_b32 acc56, v9
v_accvgpr_write_b32 acc60, v10
v_accvgpr_read_b32 v7, acc13                       // glvw 5 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v8, acc1                        // glvw 5 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v9, acc5                        // glvw 5 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v10, acc9                       // glvw 5 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_write_b32 acc5, v8
v_accvgpr_write_b32 acc9, v9
v_accvgpr_write_b32 acc13, v10
v_accvgpr_read_b32 v7, acc29                       // glvw 5 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v8, acc17                       // glvw 5 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v9, acc21                       // glvw 5 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v10, acc25                      // glvw 5 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v7
v_accvgpr_write_b32 acc21, v8
v_accvgpr_write_b32 acc25, v9
v_accvgpr_write_b32 acc29, v10
v_accvgpr_read_b32 v7, acc45                       // glvw 5 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v8, acc33                       // glvw 5 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v9, acc37                       // glvw 5 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v10, acc41                      // glvw 5 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v7
v_accvgpr_write_b32 acc37, v8
v_accvgpr_write_b32 acc41, v9
v_accvgpr_write_b32 acc45, v10
v_accvgpr_read_b32 v7, acc61                       // glvw 5 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v8, acc49                       // glvw 5 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v9, acc53                       // glvw 5 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v10, acc57                      // glvw 5 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v7
v_accvgpr_write_b32 acc53, v8
v_accvgpr_write_b32 acc57, v9
v_accvgpr_write_b32 acc61, v10
v_accvgpr_read_b32 v7, acc14                       // glvw 5 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v8, acc2                        // glvw 5 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v9, acc6                        // glvw 5 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v10, acc10                      // glvw 5 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_write_b32 acc6, v8
v_accvgpr_write_b32 acc10, v9
v_accvgpr_write_b32 acc14, v10
v_accvgpr_read_b32 v7, acc30                       // glvw 5 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v8, acc18                       // glvw 5 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v9, acc22                       // glvw 5 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v10, acc26                      // glvw 5 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v7
v_accvgpr_write_b32 acc22, v8
v_accvgpr_write_b32 acc26, v9
v_accvgpr_write_b32 acc30, v10
v_accvgpr_read_b32 v7, acc46                       // glvw 5 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v8, acc34                       // glvw 5 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v9, acc38                       // glvw 5 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v10, acc42                      // glvw 5 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v7
v_accvgpr_write_b32 acc38, v8
v_accvgpr_write_b32 acc42, v9
v_accvgpr_write_b32 acc46, v10
v_accvgpr_read_b32 v7, acc62                       // glvw 5 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v8, acc50                       // glvw 5 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v9, acc54                       // glvw 5 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v10, acc58                      // glvw 5 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v7
v_accvgpr_write_b32 acc54, v8
v_accvgpr_write_b32 acc58, v9
v_accvgpr_write_b32 acc62, v10
v_accvgpr_read_b32 v7, acc15                       // glvw 5 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v8, acc3                        // glvw 5 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v9, acc7                        // glvw 5 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v10, acc11                      // glvw 5 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
v_accvgpr_write_b32 acc7, v8
v_accvgpr_write_b32 acc11, v9
v_accvgpr_write_b32 acc15, v10
v_accvgpr_read_b32 v7, acc31                       // glvw 5 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v8, acc19                       // glvw 5 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v9, acc23                       // glvw 5 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v10, acc27                      // glvw 5 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v7
v_accvgpr_write_b32 acc23, v8
v_accvgpr_write_b32 acc27, v9
v_accvgpr_write_b32 acc31, v10
v_accvgpr_read_b32 v7, acc47                       // glvw 5 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v8, acc35                       // glvw 5 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v9, acc39                       // glvw 5 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v10, acc43                      // glvw 5 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v7
v_accvgpr_write_b32 acc39, v8
v_accvgpr_write_b32 acc43, v9
v_accvgpr_write_b32 acc47, v10
v_accvgpr_read_b32 v7, acc63                       // glvw 5 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v8, acc51                       // glvw 5 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v9, acc55                       // glvw 5 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v10, acc59                      // glvw 5 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v8, v0, v8 offset:4                // permute edge values
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v7
v_accvgpr_write_b32 acc55, v8
v_accvgpr_write_b32 acc59, v9
v_accvgpr_write_b32 acc63, v10
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=6 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM0_VW0:  /// r6 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v6, s8                       // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc8                        // glvw 6 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc12                       // glvw 6 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc0                        // glvw 6 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v10, acc4                       // glvw 6 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc4, v8
v_accvgpr_write_b32 acc8, v9
v_accvgpr_write_b32 acc12, v10
v_accvgpr_read_b32 v7, acc24                       // glvw 6 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v8, acc28                       // glvw 6 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v9, acc16                       // glvw 6 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v10, acc20                      // glvw 6 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v7
v_accvgpr_write_b32 acc20, v8
v_accvgpr_write_b32 acc24, v9
v_accvgpr_write_b32 acc28, v10
v_accvgpr_read_b32 v7, acc40                       // glvw 6 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v8, acc44                       // glvw 6 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v9, acc32                       // glvw 6 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v10, acc36                      // glvw 6 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v7
v_accvgpr_write_b32 acc36, v8
v_accvgpr_write_b32 acc40, v9
v_accvgpr_write_b32 acc44, v10
v_accvgpr_read_b32 v7, acc56                       // glvw 6 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v8, acc60                       // glvw 6 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v9, acc48                       // glvw 6 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v10, acc52                      // glvw 6 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v7
v_accvgpr_write_b32 acc52, v8
v_accvgpr_write_b32 acc56, v9
v_accvgpr_write_b32 acc60, v10
v_accvgpr_read_b32 v7, acc9                        // glvw 6 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v8, acc13                       // glvw 6 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v9, acc1                        // glvw 6 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v10, acc5                       // glvw 6 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_write_b32 acc5, v8
v_accvgpr_write_b32 acc9, v9
v_accvgpr_write_b32 acc13, v10
v_accvgpr_read_b32 v7, acc25                       // glvw 6 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v8, acc29                       // glvw 6 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v9, acc17                       // glvw 6 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v10, acc21                      // glvw 6 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v7
v_accvgpr_write_b32 acc21, v8
v_accvgpr_write_b32 acc25, v9
v_accvgpr_write_b32 acc29, v10
v_accvgpr_read_b32 v7, acc41                       // glvw 6 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v8, acc45                       // glvw 6 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v9, acc33                       // glvw 6 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v10, acc37                      // glvw 6 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v7
v_accvgpr_write_b32 acc37, v8
v_accvgpr_write_b32 acc41, v9
v_accvgpr_write_b32 acc45, v10
v_accvgpr_read_b32 v7, acc57                       // glvw 6 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v8, acc61                       // glvw 6 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v9, acc49                       // glvw 6 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v10, acc53                      // glvw 6 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v7
v_accvgpr_write_b32 acc53, v8
v_accvgpr_write_b32 acc57, v9
v_accvgpr_write_b32 acc61, v10
v_accvgpr_read_b32 v7, acc10                       // glvw 6 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v8, acc14                       // glvw 6 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v9, acc2                        // glvw 6 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v10, acc6                       // glvw 6 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_write_b32 acc6, v8
v_accvgpr_write_b32 acc10, v9
v_accvgpr_write_b32 acc14, v10
v_accvgpr_read_b32 v7, acc26                       // glvw 6 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v8, acc30                       // glvw 6 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v9, acc18                       // glvw 6 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v10, acc22                      // glvw 6 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v7
v_accvgpr_write_b32 acc22, v8
v_accvgpr_write_b32 acc26, v9
v_accvgpr_write_b32 acc30, v10
v_accvgpr_read_b32 v7, acc42                       // glvw 6 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v8, acc46                       // glvw 6 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v9, acc34                       // glvw 6 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v10, acc38                      // glvw 6 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v7
v_accvgpr_write_b32 acc38, v8
v_accvgpr_write_b32 acc42, v9
v_accvgpr_write_b32 acc46, v10
v_accvgpr_read_b32 v7, acc58                       // glvw 6 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v8, acc62                       // glvw 6 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v9, acc50                       // glvw 6 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v10, acc54                      // glvw 6 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v7
v_accvgpr_write_b32 acc54, v8
v_accvgpr_write_b32 acc58, v9
v_accvgpr_write_b32 acc62, v10
v_accvgpr_read_b32 v7, acc11                       // glvw 6 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v8, acc15                       // glvw 6 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v9, acc3                        // glvw 6 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v10, acc7                       // glvw 6 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
v_accvgpr_write_b32 acc7, v8
v_accvgpr_write_b32 acc11, v9
v_accvgpr_write_b32 acc15, v10
v_accvgpr_read_b32 v7, acc27                       // glvw 6 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v8, acc31                       // glvw 6 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v9, acc19                       // glvw 6 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v10, acc23                      // glvw 6 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v7
v_accvgpr_write_b32 acc23, v8
v_accvgpr_write_b32 acc27, v9
v_accvgpr_write_b32 acc31, v10
v_accvgpr_read_b32 v7, acc43                       // glvw 6 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v8, acc47                       // glvw 6 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v9, acc35                       // glvw 6 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v10, acc39                      // glvw 6 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v7
v_accvgpr_write_b32 acc39, v8
v_accvgpr_write_b32 acc43, v9
v_accvgpr_write_b32 acc47, v10
v_accvgpr_read_b32 v7, acc59                       // glvw 6 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v8, acc63                       // glvw 6 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v9, acc51                       // glvw 6 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v10, acc55                      // glvw 6 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v9, v0, v9 offset:4                // permute edge values
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v7
v_accvgpr_write_b32 acc55, v8
v_accvgpr_write_b32 acc59, v9
v_accvgpr_write_b32 acc63, v10
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0


/******************************************/
/* shift d0 r=7 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM0_VW0:  /// r7 mb0 vw0
s_mov_b32 s8, 0
v_cmpx_eq_u32 s[8:9], v6, s8                       // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc4                        // glvw 7 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v8, acc8                        // glvw 7 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v9, acc12                       // glvw 7 mb 0 tt1 0 r 0
v_accvgpr_read_b32 v10, acc0                       // glvw 7 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc4, v8
v_accvgpr_write_b32 acc8, v9
v_accvgpr_write_b32 acc12, v10
v_accvgpr_read_b32 v7, acc20                       // glvw 7 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v8, acc24                       // glvw 7 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v9, acc28                       // glvw 7 mb 0 tt1 1 r 0
v_accvgpr_read_b32 v10, acc16                      // glvw 7 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc16, v7
v_accvgpr_write_b32 acc20, v8
v_accvgpr_write_b32 acc24, v9
v_accvgpr_write_b32 acc28, v10
v_accvgpr_read_b32 v7, acc36                       // glvw 7 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v8, acc40                       // glvw 7 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v9, acc44                       // glvw 7 mb 0 tt1 2 r 0
v_accvgpr_read_b32 v10, acc32                      // glvw 7 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc32, v7
v_accvgpr_write_b32 acc36, v8
v_accvgpr_write_b32 acc40, v9
v_accvgpr_write_b32 acc44, v10
v_accvgpr_read_b32 v7, acc52                       // glvw 7 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v8, acc56                       // glvw 7 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v9, acc60                       // glvw 7 mb 0 tt1 3 r 0
v_accvgpr_read_b32 v10, acc48                      // glvw 7 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc48, v7
v_accvgpr_write_b32 acc52, v8
v_accvgpr_write_b32 acc56, v9
v_accvgpr_write_b32 acc60, v10
v_accvgpr_read_b32 v7, acc5                        // glvw 7 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v8, acc9                        // glvw 7 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v9, acc13                       // glvw 7 mb 0 tt1 4 r 0
v_accvgpr_read_b32 v10, acc1                       // glvw 7 mb 0 tt1 4 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_write_b32 acc5, v8
v_accvgpr_write_b32 acc9, v9
v_accvgpr_write_b32 acc13, v10
v_accvgpr_read_b32 v7, acc21                       // glvw 7 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v8, acc25                       // glvw 7 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v9, acc29                       // glvw 7 mb 0 tt1 5 r 0
v_accvgpr_read_b32 v10, acc17                      // glvw 7 mb 0 tt1 5 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc17, v7
v_accvgpr_write_b32 acc21, v8
v_accvgpr_write_b32 acc25, v9
v_accvgpr_write_b32 acc29, v10
v_accvgpr_read_b32 v7, acc37                       // glvw 7 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v8, acc41                       // glvw 7 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v9, acc45                       // glvw 7 mb 0 tt1 6 r 0
v_accvgpr_read_b32 v10, acc33                      // glvw 7 mb 0 tt1 6 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc33, v7
v_accvgpr_write_b32 acc37, v8
v_accvgpr_write_b32 acc41, v9
v_accvgpr_write_b32 acc45, v10
v_accvgpr_read_b32 v7, acc53                       // glvw 7 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v8, acc57                       // glvw 7 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v9, acc61                       // glvw 7 mb 0 tt1 7 r 0
v_accvgpr_read_b32 v10, acc49                      // glvw 7 mb 0 tt1 7 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc49, v7
v_accvgpr_write_b32 acc53, v8
v_accvgpr_write_b32 acc57, v9
v_accvgpr_write_b32 acc61, v10
v_accvgpr_read_b32 v7, acc6                        // glvw 7 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v8, acc10                       // glvw 7 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v9, acc14                       // glvw 7 mb 0 tt1 8 r 0
v_accvgpr_read_b32 v10, acc2                       // glvw 7 mb 0 tt1 8 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_write_b32 acc6, v8
v_accvgpr_write_b32 acc10, v9
v_accvgpr_write_b32 acc14, v10
v_accvgpr_read_b32 v7, acc22                       // glvw 7 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v8, acc26                       // glvw 7 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v9, acc30                       // glvw 7 mb 0 tt1 9 r 0
v_accvgpr_read_b32 v10, acc18                      // glvw 7 mb 0 tt1 9 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc18, v7
v_accvgpr_write_b32 acc22, v8
v_accvgpr_write_b32 acc26, v9
v_accvgpr_write_b32 acc30, v10
v_accvgpr_read_b32 v7, acc38                       // glvw 7 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v8, acc42                       // glvw 7 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v9, acc46                       // glvw 7 mb 0 tt1 10 r 0
v_accvgpr_read_b32 v10, acc34                      // glvw 7 mb 0 tt1 10 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc34, v7
v_accvgpr_write_b32 acc38, v8
v_accvgpr_write_b32 acc42, v9
v_accvgpr_write_b32 acc46, v10
v_accvgpr_read_b32 v7, acc54                       // glvw 7 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v8, acc58                       // glvw 7 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v9, acc62                       // glvw 7 mb 0 tt1 11 r 0
v_accvgpr_read_b32 v10, acc50                      // glvw 7 mb 0 tt1 11 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc50, v7
v_accvgpr_write_b32 acc54, v8
v_accvgpr_write_b32 acc58, v9
v_accvgpr_write_b32 acc62, v10
v_accvgpr_read_b32 v7, acc7                        // glvw 7 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v8, acc11                       // glvw 7 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v9, acc15                       // glvw 7 mb 0 tt1 12 r 0
v_accvgpr_read_b32 v10, acc3                       // glvw 7 mb 0 tt1 12 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
v_accvgpr_write_b32 acc7, v8
v_accvgpr_write_b32 acc11, v9
v_accvgpr_write_b32 acc15, v10
v_accvgpr_read_b32 v7, acc23                       // glvw 7 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v8, acc27                       // glvw 7 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v9, acc31                       // glvw 7 mb 0 tt1 13 r 0
v_accvgpr_read_b32 v10, acc19                      // glvw 7 mb 0 tt1 13 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc19, v7
v_accvgpr_write_b32 acc23, v8
v_accvgpr_write_b32 acc27, v9
v_accvgpr_write_b32 acc31, v10
v_accvgpr_read_b32 v7, acc39                       // glvw 7 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v8, acc43                       // glvw 7 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v9, acc47                       // glvw 7 mb 0 tt1 14 r 0
v_accvgpr_read_b32 v10, acc35                      // glvw 7 mb 0 tt1 14 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc35, v7
v_accvgpr_write_b32 acc39, v8
v_accvgpr_write_b32 acc43, v9
v_accvgpr_write_b32 acc47, v10
v_accvgpr_read_b32 v7, acc55                       // glvw 7 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v8, acc59                       // glvw 7 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v9, acc63                       // glvw 7 mb 0 tt1 15 r 0
v_accvgpr_read_b32 v10, acc51                      // glvw 7 mb 0 tt1 15 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v10, v0, v10 offset:4              // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc51, v7
v_accvgpr_write_b32 acc55, v8
v_accvgpr_write_b32 acc59, v9
v_accvgpr_write_b32 acc63, v10
s_mov_b64 s[8:9], 0xFFFFFFFFFFFFFFFF               // to restore all threads active
s_or_saveexec_b64 vcc, s[8:9]                      // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0

label_ShiftVectorComponents0_GLVW0:  /// end shift0

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
s_cmp_eq_u64 s[sgprAddressFlags:sgprAddressFlags+1], 0x0 // Check for synchronizer
s_cbranch_scc0 label_GSU                           // Branch to stream-k store code
s_cmp_eq_u32 s[sgprskTiles], 1                     // split == 1 ?
s_cbranch_scc1 label_GSU                           // branch if split == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:

/* Edge/NonEdge store path check (M): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s62, 127, s[sgprSizeI]                   // s62 = s[sgprSizeI] % 128
s_add_u32 s63, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s63                // wg0 >= nwg0-1 ?
s_cselect_b32 s62, s62, 0                          // set rem
s_cmpk_gt_u32 s62, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s62, 127, s[sgprSizeJ]                   // s62 = s[sgprSizeJ] % 128
s_add_u32 s63, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s63                // wg1 >= nwg1-1
s_cselect_b32 s62, s62, 0                          // set rem
s_cmpk_gt_u32 s62, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MB_Then         // jump if edges required
label_GW_B0_FD0_VW4_MB_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=42 */
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
v_add_lshl_u32 v11, v3, v0, 2                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+20], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+21], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+22], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+23], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+24], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+25], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+26], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+27], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+28], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+29], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+30], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+31], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+36], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+37], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+38], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+39], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+40], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+41], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+42], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+43], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+44], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+45], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+46], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+47], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+48], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+49], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+50], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+51], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+52], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+53], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+54], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+55], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+56], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+57], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+58], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+59], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+60], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+61], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+62], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+63], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+64], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+65], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+66], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+67], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+68], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+69], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+70], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+71], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+72], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+73], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+74], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+75], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+76], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+77], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+78], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+79], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[16:19], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[20:23], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[24:27], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[28:31], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[32:35], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[36:39], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[40:43], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[44:47], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[48:51], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[52:55], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[56:59], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[60:63], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[64:67], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[68:71], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[72:75], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s8, s[sgprStrideD1J], 2                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx4 v[76:79], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_MB_NonEdgeEnd:
label_GW_B0_FD0_VW4_MB_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=34 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v11, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v76, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v76, v6, v76, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v77, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v6, v77, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v78, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v78, v6, v78, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v79, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v79, v6, v79, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v80, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v80, v6, v80, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v81, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v6, v81, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v82, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v6, v82, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v83, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v6, v83, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v84, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v6, v84, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v85, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v85, v6, v85, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v86, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v6, v86, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v87, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v6, v87, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v88, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v6, v88, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v89, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v89, v6, v89, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v90, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v6, v90, s[66:67]               // LDD clip if OOB. offset
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
v_accvgpr_read_b32 v[vgprValuC+72], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+73], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+74], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+75], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx4 v[12:15], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[16:19], v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[20:23], v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[24:27], v78, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[28:31], v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[32:35], v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[36:39], v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[40:43], v82, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[44:47], v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[48:51], v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[52:55], v85, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[56:59], v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[60:63], v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[64:67], v88, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[68:71], v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx4 v[72:75], v90, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_MB_Else:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=88 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,8,0:vw1); (0,0,8,1:vw1); (0,0,8,2:vw1); (0,0,8,3:vw1); (0,0,9,0:vw1); (0,0,9,1:vw1); (0,0,9,2:vw1); (0,0,9,3:vw1); (0,0,10,0:vw1); (0,0,10,1:vw1); (0,0,10,2:vw1); (0,0,10,3:vw1); (0,0,11,0:vw1); (0,0,11,1:vw1); (0,0,11,2:vw1); (0,0,11,3:vw1); (0,0,12,0:vw1); (0,0,12,1:vw1); (0,0,12,2:vw1); (0,0,12,3:vw1); (0,0,13,0:vw1); (0,0,13,1:vw1); (0,0,13,2:vw1); (0,0,13,3:vw1); (0,0,14,0:vw1); (0,0,14,1:vw1); (0,0,14,2:vw1); (0,0,14,3:vw1); (0,0,15,0:vw1); (0,0,15,1:vw1); (0,0,15,2:vw1); (0,0,15,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v75, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v6, v75, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v76, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v76, v6, v76, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v77, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v6, v77, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v78, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v78, v6, v78, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v79, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v79, v6, v79, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v80, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v80, v6, v80, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v81, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v6, v81, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v82, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v6, v82, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v83, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v6, v83, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v84, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v6, v84, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v85, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v85, v6, v85, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v86, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v6, v86, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v87, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v6, v87, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v88, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v6, v88, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v89, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v89, v6, v89, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v90, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v6, v90, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v91, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v91, v6, v91, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v92, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v6, v92, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v93, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v6, v93, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v94, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v94, v6, v94, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v95, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v95, v6, v95, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v96, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v6, v96, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v97, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v97, v6, v97, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v98, v3, v4, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v6, v98, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v99, v3, v0, 2                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v6, v99, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v100, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v6, v100, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v101, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v6, v101, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v102, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v6, v102, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v103, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v103, v6, v103, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v104, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v6, v104, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v105, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v6, v105, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v106, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v6, v106, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v107, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v6, v107, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v108, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v6, v108, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v109, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v110, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v111, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v112, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v6, v112, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v113, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v6, v113, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v114, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v6, v114, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v115, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v6, v115, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v116, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v117, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v118, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v6, v118, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v119, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v6, v119, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v120, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v6, v120, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v121, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v6, v121, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v122, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v6, v122, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v123, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v6, v123, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v125, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v6, v125, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v126, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v6, v126, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v127, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v6, v127, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v128, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v6, v128, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v129, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v6, v129, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v130, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v6, v130, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v131, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v6, v131, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v132, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v6, v132, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v133, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v6, v133, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v134, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v6, v134, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v135, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v6, v135, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v136, v3, v0, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v6, v136, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v137, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v6, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v138, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v6, v138, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v139, v3, v4, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v6, v139, s[66:67]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+14], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+15], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+16], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+17], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+18], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+19], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+20], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+21], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+22], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+23], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+24], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+25], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+26], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+27], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+28], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+29], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+30], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+31], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+32], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+33], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+34], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+35], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+36], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+37], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+38], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+39], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+40], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+41], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+42], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+43], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+44], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+45], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+46], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+47], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+48], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+49], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+50], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+51], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+52], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+53], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+54], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+55], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+56], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+57], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+58], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+59], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+60], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+61], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+62], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+63], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+64], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+65], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+66], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+67], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+68], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+69], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+70], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+71], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+72], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+73], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+74], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 8, 0), (0, 0, 8, 1), (0, 0, 8, 2), (0, 0, 8, 3), (0, 0, 9, 0), (0, 0, 9, 1), (0, 0, 9, 2), (0, 0, 9, 3), (0, 0, 10, 0), (0, 0, 10, 1), (0, 0, 10, 2), (0, 0, 10, 3), (0, 0, 11, 0), (0, 0, 11, 1), (0, 0, 11, 2), (0, 0, 11, 3), (0, 0, 12, 0), (0, 0, 12, 1), (0, 0, 12, 2), (0, 0, 12, 3), (0, 0, 13, 0), (0, 0, 13, 1), (0, 0, 13, 2), (0, 0, 13, 3), (0, 0, 14, 0), (0, 0, 14, 1), (0, 0, 14, 2), (0, 0, 14, 3), (0, 0, 15, 0), (0, 0, 15, 1), (0, 0, 15, 2), (0, 0, 15, 3)] */

/* apply mask, calc new C and issue writes */
buffer_store_dword v11, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v12, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v13, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v14, v78, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v15, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v16, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v17, v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v18, v82, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v19, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v20, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v21, v85, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v22, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v23, v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v24, v88, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v25, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v26, v90, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v27, v91, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v28, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v29, v93, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v30, v94, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v31, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v32, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v33, v97, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v34, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v35, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v36, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v37, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v38, v102, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v39, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v40, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v41, v105, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v42, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v43, v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v44, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v45, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v46, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v47, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v48, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v49, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v50, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v51, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v52, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v53, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v54, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v55, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v56, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v57, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v58, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v59, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v60, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v61, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v62, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v63, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v64, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v65, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v66, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v67, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v68, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v69, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v70, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v71, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v72, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v73, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v74, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s64, label_KernelEnd, 4                  // target branch offset
s_add_u32 s62, s62, s64                            // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_setpc_b64 s[62:63]                               // branch to label_KernelEnd
label_GSU:
s_cmp_eq_u32 s[sgprStreamKLocalStart], 0           // does wg start tile?
s_cbranch_scc1 label_NoBranch_14                   // Only branch on scc0
s_getpc_b64 s[66:67]                               // addr of next instr
s_add_i32 s68, label_SK_Partials_1, 4              // target branch offset
s_add_u32 s66, s66, s68                            // add target branch offset
s_addc_u32 s67, s67, 0                             // add high and carry
s_setpc_b64 s[66:67]                               // branch to label_SK_Partials_1
label_NoBranch_14:
s_cmp_eq_u32 s[sgprStreamKLocalEnd], s[sgprItersPerTile] // does wg finish tile?
s_cbranch_scc1 label_SK_Store                      // Branch if started and finished tile, go to regular store code
s_add_u32 s8, s[sgprPersistentWorkGroupIndex], 1   // input partial tile index
s_mul_hi_u32 s62, s[sgprPersistentIterationEnd], s[sgprMagicNumberItersPerTile] // s_magic mul, div alg 2
s_lshr_b32 s63, s[sgprMagicShiftItersPerTile], 31  // tmpS = extract abit
s_mul_i32 s61, s[sgprPersistentIterationEnd], s63  // s_magic mul, div alg 2
s_add_u32 s61, s61, s62
s_and_b32 s63, s[sgprMagicShiftItersPerTile], 2147483647 // tmpS = remove abit to final shift
s_lshr_b32 s61, s61, s63                           // sMagicDiv Alg 2
s_mul_i32 s61, s61, s[sgprItersPerTile]            // start iteration of partial tile
s_sub_u32 s9, s[sgprPersistentIterationEnd], s61   // calc iterations completed by this WG
label_SK_Fixup:
s_lshl_b32 s61, s8, 2                              // flag offset based on CTA index
s_waitcnt vmcnt(0)                                 // acquire: drain before reading partials
v_mov_b32 v12, 0                                   // zero vaddr offset
s_mov_b64 s[68:69], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s70, BufferOOB
s_mov_b32 s71, Srd127_96
buffer_load_dword v11, v12, s[68:71], s61 offen offset:0 sc0 sc1 // acquire: get flag (VMEM)
s_waitcnt vmcnt(0)                                 // acquire: wait VMEM flag load
v_readfirstlane_b32 s63, v11                       // move VMEM flag to SGPR for compare
s_cmp_eq_u32 s63, 1                                // check if ready
s_cbranch_scc0 label_SK_Fixup                      // if flag not set, wait and check again
s_waitcnt vmcnt(0)                                 // acquire: drain before reading partials
s_barrier                                          // wait for all workgroups before resetting flag
v_readfirstlane_b32 s63, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s63, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagReset              // Skip flag reset
v_mov_b32 v11, s63                                 // move flag value to vgpr
v_mov_b32 v12, 0                                   // zero vaddr offset
s_mov_b64 s[68:69], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s70, BufferOOB
s_mov_b32 s71, Srd127_96
buffer_store_dword v11, v12, s[68:71], s61 offen offset:0 sc0 sc1 // reset flag
s_waitcnt vmcnt(0)                                 // release: wait for partials stores before flag
label_SK_SkipFlagReset:
label_Fixup_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=20 */
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s62, 0x10000, s8                         // Offset to correct partials tile (low word)
s_mul_hi_u32 s61, 0x10000, s8                      // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s62      // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s61     // add hi (offset high word + lo carry) to SRD
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Fixup Batch #0 (d1,d0,vc1,vc0) =       */
/*      (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_lshlrev_b32 v12, 4, v[vgprSerial]                // v12 = v[vgprSerial] * 16
s_mov_b32 s62, 0                                   // Init sgpr offset
buffer_load_dwordx4 v[80:83], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[84:87], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[88:91], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[92:95], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[96:99], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[100:103], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[104:107], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[108:111], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[112:115], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[116:119], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[120:123], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[128:131], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[132:135], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[136:139], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[140:143], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
s_add_u32 s62, s62, 4096                           // Inc sgpr offset
buffer_load_dwordx4 v[144:147], v12, s[sgprSrdWS:sgprSrdWS+3], s62 offen offset:0 sc0 sc1 // load WS
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+20], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+21], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+22], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+23], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+24], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+25], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+26], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+27], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+28], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+29], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+30], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+31], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+36], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+37], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+38], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+39], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+40], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+41], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+42], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+43], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+44], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+45], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+46], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+47], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+48], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+49], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+50], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+51], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+52], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+53], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+54], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+55], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+56], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+57], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+58], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+59], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+60], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+61], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+62], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+63], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+64], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+65], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+66], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+67], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+68], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+69], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+70], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+71], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+72], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+73], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+74], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+75], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+76], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+77], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+78], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+79], acc63          // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // wait C (interleaved) 15 = 16 - 0 + 0 - 1
v_add_f32 v[vgprValuC+16], v[vgprValuC+16], v80    // accum partials
v_add_f32 v[vgprValuC+17], v[vgprValuC+17], v81    // accum partials
v_add_f32 v[vgprValuC+18], v[vgprValuC+18], v82    // accum partials
v_add_f32 v[vgprValuC+19], v[vgprValuC+19], v83    // accum partials

s_waitcnt vmcnt(14)                                // wait C (interleaved) 14 = 16 - 1 + 0 - 1
v_add_f32 v[vgprValuC+20], v[vgprValuC+20], v84    // accum partials
v_add_f32 v[vgprValuC+21], v[vgprValuC+21], v85    // accum partials
v_add_f32 v[vgprValuC+22], v[vgprValuC+22], v86    // accum partials
v_add_f32 v[vgprValuC+23], v[vgprValuC+23], v87    // accum partials

s_waitcnt vmcnt(13)                                // wait C (interleaved) 13 = 16 - 2 + 0 - 1
v_add_f32 v[vgprValuC+24], v[vgprValuC+24], v88    // accum partials
v_add_f32 v[vgprValuC+25], v[vgprValuC+25], v89    // accum partials
v_add_f32 v[vgprValuC+26], v[vgprValuC+26], v90    // accum partials
v_add_f32 v[vgprValuC+27], v[vgprValuC+27], v91    // accum partials

s_waitcnt vmcnt(12)                                // wait C (interleaved) 12 = 16 - 3 + 0 - 1
v_add_f32 v[vgprValuC+28], v[vgprValuC+28], v92    // accum partials
v_add_f32 v[vgprValuC+29], v[vgprValuC+29], v93    // accum partials
v_add_f32 v[vgprValuC+30], v[vgprValuC+30], v94    // accum partials
v_add_f32 v[vgprValuC+31], v[vgprValuC+31], v95    // accum partials

s_waitcnt vmcnt(11)                                // wait C (interleaved) 11 = 16 - 4 + 0 - 1
v_add_f32 v[vgprValuC+32], v[vgprValuC+32], v96    // accum partials
v_add_f32 v[vgprValuC+33], v[vgprValuC+33], v97    // accum partials
v_add_f32 v[vgprValuC+34], v[vgprValuC+34], v98    // accum partials
v_add_f32 v[vgprValuC+35], v[vgprValuC+35], v99    // accum partials

s_waitcnt vmcnt(10)                                // wait C (interleaved) 10 = 16 - 5 + 0 - 1
v_add_f32 v[vgprValuC+36], v[vgprValuC+36], v100   // accum partials
v_add_f32 v[vgprValuC+37], v[vgprValuC+37], v101   // accum partials
v_add_f32 v[vgprValuC+38], v[vgprValuC+38], v102   // accum partials
v_add_f32 v[vgprValuC+39], v[vgprValuC+39], v103   // accum partials

s_waitcnt vmcnt(9)                                 // wait C (interleaved) 9 = 16 - 6 + 0 - 1
v_add_f32 v[vgprValuC+40], v[vgprValuC+40], v104   // accum partials
v_add_f32 v[vgprValuC+41], v[vgprValuC+41], v105   // accum partials
v_add_f32 v[vgprValuC+42], v[vgprValuC+42], v106   // accum partials
v_add_f32 v[vgprValuC+43], v[vgprValuC+43], v107   // accum partials

s_waitcnt vmcnt(8)                                 // wait C (interleaved) 8 = 16 - 7 + 0 - 1
v_add_f32 v[vgprValuC+44], v[vgprValuC+44], v108   // accum partials
v_add_f32 v[vgprValuC+45], v[vgprValuC+45], v109   // accum partials
v_add_f32 v[vgprValuC+46], v[vgprValuC+46], v110   // accum partials
v_add_f32 v[vgprValuC+47], v[vgprValuC+47], v111   // accum partials

s_waitcnt vmcnt(7)                                 // wait C (interleaved) 7 = 16 - 8 + 0 - 1
v_add_f32 v[vgprValuC+48], v[vgprValuC+48], v112   // accum partials
v_add_f32 v[vgprValuC+49], v[vgprValuC+49], v113   // accum partials
v_add_f32 v[vgprValuC+50], v[vgprValuC+50], v114   // accum partials
v_add_f32 v[vgprValuC+51], v[vgprValuC+51], v115   // accum partials

s_waitcnt vmcnt(6)                                 // wait C (interleaved) 6 = 16 - 9 + 0 - 1
v_add_f32 v[vgprValuC+52], v[vgprValuC+52], v116   // accum partials
v_add_f32 v[vgprValuC+53], v[vgprValuC+53], v117   // accum partials
v_add_f32 v[vgprValuC+54], v[vgprValuC+54], v118   // accum partials
v_add_f32 v[vgprValuC+55], v[vgprValuC+55], v119   // accum partials

s_waitcnt vmcnt(5)                                 // wait C (interleaved) 5 = 16 - 10 + 0 - 1
v_add_f32 v[vgprValuC+56], v[vgprValuC+56], v120   // accum partials
v_add_f32 v[vgprValuC+57], v[vgprValuC+57], v121   // accum partials
v_add_f32 v[vgprValuC+58], v[vgprValuC+58], v122   // accum partials
v_add_f32 v[vgprValuC+59], v[vgprValuC+59], v123   // accum partials

s_waitcnt vmcnt(4)                                 // wait C (interleaved) 4 = 16 - 11 + 0 - 1
v_add_f32 v[vgprValuC+60], v[vgprValuC+60], v128   // accum partials
v_add_f32 v[vgprValuC+61], v[vgprValuC+61], v129   // accum partials
v_add_f32 v[vgprValuC+62], v[vgprValuC+62], v130   // accum partials
v_add_f32 v[vgprValuC+63], v[vgprValuC+63], v131   // accum partials

s_waitcnt vmcnt(3)                                 // wait C (interleaved) 3 = 16 - 12 + 0 - 1
v_add_f32 v[vgprValuC+64], v[vgprValuC+64], v132   // accum partials
v_add_f32 v[vgprValuC+65], v[vgprValuC+65], v133   // accum partials
v_add_f32 v[vgprValuC+66], v[vgprValuC+66], v134   // accum partials
v_add_f32 v[vgprValuC+67], v[vgprValuC+67], v135   // accum partials

s_waitcnt vmcnt(2)                                 // wait C (interleaved) 2 = 16 - 13 + 0 - 1
v_add_f32 v[vgprValuC+68], v[vgprValuC+68], v136   // accum partials
v_add_f32 v[vgprValuC+69], v[vgprValuC+69], v137   // accum partials
v_add_f32 v[vgprValuC+70], v[vgprValuC+70], v138   // accum partials
v_add_f32 v[vgprValuC+71], v[vgprValuC+71], v139   // accum partials

s_waitcnt vmcnt(1)                                 // wait C (interleaved) 1 = 16 - 14 + 0 - 1
v_add_f32 v[vgprValuC+72], v[vgprValuC+72], v140   // accum partials
v_add_f32 v[vgprValuC+73], v[vgprValuC+73], v141   // accum partials
v_add_f32 v[vgprValuC+74], v[vgprValuC+74], v142   // accum partials
v_add_f32 v[vgprValuC+75], v[vgprValuC+75], v143   // accum partials

s_waitcnt vmcnt(0)                                 // wait C (interleaved) 0 = 16 - 15 + 0 - 1
v_add_f32 v[vgprValuC+76], v[vgprValuC+76], v144   // accum partials
v_add_f32 v[vgprValuC+77], v[vgprValuC+77], v145   // accum partials
v_add_f32 v[vgprValuC+78], v[vgprValuC+78], v146   // accum partials
v_add_f32 v[vgprValuC+79], v[vgprValuC+79], v147   // accum partials
v_accvgpr_write_b32 acc0, v[vgprValuC+16]          // copy vreg[0] to acc
v_accvgpr_write_b32 acc4, v[vgprValuC+17]          // copy vreg[1] to acc
v_accvgpr_write_b32 acc8, v[vgprValuC+18]          // copy vreg[2] to acc
v_accvgpr_write_b32 acc12, v[vgprValuC+19]         // copy vreg[3] to acc
v_accvgpr_write_b32 acc16, v[vgprValuC+20]         // copy vreg[4] to acc
v_accvgpr_write_b32 acc20, v[vgprValuC+21]         // copy vreg[5] to acc
v_accvgpr_write_b32 acc24, v[vgprValuC+22]         // copy vreg[6] to acc
v_accvgpr_write_b32 acc28, v[vgprValuC+23]         // copy vreg[7] to acc
v_accvgpr_write_b32 acc32, v[vgprValuC+24]         // copy vreg[8] to acc
v_accvgpr_write_b32 acc36, v[vgprValuC+25]         // copy vreg[9] to acc
v_accvgpr_write_b32 acc40, v[vgprValuC+26]         // copy vreg[10] to acc
v_accvgpr_write_b32 acc44, v[vgprValuC+27]         // copy vreg[11] to acc
v_accvgpr_write_b32 acc48, v[vgprValuC+28]         // copy vreg[12] to acc
v_accvgpr_write_b32 acc52, v[vgprValuC+29]         // copy vreg[13] to acc
v_accvgpr_write_b32 acc56, v[vgprValuC+30]         // copy vreg[14] to acc
v_accvgpr_write_b32 acc60, v[vgprValuC+31]         // copy vreg[15] to acc
v_accvgpr_write_b32 acc1, v[vgprValuC+32]          // copy vreg[16] to acc
v_accvgpr_write_b32 acc5, v[vgprValuC+33]          // copy vreg[17] to acc
v_accvgpr_write_b32 acc9, v[vgprValuC+34]          // copy vreg[18] to acc
v_accvgpr_write_b32 acc13, v[vgprValuC+35]         // copy vreg[19] to acc
v_accvgpr_write_b32 acc17, v[vgprValuC+36]         // copy vreg[20] to acc
v_accvgpr_write_b32 acc21, v[vgprValuC+37]         // copy vreg[21] to acc
v_accvgpr_write_b32 acc25, v[vgprValuC+38]         // copy vreg[22] to acc
v_accvgpr_write_b32 acc29, v[vgprValuC+39]         // copy vreg[23] to acc
v_accvgpr_write_b32 acc33, v[vgprValuC+40]         // copy vreg[24] to acc
v_accvgpr_write_b32 acc37, v[vgprValuC+41]         // copy vreg[25] to acc
v_accvgpr_write_b32 acc41, v[vgprValuC+42]         // copy vreg[26] to acc
v_accvgpr_write_b32 acc45, v[vgprValuC+43]         // copy vreg[27] to acc
v_accvgpr_write_b32 acc49, v[vgprValuC+44]         // copy vreg[28] to acc
v_accvgpr_write_b32 acc53, v[vgprValuC+45]         // copy vreg[29] to acc
v_accvgpr_write_b32 acc57, v[vgprValuC+46]         // copy vreg[30] to acc
v_accvgpr_write_b32 acc61, v[vgprValuC+47]         // copy vreg[31] to acc
v_accvgpr_write_b32 acc2, v[vgprValuC+48]          // copy vreg[32] to acc
v_accvgpr_write_b32 acc6, v[vgprValuC+49]          // copy vreg[33] to acc
v_accvgpr_write_b32 acc10, v[vgprValuC+50]         // copy vreg[34] to acc
v_accvgpr_write_b32 acc14, v[vgprValuC+51]         // copy vreg[35] to acc
v_accvgpr_write_b32 acc18, v[vgprValuC+52]         // copy vreg[36] to acc
v_accvgpr_write_b32 acc22, v[vgprValuC+53]         // copy vreg[37] to acc
v_accvgpr_write_b32 acc26, v[vgprValuC+54]         // copy vreg[38] to acc
v_accvgpr_write_b32 acc30, v[vgprValuC+55]         // copy vreg[39] to acc
v_accvgpr_write_b32 acc34, v[vgprValuC+56]         // copy vreg[40] to acc
v_accvgpr_write_b32 acc38, v[vgprValuC+57]         // copy vreg[41] to acc
v_accvgpr_write_b32 acc42, v[vgprValuC+58]         // copy vreg[42] to acc
v_accvgpr_write_b32 acc46, v[vgprValuC+59]         // copy vreg[43] to acc
v_accvgpr_write_b32 acc50, v[vgprValuC+60]         // copy vreg[44] to acc
v_accvgpr_write_b32 acc54, v[vgprValuC+61]         // copy vreg[45] to acc
v_accvgpr_write_b32 acc58, v[vgprValuC+62]         // copy vreg[46] to acc
v_accvgpr_write_b32 acc62, v[vgprValuC+63]         // copy vreg[47] to acc
v_accvgpr_write_b32 acc3, v[vgprValuC+64]          // copy vreg[48] to acc
v_accvgpr_write_b32 acc7, v[vgprValuC+65]          // copy vreg[49] to acc
v_accvgpr_write_b32 acc11, v[vgprValuC+66]         // copy vreg[50] to acc
v_accvgpr_write_b32 acc15, v[vgprValuC+67]         // copy vreg[51] to acc
v_accvgpr_write_b32 acc19, v[vgprValuC+68]         // copy vreg[52] to acc
v_accvgpr_write_b32 acc23, v[vgprValuC+69]         // copy vreg[53] to acc
v_accvgpr_write_b32 acc27, v[vgprValuC+70]         // copy vreg[54] to acc
v_accvgpr_write_b32 acc31, v[vgprValuC+71]         // copy vreg[55] to acc
v_accvgpr_write_b32 acc35, v[vgprValuC+72]         // copy vreg[56] to acc
v_accvgpr_write_b32 acc39, v[vgprValuC+73]         // copy vreg[57] to acc
v_accvgpr_write_b32 acc43, v[vgprValuC+74]         // copy vreg[58] to acc
v_accvgpr_write_b32 acc47, v[vgprValuC+75]         // copy vreg[59] to acc
v_accvgpr_write_b32 acc51, v[vgprValuC+76]         // copy vreg[60] to acc
v_accvgpr_write_b32 acc55, v[vgprValuC+77]         // copy vreg[61] to acc
v_accvgpr_write_b32 acc59, v[vgprValuC+78]         // copy vreg[62] to acc
v_accvgpr_write_b32 acc63, v[vgprValuC+79]         // copy vreg[63] to acc
s_nop 1                                            // 2 wait states required before reading vgpr
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_mul_i32 s61, s[sgprskTiles], s[sgprItersPerTile]
s_mul_i32 s62, s[sgprSKItersPerWG], s[sgprskGrid]
s_sub_u32 s61, s61, s62                            // skTiles * ItersPerTile - SKItersPerWG * skGrid
s_bitcmp1_b32 s[sgprMagicShiftItersPerTile], 29    // USO on? (bit 29 of MagicShiftItersPerTile); off -> historical global peer size
s_cbranch_scc0 label_SK_PeerGlobal                 // USO off -> historical global mapping
s_cmp_eq_u32 s[sgprskTiles], 0                     // skTiles == 0?
s_cbranch_scc1 label_SK_PeerNoTiles                // global peer size
v_cvt_f32_u32 v11, s[sgprskTiles]                  // skGrid % skTiles
v_rcp_iflag_f32 v11, v11                           // skGrid % skTiles
v_cvt_f32_u32 v12, s[sgprskGrid]                   // skGrid % skTiles
v_mul_f32 v11, v11, v12                            // skGrid % skTiles
v_cvt_u32_f32 v11, v11                             // skGrid % skTiles
v_mul_u32_u24 v12, v11, s[sgprskTiles]             // skGrid % skTiles
v_sub_u32 v12, s[sgprskGrid], v12                  // skGrid % skTiles
v_cmpx_eq_u32 exec, v12, s[sgprskTiles]            // skGrid % skTiles
v_add_u32 v11, 1, v11                              // skGrid % skTiles
v_mov_b32 v12, 0                                   // skGrid % skTiles
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v12, s[sgprskTiles]            // overflow happened in remainder
v_sub_u32 v11, v11, 1                              // quotient - 1
v_mul_u32_u24 v12, v11, s[sgprskTiles]             // re-calculate remainder
v_sub_u32 v12, s[sgprskGrid], v12                  // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s62, v11                       // quotient
v_readfirstlane_b32 s62, v12                       // remainder
s_cmp_eq_u32 s62, 0                                // skGrid % skTiles == 0?
s_cbranch_scc0 label_SK_PeerGlobal                 // ragged -> global peer
label_SK_PeerPerTile:
v_cvt_f32_u32 v11, s[sgprskTiles]                  // F = skGrid / skTiles
v_rcp_iflag_f32 v11, v11                           // F = skGrid / skTiles
v_cvt_f32_u32 v12, s[sgprskGrid]                   // F = skGrid / skTiles
v_mul_f32 v11, v11, v12                            // F = skGrid / skTiles
v_cvt_u32_f32 v11, v11                             // F = skGrid / skTiles
v_mul_u32_u24 v12, v11, s[sgprskTiles]             // F = skGrid / skTiles
v_sub_u32 v12, s[sgprskGrid], v12                  // F = skGrid / skTiles
v_cmpx_eq_u32 exec, v12, s[sgprskTiles]            // F = skGrid / skTiles
v_add_u32 v11, 1, v11                              // F = skGrid / skTiles
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v12, s[sgprskTiles]            // overflow happened in remainder
v_sub_u32 v11, v11, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s62, v11                       // quotient
s_mov_b32 s61, s62                                 // F = skGrid / skTiles
v_cvt_f32_u32 v11, s61                             // s = cta % F
v_rcp_iflag_f32 v11, v11                           // s = cta % F
v_cvt_f32_u32 v12, s8                              // s = cta % F
v_mul_f32 v11, v11, v12                            // s = cta % F
v_cvt_u32_f32 v11, v11                             // s = cta % F
v_mul_u32_u24 v12, v11, s61                        // s = cta % F
v_sub_u32 v12, s8, v12                             // s = cta % F
v_cmpx_eq_u32 exec, v12, s61                       // s = cta % F
v_add_u32 v11, 1, v11                              // s = cta % F
v_mov_b32 v12, 0                                   // s = cta % F
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v12, s61                       // overflow happened in remainder
v_sub_u32 v11, v11, 1                              // quotient - 1
v_mul_u32_u24 v12, v11, s61                        // re-calculate remainder
v_sub_u32 v12, s8, v12                             // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s62, v11                       // quotient
v_readfirstlane_b32 s62, v12                       // remainder
s_mul_i32 s61, s61, s[sgprSKItersPerWG]            // F * SKItersPerWG
s_sub_u32 s61, s[sgprItersPerTile], s61            // remI = ItersPerTile - F*W
s_cmp_lt_u32 s62, s61                              // s < remI?
s_cselect_b32 s62, 1, 0                            // extra iter within tile
s_add_u32 s62, s[sgprSKItersPerWG], s62            // chunk = W + (s < remI)
s_branch label_SK_PeerDone                         // skip global peer
label_SK_PeerNoTiles:
label_SK_PeerGlobal:
s_add_u32 s62, s[sgprSKItersPerWG], 1              // Add extra iter
s_cmp_lt_u32 s8, s61                               // Check if next WG had an extra iteration
s_cselect_b32 s62, s62, s[sgprSKItersPerWG]        // Select correct number of iterations for next WG
label_SK_PeerDone:
s_add_u32 s9, s9, s62                              // next partial tile iteration
s_add_u32 s8, s8, 1                                // next partial tile index
s_cmp_lt_u32 s9, s[sgprItersPerTile]               // done loading partial tiles?
s_cbranch_scc1 label_SK_Fixup                      // Branch to continue fixup loop
label_SK_Store:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s62, 127, s[sgprSizeI]                   // s62 = s[sgprSizeI] % 128
s_add_u32 s63, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s63                // wg0 >= nwg0-1 ?
s_cselect_b32 s62, s62, 0                          // set rem
s_cmpk_gt_u32 s62, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s62, 127, s[sgprSizeJ]                   // s62 = s[sgprSizeJ] % 128
s_add_u32 s63, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s63                // wg1 >= nwg1-1
s_cselect_b32 s62, s62, 0                          // set rem
s_cmpk_gt_u32 s62, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=42 */
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
v_add_lshl_u32 v11, v3, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+20], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+21], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+22], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+23], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+24], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+25], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+26], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+27], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+28], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+29], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+30], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+31], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+36], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+37], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+38], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+39], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+40], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+41], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+42], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+43], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+44], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+45], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+46], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+47], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+48], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+49], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+50], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+51], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+52], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+53], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+54], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+55], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+56], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+57], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+58], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+59], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+60], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+61], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+62], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+63], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+64], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+65], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+66], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+67], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+68], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+69], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+70], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+71], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+72], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+73], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+74], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+75], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+76], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+77], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+78], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+79], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v20, v[vgprValuC+20], v[vgprValuC+21] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v21, v[vgprValuC+22], v[vgprValuC+23] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[20:21], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[24:25], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v28, v[vgprValuC+28], v[vgprValuC+29] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v29, v[vgprValuC+30], v[vgprValuC+31] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[28:29], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v32, v[vgprValuC+32], v[vgprValuC+33] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v33, v[vgprValuC+34], v[vgprValuC+35] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[32:33], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v36, v[vgprValuC+36], v[vgprValuC+37] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v37, v[vgprValuC+38], v[vgprValuC+39] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[36:37], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v40, v[vgprValuC+40], v[vgprValuC+41] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v41, v[vgprValuC+42], v[vgprValuC+43] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[40:41], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v44, v[vgprValuC+44], v[vgprValuC+45] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v45, v[vgprValuC+46], v[vgprValuC+47] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[44:45], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v48, v[vgprValuC+48], v[vgprValuC+49] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v49, v[vgprValuC+50], v[vgprValuC+51] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[48:49], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v52, v[vgprValuC+52], v[vgprValuC+53] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v53, v[vgprValuC+54], v[vgprValuC+55] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[52:53], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v56, v[vgprValuC+56], v[vgprValuC+57] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v57, v[vgprValuC+58], v[vgprValuC+59] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[56:57], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v60, v[vgprValuC+60], v[vgprValuC+61] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v61, v[vgprValuC+62], v[vgprValuC+63] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[60:61], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v64, v[vgprValuC+64], v[vgprValuC+65] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v65, v[vgprValuC+66], v[vgprValuC+67] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[64:65], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v68, v[vgprValuC+68], v[vgprValuC+69] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v69, v[vgprValuC+70], v[vgprValuC+71] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[68:69], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v72, v[vgprValuC+72], v[vgprValuC+73] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v73, v[vgprValuC+74], v[vgprValuC+75] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[72:73], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v76, v[vgprValuC+76], v[vgprValuC+77] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v77, v[vgprValuC+78], v[vgprValuC+79] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[76:77], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=34 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v11, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v76, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v76, v6, v76, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v77, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v6, v77, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v78, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v78, v6, v78, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v79, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v79, v6, v79, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v80, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v80, v6, v80, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v81, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v6, v81, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v82, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v6, v82, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v83, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v6, v83, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v84, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v6, v84, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v85, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v85, v6, v85, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v86, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v6, v86, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v87, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v6, v87, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v88, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v6, v88, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v89, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v89, v6, v89, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v90, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v6, v90, s[66:67]               // LDD clip if OOB. offset
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
v_accvgpr_read_b32 v[vgprValuC+72], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+73], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+74], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+75], acc63          // copy acc to vreg[63]

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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_pk_f16_f32 v12, v[vgprValuC+12], v[vgprValuC+13] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v13, v[vgprValuC+14], v[vgprValuC+15] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[12:13], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v20, v[vgprValuC+20], v[vgprValuC+21] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v21, v[vgprValuC+22], v[vgprValuC+23] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[20:21], v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[24:25], v78, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v28, v[vgprValuC+28], v[vgprValuC+29] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v29, v[vgprValuC+30], v[vgprValuC+31] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[28:29], v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v32, v[vgprValuC+32], v[vgprValuC+33] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v33, v[vgprValuC+34], v[vgprValuC+35] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[32:33], v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v36, v[vgprValuC+36], v[vgprValuC+37] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v37, v[vgprValuC+38], v[vgprValuC+39] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[36:37], v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v40, v[vgprValuC+40], v[vgprValuC+41] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v41, v[vgprValuC+42], v[vgprValuC+43] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[40:41], v82, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v44, v[vgprValuC+44], v[vgprValuC+45] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v45, v[vgprValuC+46], v[vgprValuC+47] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[44:45], v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v48, v[vgprValuC+48], v[vgprValuC+49] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v49, v[vgprValuC+50], v[vgprValuC+51] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[48:49], v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v52, v[vgprValuC+52], v[vgprValuC+53] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v53, v[vgprValuC+54], v[vgprValuC+55] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[52:53], v85, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v56, v[vgprValuC+56], v[vgprValuC+57] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v57, v[vgprValuC+58], v[vgprValuC+59] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[56:57], v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v60, v[vgprValuC+60], v[vgprValuC+61] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v61, v[vgprValuC+62], v[vgprValuC+63] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[60:61], v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v64, v[vgprValuC+64], v[vgprValuC+65] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v65, v[vgprValuC+66], v[vgprValuC+67] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[64:65], v88, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v68, v[vgprValuC+68], v[vgprValuC+69] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v69, v[vgprValuC+70], v[vgprValuC+71] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[68:69], v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_pk_f16_f32 v72, v[vgprValuC+72], v[vgprValuC+73] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v73, v[vgprValuC+74], v[vgprValuC+75] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[72:73], v90, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=88 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,8,0:vw1); (0,0,8,1:vw1); (0,0,8,2:vw1); (0,0,8,3:vw1); (0,0,9,0:vw1); (0,0,9,1:vw1); (0,0,9,2:vw1); (0,0,9,3:vw1); (0,0,10,0:vw1); (0,0,10,1:vw1); (0,0,10,2:vw1); (0,0,10,3:vw1); (0,0,11,0:vw1); (0,0,11,1:vw1); (0,0,11,2:vw1); (0,0,11,3:vw1); (0,0,12,0:vw1); (0,0,12,1:vw1); (0,0,12,2:vw1); (0,0,12,3:vw1); (0,0,13,0:vw1); (0,0,13,1:vw1); (0,0,13,2:vw1); (0,0,13,3:vw1); (0,0,14,0:vw1); (0,0,14,1:vw1); (0,0,14,2:vw1); (0,0,14,3:vw1); (0,0,15,0:vw1); (0,0,15,1:vw1); (0,0,15,2:vw1); (0,0,15,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v75, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v6, v75, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v76, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v76, v6, v76, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v77, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v6, v77, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v78, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v78, v6, v78, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v79, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v79, v6, v79, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v80, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v80, v6, v80, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v81, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v6, v81, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v82, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v6, v82, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v83, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v6, v83, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v84, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v6, v84, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v85, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v85, v6, v85, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v86, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v6, v86, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v87, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v6, v87, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v88, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v6, v88, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v89, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v89, v6, v89, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v90, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v6, v90, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v91, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v91, v6, v91, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v92, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v6, v92, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v93, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v6, v93, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v94, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v94, v6, v94, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v95, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v95, v6, v95, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v96, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v6, v96, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v97, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v97, v6, v97, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v98, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v6, v98, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v99, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v6, v99, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v100, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v6, v100, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v101, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v6, v101, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v102, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v6, v102, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v103, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v103, v6, v103, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v104, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v6, v104, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v105, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v6, v105, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v106, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v6, v106, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v107, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v6, v107, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v108, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v6, v108, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v109, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v109, v6, v109, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v110, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v111, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v112, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v6, v112, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v113, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v113, v6, v113, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v114, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v6, v114, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v115, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v115, v6, v115, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v116, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v117, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v118, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v6, v118, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v119, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v119, v6, v119, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v120, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v6, v120, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v121, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v121, v6, v121, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v122, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v6, v122, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v123, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v123, v6, v123, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v125, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v6, v125, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v126, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v126, v6, v126, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v127, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v6, v127, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v128, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v128, v6, v128, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v129, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v6, v129, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v130, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v6, v130, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v131, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v6, v131, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v132, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v132, v6, v132, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v133, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v6, v133, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v134, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v134, v6, v134, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v135, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v6, v135, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v136, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v6, v136, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v137, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v6, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v138, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v138, v6, v138, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v139, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v6, v139, s[66:67]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+14], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+15], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+16], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+17], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+18], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+19], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+20], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+21], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+22], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+23], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+24], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+25], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+26], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+27], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+28], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+29], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+30], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+31], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+32], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+33], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+34], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+35], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+36], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+37], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+38], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+39], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+40], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+41], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+42], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+43], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+44], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+45], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+46], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+47], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+48], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+49], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+50], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+51], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+52], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+53], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+54], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+55], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+56], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+57], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+58], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+59], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+60], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+61], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+62], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+63], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+64], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+65], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+66], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+67], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+68], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+69], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+70], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+71], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+72], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+73], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+74], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 8, 0), (0, 0, 8, 1), (0, 0, 8, 2), (0, 0, 8, 3), (0, 0, 9, 0), (0, 0, 9, 1), (0, 0, 9, 2), (0, 0, 9, 3), (0, 0, 10, 0), (0, 0, 10, 1), (0, 0, 10, 2), (0, 0, 10, 3), (0, 0, 11, 0), (0, 0, 11, 1), (0, 0, 11, 2), (0, 0, 11, 3), (0, 0, 12, 0), (0, 0, 12, 1), (0, 0, 12, 2), (0, 0, 12, 3), (0, 0, 13, 0), (0, 0, 13, 1), (0, 0, 13, 2), (0, 0, 13, 3), (0, 0, 14, 0), (0, 0, 14, 1), (0, 0, 14, 2), (0, 0, 14, 3), (0, 0, 15, 0), (0, 0, 15, 1), (0, 0, 15, 2), (0, 0, 15, 3)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v14, v[vgprValuC+14] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v14, v78, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v15, v[vgprValuC+15] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v15, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v16, v[vgprValuC+16] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v16, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v17, v[vgprValuC+17] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v17, v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v18, v[vgprValuC+18] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v18, v82, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v19, v[vgprValuC+19] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v19, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v20, v[vgprValuC+20] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v20, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v21, v[vgprValuC+21] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v21, v85, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v22, v[vgprValuC+22] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v22, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v23, v[vgprValuC+23] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v23, v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v24, v[vgprValuC+24] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v24, v88, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v25, v[vgprValuC+25] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v25, v89, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v26, v[vgprValuC+26] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v26, v90, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v27, v[vgprValuC+27] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v27, v91, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v28, v[vgprValuC+28] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v28, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v29, v[vgprValuC+29] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v29, v93, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v30, v[vgprValuC+30] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v30, v94, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v31, v[vgprValuC+31] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v31, v95, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v32, v[vgprValuC+32] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v32, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v33, v[vgprValuC+33] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v33, v97, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v34, v[vgprValuC+34] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v34, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v35, v[vgprValuC+35] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v35, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v36, v[vgprValuC+36] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v36, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v37, v[vgprValuC+37] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v37, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v38, v[vgprValuC+38] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v38, v102, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v39, v[vgprValuC+39] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v39, v103, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v40, v[vgprValuC+40] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v40, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v41, v[vgprValuC+41] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v41, v105, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v42, v[vgprValuC+42] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v42, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v43, v[vgprValuC+43] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v43, v107, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v44, v[vgprValuC+44] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v44, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v45, v[vgprValuC+45] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v45, v109, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v46, v[vgprValuC+46] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v46, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v47, v[vgprValuC+47] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v47, v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v48, v[vgprValuC+48] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v48, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v49, v[vgprValuC+49] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v49, v113, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v50, v[vgprValuC+50] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v50, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v51, v[vgprValuC+51] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v51, v115, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v52, v[vgprValuC+52] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v52, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v53, v[vgprValuC+53] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v53, v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v54, v[vgprValuC+54] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v54, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v55, v[vgprValuC+55] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v55, v119, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v56, v[vgprValuC+56] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v56, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v57, v[vgprValuC+57] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v57, v121, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v58, v[vgprValuC+58] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v58, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v59, v[vgprValuC+59] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v59, v123, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v60, v[vgprValuC+60] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v60, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v61, v[vgprValuC+61] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v61, v126, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v62, v[vgprValuC+62] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v62, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v63, v[vgprValuC+63] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v63, v128, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v64, v[vgprValuC+64] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v64, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v65, v[vgprValuC+65] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v65, v130, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v66, v[vgprValuC+66] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v66, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v67, v[vgprValuC+67] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v67, v132, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v68, v[vgprValuC+68] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v68, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v69, v[vgprValuC+69] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v69, v134, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v70, v[vgprValuC+70] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v70, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v71, v[vgprValuC+71] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v71, v136, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v72, v[vgprValuC+72] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v72, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v73, v[vgprValuC+73] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v73, v138, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v74, v[vgprValuC+74] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v74, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s62, 127, s[sgprSizeI]                   // s62 = s[sgprSizeI] % 128
s_add_u32 s63, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s63                // wg0 >= nwg0-1 ?
s_cselect_b32 s62, s62, 0                          // set rem
s_cmpk_gt_u32 s62, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 128 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s62, 127, s[sgprSizeJ]                   // s62 = s[sgprSizeJ] % 128
s_add_u32 s63, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s63                // wg1 >= nwg1-1
s_cselect_b32 s62, s62, 0                          // set rem
s_cmpk_gt_u32 s62, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=28 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v12, v2, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
buffer_load_dwordx2 v[14:15], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[80:81], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[82:83], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[84:85], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[86:87], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[88:89], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[90:91], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[92:93], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[94:95], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[96:97], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[98:99], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[100:101], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[102:103], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[104:105], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[106:107], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_dwordx2 v[108:109], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v11, v3, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+20], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+21], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+22], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+23], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+24], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+25], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+26], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+27], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+28], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+29], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+30], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+31], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+36], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+37], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+38], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+39], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+40], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+41], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+42], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+43], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+44], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+45], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+46], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+47], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+48], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+49], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+50], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+51], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+52], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+53], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+54], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+55], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+56], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+57], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+58], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+59], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+60], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+61], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+62], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+63], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+64], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+65], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+66], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+67], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+68], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+69], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+70], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+71], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+72], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+73], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+74], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+75], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+76], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+77], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+78], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+79], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (0, 0, 4, 0), (0, 0, 5, 0), (0, 0, 6, 0), (0, 0, 7, 0), (0, 0, 8, 0), (0, 0, 9, 0), (0, 0, 10, 0), (0, 0, 11, 0), (0, 0, 12, 0), (0, 0, 13, 0), (0, 0, 14, 0), (0, 0, 15, 0)] */
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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(15)                                // vlcnt(15) = 16 - 1 (beta) vscnt(0) (interleaved)
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v14, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v14, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v15, v[vgprValuC+18] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v15, v[vgprValuC+19] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(14) = 16 - 2 (beta) vscnt(1) (interleaved)
v_fma_mix_f32 v[vgprValuC+20], s[sgprBeta], v80, v[vgprValuC+20] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+21], s[sgprBeta], v80, v[vgprValuC+21] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+22], s[sgprBeta], v81, v[vgprValuC+22] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+23], s[sgprBeta], v81, v[vgprValuC+23] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v20, v[vgprValuC+20], v[vgprValuC+21] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v21, v[vgprValuC+22], v[vgprValuC+23] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[20:21], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(13) = 16 - 3 (beta) vscnt(2) (interleaved)
v_fma_mix_f32 v[vgprValuC+24], s[sgprBeta], v82, v[vgprValuC+24] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+25], s[sgprBeta], v82, v[vgprValuC+25] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+26], s[sgprBeta], v83, v[vgprValuC+26] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+27], s[sgprBeta], v83, v[vgprValuC+27] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[24:25], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(12) = 16 - 4 (beta) vscnt(3) (interleaved)
v_fma_mix_f32 v[vgprValuC+28], s[sgprBeta], v84, v[vgprValuC+28] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+29], s[sgprBeta], v84, v[vgprValuC+29] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+30], s[sgprBeta], v85, v[vgprValuC+30] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+31], s[sgprBeta], v85, v[vgprValuC+31] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v28, v[vgprValuC+28], v[vgprValuC+29] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v29, v[vgprValuC+30], v[vgprValuC+31] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[28:29], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(11) = 16 - 5 (beta) vscnt(4) (interleaved)
v_fma_mix_f32 v[vgprValuC+32], s[sgprBeta], v86, v[vgprValuC+32] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+33], s[sgprBeta], v86, v[vgprValuC+33] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+34], s[sgprBeta], v87, v[vgprValuC+34] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+35], s[sgprBeta], v87, v[vgprValuC+35] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v32, v[vgprValuC+32], v[vgprValuC+33] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v33, v[vgprValuC+34], v[vgprValuC+35] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[32:33], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(10) = 16 - 6 (beta) vscnt(5) (interleaved)
v_fma_mix_f32 v[vgprValuC+36], s[sgprBeta], v88, v[vgprValuC+36] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+37], s[sgprBeta], v88, v[vgprValuC+37] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+38], s[sgprBeta], v89, v[vgprValuC+38] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+39], s[sgprBeta], v89, v[vgprValuC+39] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v36, v[vgprValuC+36], v[vgprValuC+37] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v37, v[vgprValuC+38], v[vgprValuC+39] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[36:37], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(9) = 16 - 7 (beta) vscnt(6) (interleaved)
v_fma_mix_f32 v[vgprValuC+40], s[sgprBeta], v90, v[vgprValuC+40] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+41], s[sgprBeta], v90, v[vgprValuC+41] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+42], s[sgprBeta], v91, v[vgprValuC+42] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+43], s[sgprBeta], v91, v[vgprValuC+43] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v40, v[vgprValuC+40], v[vgprValuC+41] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v41, v[vgprValuC+42], v[vgprValuC+43] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[40:41], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(8) = 16 - 8 (beta) vscnt(7) (interleaved)
v_fma_mix_f32 v[vgprValuC+44], s[sgprBeta], v92, v[vgprValuC+44] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+45], s[sgprBeta], v92, v[vgprValuC+45] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+46], s[sgprBeta], v93, v[vgprValuC+46] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+47], s[sgprBeta], v93, v[vgprValuC+47] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v44, v[vgprValuC+44], v[vgprValuC+45] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v45, v[vgprValuC+46], v[vgprValuC+47] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[44:45], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(7) = 16 - 9 (beta) vscnt(8) (interleaved)
v_fma_mix_f32 v[vgprValuC+48], s[sgprBeta], v94, v[vgprValuC+48] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+49], s[sgprBeta], v94, v[vgprValuC+49] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+50], s[sgprBeta], v95, v[vgprValuC+50] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+51], s[sgprBeta], v95, v[vgprValuC+51] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v48, v[vgprValuC+48], v[vgprValuC+49] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v49, v[vgprValuC+50], v[vgprValuC+51] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[48:49], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(6) = 16 - 10 (beta) vscnt(9) (interleaved)
v_fma_mix_f32 v[vgprValuC+52], s[sgprBeta], v96, v[vgprValuC+52] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+53], s[sgprBeta], v96, v[vgprValuC+53] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+54], s[sgprBeta], v97, v[vgprValuC+54] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+55], s[sgprBeta], v97, v[vgprValuC+55] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v52, v[vgprValuC+52], v[vgprValuC+53] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v53, v[vgprValuC+54], v[vgprValuC+55] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[52:53], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(5) = 16 - 11 (beta) vscnt(10) (interleaved)
v_fma_mix_f32 v[vgprValuC+56], s[sgprBeta], v98, v[vgprValuC+56] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+57], s[sgprBeta], v98, v[vgprValuC+57] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+58], s[sgprBeta], v99, v[vgprValuC+58] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+59], s[sgprBeta], v99, v[vgprValuC+59] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v56, v[vgprValuC+56], v[vgprValuC+57] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v57, v[vgprValuC+58], v[vgprValuC+59] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[56:57], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(4) = 16 - 12 (beta) vscnt(11) (interleaved)
v_fma_mix_f32 v[vgprValuC+60], s[sgprBeta], v100, v[vgprValuC+60] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+61], s[sgprBeta], v100, v[vgprValuC+61] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+62], s[sgprBeta], v101, v[vgprValuC+62] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+63], s[sgprBeta], v101, v[vgprValuC+63] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v60, v[vgprValuC+60], v[vgprValuC+61] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v61, v[vgprValuC+62], v[vgprValuC+63] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[60:61], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(3) = 16 - 13 (beta) vscnt(12) (interleaved)
v_fma_mix_f32 v[vgprValuC+64], s[sgprBeta], v102, v[vgprValuC+64] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+65], s[sgprBeta], v102, v[vgprValuC+65] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+66], s[sgprBeta], v103, v[vgprValuC+66] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+67], s[sgprBeta], v103, v[vgprValuC+67] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v64, v[vgprValuC+64], v[vgprValuC+65] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v65, v[vgprValuC+66], v[vgprValuC+67] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[64:65], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(2) = 16 - 14 (beta) vscnt(13) (interleaved)
v_fma_mix_f32 v[vgprValuC+68], s[sgprBeta], v104, v[vgprValuC+68] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+69], s[sgprBeta], v104, v[vgprValuC+69] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+70], s[sgprBeta], v105, v[vgprValuC+70] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+71], s[sgprBeta], v105, v[vgprValuC+71] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v68, v[vgprValuC+68], v[vgprValuC+69] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v69, v[vgprValuC+70], v[vgprValuC+71] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[68:69], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(1) = 16 - 15 (beta) vscnt(14) (interleaved)
v_fma_mix_f32 v[vgprValuC+72], s[sgprBeta], v106, v[vgprValuC+72] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+73], s[sgprBeta], v106, v[vgprValuC+73] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+74], s[sgprBeta], v107, v[vgprValuC+74] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+75], s[sgprBeta], v107, v[vgprValuC+75] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v72, v[vgprValuC+72], v[vgprValuC+73] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v73, v[vgprValuC+74], v[vgprValuC+75] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[72:73], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(15)                                // vlcnt(0) = 16 - 16 (beta) vscnt(15) (interleaved)
v_fma_mix_f32 v[vgprValuC+76], s[sgprBeta], v108, v[vgprValuC+76] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+77], s[sgprBeta], v108, v[vgprValuC+77] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+78], s[sgprBeta], v109, v[vgprValuC+78] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+79], s[sgprBeta], v109, v[vgprValuC+79] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v76, v[vgprValuC+76], v[vgprValuC+77] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v77, v[vgprValuC+78], v[vgprValuC+79] // convert C to fp16 and pack with neighbor
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_dwordx2 v[76:77], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=24 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v11, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[76:77], v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v11, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v80, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v80, v6, v80, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[78:79], v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v80, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v80, v6, v80, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v81, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v6, v81, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[82:83], v81, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v81, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v6, v81, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v86, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v6, v86, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[84:85], v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v86, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v6, v86, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v87, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v6, v87, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[88:89], v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v87, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v6, v87, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v92, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v6, v92, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[90:91], v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v92, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v6, v92, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v93, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v6, v93, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[94:95], v93, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v93, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v6, v93, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v98, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v6, v98, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[96:97], v98, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v98, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v6, v98, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v99, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v6, v99, s[66:67]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[100:101], v99, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v99, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v6, v99, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v104, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v6, v104, s[66:67]             // LDC clip if OOB. offset
buffer_load_dwordx2 v[102:103], v104, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v104, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v6, v104, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v105, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v6, v105, s[66:67]             // LDC clip if OOB. offset
buffer_load_dwordx2 v[106:107], v105, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v105, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v6, v105, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v110, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[66:67]             // LDC clip if OOB. offset
buffer_load_dwordx2 v[108:109], v110, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v110, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v111, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[66:67]             // LDC clip if OOB. offset
buffer_load_dwordx2 v[112:113], v111, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v111, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v6, v111, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v116, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[66:67]             // LDC clip if OOB. offset
buffer_load_dwordx2 v[114:115], v116, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v116, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v117, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[66:67]             // LDC clip if OOB. offset
buffer_load_dwordx2 v[118:119], v117, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v117, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v117, v6, v117, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v122, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v6, v122, s[66:67]             // LDC clip if OOB. offset
buffer_load_dwordx2 v[120:121], v122, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v122, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v6, v122, s[66:67]             // LDD clip if OOB. offset
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
v_accvgpr_read_b32 v[vgprValuC+72], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+73], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+74], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+75], acc63          // copy acc to vreg[63]

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
v_pk_mul_f32 v[vgprValuC+72:vgprValuC+72+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+72:vgprValuC+72+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+74:vgprValuC+74+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+74:vgprValuC+74+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v76, v[vgprValuC+12] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v76, v[vgprValuC+13] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+14], s[sgprBeta], v77, v[vgprValuC+14] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+15], s[sgprBeta], v77, v[vgprValuC+15] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v12, v[vgprValuC+12], v[vgprValuC+13] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v13, v[vgprValuC+14], v[vgprValuC+15] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[12:13], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v78, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v78, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v79, v[vgprValuC+18] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v79, v[vgprValuC+19] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+20], s[sgprBeta], v82, v[vgprValuC+20] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+21], s[sgprBeta], v82, v[vgprValuC+21] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+22], s[sgprBeta], v83, v[vgprValuC+22] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+23], s[sgprBeta], v83, v[vgprValuC+23] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v20, v[vgprValuC+20], v[vgprValuC+21] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v21, v[vgprValuC+22], v[vgprValuC+23] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[20:21], v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+24], s[sgprBeta], v84, v[vgprValuC+24] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+25], s[sgprBeta], v84, v[vgprValuC+25] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+26], s[sgprBeta], v85, v[vgprValuC+26] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+27], s[sgprBeta], v85, v[vgprValuC+27] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[24:25], v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+28], s[sgprBeta], v88, v[vgprValuC+28] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+29], s[sgprBeta], v88, v[vgprValuC+29] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+30], s[sgprBeta], v89, v[vgprValuC+30] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+31], s[sgprBeta], v89, v[vgprValuC+31] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v28, v[vgprValuC+28], v[vgprValuC+29] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v29, v[vgprValuC+30], v[vgprValuC+31] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[28:29], v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+32], s[sgprBeta], v90, v[vgprValuC+32] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+33], s[sgprBeta], v90, v[vgprValuC+33] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+34], s[sgprBeta], v91, v[vgprValuC+34] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+35], s[sgprBeta], v91, v[vgprValuC+35] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v32, v[vgprValuC+32], v[vgprValuC+33] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v33, v[vgprValuC+34], v[vgprValuC+35] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[32:33], v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+36], s[sgprBeta], v94, v[vgprValuC+36] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+37], s[sgprBeta], v94, v[vgprValuC+37] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+38], s[sgprBeta], v95, v[vgprValuC+38] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+39], s[sgprBeta], v95, v[vgprValuC+39] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v36, v[vgprValuC+36], v[vgprValuC+37] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v37, v[vgprValuC+38], v[vgprValuC+39] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[36:37], v93, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+40], s[sgprBeta], v96, v[vgprValuC+40] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+41], s[sgprBeta], v96, v[vgprValuC+41] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+42], s[sgprBeta], v97, v[vgprValuC+42] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+43], s[sgprBeta], v97, v[vgprValuC+43] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v40, v[vgprValuC+40], v[vgprValuC+41] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v41, v[vgprValuC+42], v[vgprValuC+43] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[40:41], v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+44], s[sgprBeta], v100, v[vgprValuC+44] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+45], s[sgprBeta], v100, v[vgprValuC+45] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+46], s[sgprBeta], v101, v[vgprValuC+46] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+47], s[sgprBeta], v101, v[vgprValuC+47] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v44, v[vgprValuC+44], v[vgprValuC+45] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v45, v[vgprValuC+46], v[vgprValuC+47] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[44:45], v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+48], s[sgprBeta], v102, v[vgprValuC+48] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+49], s[sgprBeta], v102, v[vgprValuC+49] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+50], s[sgprBeta], v103, v[vgprValuC+50] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+51], s[sgprBeta], v103, v[vgprValuC+51] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v48, v[vgprValuC+48], v[vgprValuC+49] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v49, v[vgprValuC+50], v[vgprValuC+51] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[48:49], v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+52], s[sgprBeta], v106, v[vgprValuC+52] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+53], s[sgprBeta], v106, v[vgprValuC+53] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+54], s[sgprBeta], v107, v[vgprValuC+54] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+55], s[sgprBeta], v107, v[vgprValuC+55] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v52, v[vgprValuC+52], v[vgprValuC+53] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v53, v[vgprValuC+54], v[vgprValuC+55] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[52:53], v105, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+56], s[sgprBeta], v108, v[vgprValuC+56] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+57], s[sgprBeta], v108, v[vgprValuC+57] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+58], s[sgprBeta], v109, v[vgprValuC+58] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+59], s[sgprBeta], v109, v[vgprValuC+59] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v56, v[vgprValuC+56], v[vgprValuC+57] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v57, v[vgprValuC+58], v[vgprValuC+59] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[56:57], v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+60], s[sgprBeta], v112, v[vgprValuC+60] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+61], s[sgprBeta], v112, v[vgprValuC+61] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+62], s[sgprBeta], v113, v[vgprValuC+62] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+63], s[sgprBeta], v113, v[vgprValuC+63] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v60, v[vgprValuC+60], v[vgprValuC+61] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v61, v[vgprValuC+62], v[vgprValuC+63] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[60:61], v111, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+64], s[sgprBeta], v114, v[vgprValuC+64] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+65], s[sgprBeta], v114, v[vgprValuC+65] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+66], s[sgprBeta], v115, v[vgprValuC+66] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+67], s[sgprBeta], v115, v[vgprValuC+67] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v64, v[vgprValuC+64], v[vgprValuC+65] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v65, v[vgprValuC+66], v[vgprValuC+67] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[64:65], v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+68], s[sgprBeta], v118, v[vgprValuC+68] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+69], s[sgprBeta], v118, v[vgprValuC+69] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+70], s[sgprBeta], v119, v[vgprValuC+70] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+71], s[sgprBeta], v119, v[vgprValuC+71] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v68, v[vgprValuC+68], v[vgprValuC+69] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v69, v[vgprValuC+70], v[vgprValuC+71] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[68:69], v117, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+72], s[sgprBeta], v120, v[vgprValuC+72] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+73], s[sgprBeta], v120, v[vgprValuC+73] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+74], s[sgprBeta], v121, v[vgprValuC+74] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+75], s[sgprBeta], v121, v[vgprValuC+75] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v72, v[vgprValuC+72], v[vgprValuC+73] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v73, v[vgprValuC+74], v[vgprValuC+75] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[72:73], v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=58 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,1,0:vw1); (0,0,1,1:vw1); (0,0,1,2:vw1); (0,0,1,3:vw1); (0,0,2,0:vw1); (0,0,2,1:vw1); (0,0,2,2:vw1); (0,0,2,3:vw1); (0,0,3,0:vw1); (0,0,3,1:vw1); (0,0,3,2:vw1); (0,0,3,3:vw1); (0,0,4,0:vw1); (0,0,4,1:vw1); (0,0,4,2:vw1); (0,0,4,3:vw1); (0,0,5,0:vw1); (0,0,5,1:vw1); (0,0,5,2:vw1); (0,0,5,3:vw1); (0,0,6,0:vw1); (0,0,6,1:vw1); (0,0,6,2:vw1); (0,0,6,3:vw1); (0,0,7,0:vw1); (0,0,7,1:vw1); (0,0,7,2:vw1); (0,0,7,3:vw1); (0,0,8,0:vw1); (0,0,8,1:vw1); (0,0,8,2:vw1); (0,0,8,3:vw1); (0,0,9,0:vw1); (0,0,9,1:vw1); (0,0,9,2:vw1); (0,0,9,3:vw1); (0,0,10,0:vw1); (0,0,10,1:vw1); (0,0,10,2:vw1); (0,0,10,3:vw1); (0,0,11,0:vw1); (0,0,11,1:vw1); (0,0,11,2:vw1); (0,0,11,3:vw1); (0,0,12,0:vw1); (0,0,12,1:vw1); (0,0,12,2:vw1); (0,0,12,3:vw1); (0,0,13,0:vw1); (0,0,13,1:vw1); (0,0,13,2:vw1); (0,0,13,3:vw1); (0,0,14,0:vw1); (0,0,14,1:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v70, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v70, v6, v70, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v69, v70, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v70, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v70, v6, v70, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v72, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v71, v72, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v72, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v6, v72, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v74, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v74, v6, v74, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v73, v74, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v74, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v74, v6, v74, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v76, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v76, v6, v76, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v75, v76, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v76, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v76, v6, v76, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v78, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v78, v6, v78, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v77, v78, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v78, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v78, v6, v78, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v80, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v80, v6, v80, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v79, v80, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v80, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v80, v6, v80, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v82, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v6, v82, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v81, v82, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v82, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v82, v6, v82, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v84, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v6, v84, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v83, v84, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v84, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v84, v6, v84, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v86, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v6, v86, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v85, v86, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v86, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v86, v6, v86, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v88, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v6, v88, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v87, v88, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v88, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v88, v6, v88, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v90, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v6, v90, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v89, v90, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v90, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v90, v6, v90, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v92, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v6, v92, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v91, v92, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v92, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v6, v92, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v94, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v94, v6, v94, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v93, v94, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v94, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v94, v6, v94, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v96, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v6, v96, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v95, v96, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v96, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v96, v6, v96, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v98, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v6, v98, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v97, v98, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v98, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v6, v98, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v100, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v6, v100, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v99, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v100, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v6, v100, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v102, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v6, v102, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v101, v102, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v102, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v102, v6, v102, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v104, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v6, v104, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v103, v104, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v104, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v104, v6, v104, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v106, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v6, v106, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v105, v106, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v106, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v6, v106, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,4,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v108, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v6, v108, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v107, v108, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v108, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v108, v6, v108, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v110, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v109, v110, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v110, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v6, v110, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v112, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v6, v112, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v111, v112, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v112, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v112, v6, v112, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v114, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v6, v114, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v113, v114, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v114, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v114, v6, v114, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,5,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v116, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v115, v116, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v116, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v116, v6, v116, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v118, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v6, v118, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v117, v118, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v118, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v118, v6, v118, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v120, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v6, v120, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v119, v120, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v120, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v120, v6, v120, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v122, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v6, v122, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v121, v122, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v122, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v122, v6, v122, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,6,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v125, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v6, v125, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v123, v125, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v125, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v125, v6, v125, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v127, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v6, v127, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v126, v127, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v127, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v127, v6, v127, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v129, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v6, v129, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v128, v129, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v129, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v129, v6, v129, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v131, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v6, v131, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v130, v131, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v131, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v6, v131, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,7,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v133, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v6, v133, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v132, v133, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v133, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v133, v6, v133, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v135, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v6, v135, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v134, v135, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v135, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v135, v6, v135, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v137, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v6, v137, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v136, v137, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v137, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v6, v137, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v139, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v6, v139, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v138, v139, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v139, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v139, v6, v139, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,8,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v141, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v6, v141, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v140, v141, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v141, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v141, v6, v141, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v143, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v6, v143, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v142, v143, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v143, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v6, v143, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v145, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v6, v145, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v144, v145, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v145, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v145, v6, v145, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v147, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v6, v147, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v146, v147, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v147, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v6, v147, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,9,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v149, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v6, v149, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v148, v149, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v149, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v6, v149, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v151, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v6, v151, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v150, v151, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v151, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v6, v151, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v153, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v6, v153, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v152, v153, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v153, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v6, v153, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v155, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v6, v155, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v154, v155, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v155, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v6, v155, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,10,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v157, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v6, v157, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v156, v157, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v157, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v6, v157, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v159, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v6, v159, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v158, v159, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v159, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v6, v159, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v161, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v6, v161, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v160, v161, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v161, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v6, v161, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v163, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v6, v163, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v162, v163, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v163, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v6, v163, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,11,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v165, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v6, v165, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v164, v165, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v165, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v6, v165, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v167, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v6, v167, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v166, v167, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v167, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v6, v167, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v169, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v6, v169, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v168, v169, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v169, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v6, v169, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v171, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v171, v6, v171, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v170, v171, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v171, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v171, v6, v171, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,12,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v173, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v6, v173, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v172, v173, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v173, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v6, v173, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v175, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v6, v175, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v174, v175, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v175, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v175, v6, v175, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v177, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v6, v177, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v176, v177, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v177, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v177, v6, v177, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v179, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v6, v179, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v178, v179, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v179, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v6, v179, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,13,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v181, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v6, v181, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v180, v181, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v181, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v6, v181, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v183, v2, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v6, v183, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16 v182, v183, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v183, v3, v0, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v183, v6, v183, s[66:67]             // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v185, v2, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v6, v185, s[66:67]             // LDC clip if OOB. offset
buffer_load_short_d16_hi v184, v185, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v185, v3, v4, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v6, v185, s[66:67]             // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+14], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+15], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+16], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+17], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+18], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+19], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+20], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+21], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+22], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+23], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+24], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+25], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+26], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+27], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+28], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+29], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+30], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+31], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+32], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+33], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+34], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+35], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+36], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+37], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+38], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+39], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+40], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+41], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+42], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+43], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+44], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+45], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+46], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+47], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+48], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+49], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+50], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+51], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+52], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+53], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+54], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+55], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+56], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+57], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+58], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+59], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+60], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+61], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+62], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+63], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+64], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+65], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+66], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+67], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+68], acc39          // copy acc to vreg[57]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2), (0, 0, 1, 3), (0, 0, 2, 0), (0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 0), (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3), (0, 0, 4, 0), (0, 0, 4, 1), (0, 0, 4, 2), (0, 0, 4, 3), (0, 0, 5, 0), (0, 0, 5, 1), (0, 0, 5, 2), (0, 0, 5, 3), (0, 0, 6, 0), (0, 0, 6, 1), (0, 0, 6, 2), (0, 0, 6, 3), (0, 0, 7, 0), (0, 0, 7, 1), (0, 0, 7, 2), (0, 0, 7, 3), (0, 0, 8, 0), (0, 0, 8, 1), (0, 0, 8, 2), (0, 0, 8, 3), (0, 0, 9, 0), (0, 0, 9, 1), (0, 0, 9, 2), (0, 0, 9, 3), (0, 0, 10, 0), (0, 0, 10, 1), (0, 0, 10, 2), (0, 0, 10, 3), (0, 0, 11, 0), (0, 0, 11, 1), (0, 0, 11, 2), (0, 0, 11, 3), (0, 0, 12, 0), (0, 0, 12, 1), (0, 0, 12, 2), (0, 0, 12, 3), (0, 0, 13, 0), (0, 0, 13, 1), (0, 0, 13, 2), (0, 0, 13, 3), (0, 0, 14, 0), (0, 0, 14, 1)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
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
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+11], s[sgprBeta], v69, v[vgprValuC+11] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v70, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v71, v[vgprValuC+12] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v73, v[vgprValuC+13] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+14], s[sgprBeta], v75, v[vgprValuC+14] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v14, v[vgprValuC+14] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v14, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+15], s[sgprBeta], v77, v[vgprValuC+15] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v15, v[vgprValuC+15] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v15, v78, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v79, v[vgprValuC+16] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v16, v[vgprValuC+16] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v16, v80, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v81, v[vgprValuC+17] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v17, v[vgprValuC+17] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v17, v82, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v83, v[vgprValuC+18] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v18, v[vgprValuC+18] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v18, v84, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v85, v[vgprValuC+19] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v19, v[vgprValuC+19] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v19, v86, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+20], s[sgprBeta], v87, v[vgprValuC+20] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v20, v[vgprValuC+20] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v20, v88, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+21], s[sgprBeta], v89, v[vgprValuC+21] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v21, v[vgprValuC+21] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v21, v90, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+22], s[sgprBeta], v91, v[vgprValuC+22] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v22, v[vgprValuC+22] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v22, v92, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+23], s[sgprBeta], v93, v[vgprValuC+23] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v23, v[vgprValuC+23] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v23, v94, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+24], s[sgprBeta], v95, v[vgprValuC+24] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v24, v[vgprValuC+24] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v24, v96, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+25], s[sgprBeta], v97, v[vgprValuC+25] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v25, v[vgprValuC+25] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v25, v98, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+26], s[sgprBeta], v99, v[vgprValuC+26] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v26, v[vgprValuC+26] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v26, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+27], s[sgprBeta], v101, v[vgprValuC+27] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v27, v[vgprValuC+27] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v27, v102, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+28], s[sgprBeta], v103, v[vgprValuC+28] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v28, v[vgprValuC+28] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v28, v104, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+29], s[sgprBeta], v105, v[vgprValuC+29] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v29, v[vgprValuC+29] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v29, v106, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+30], s[sgprBeta], v107, v[vgprValuC+30] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v30, v[vgprValuC+30] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v30, v108, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+31], s[sgprBeta], v109, v[vgprValuC+31] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v31, v[vgprValuC+31] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v31, v110, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+32], s[sgprBeta], v111, v[vgprValuC+32] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v32, v[vgprValuC+32] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v32, v112, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+33], s[sgprBeta], v113, v[vgprValuC+33] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v33, v[vgprValuC+33] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v33, v114, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+34], s[sgprBeta], v115, v[vgprValuC+34] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v34, v[vgprValuC+34] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v34, v116, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+35], s[sgprBeta], v117, v[vgprValuC+35] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v35, v[vgprValuC+35] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v35, v118, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+36], s[sgprBeta], v119, v[vgprValuC+36] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v36, v[vgprValuC+36] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v36, v120, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+37], s[sgprBeta], v121, v[vgprValuC+37] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v37, v[vgprValuC+37] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v37, v122, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+38], s[sgprBeta], v123, v[vgprValuC+38] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v38, v[vgprValuC+38] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v38, v125, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+39], s[sgprBeta], v126, v[vgprValuC+39] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v39, v[vgprValuC+39] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v39, v127, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+40], s[sgprBeta], v128, v[vgprValuC+40] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v40, v[vgprValuC+40] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v40, v129, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+41], s[sgprBeta], v130, v[vgprValuC+41] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v41, v[vgprValuC+41] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v41, v131, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+42], s[sgprBeta], v132, v[vgprValuC+42] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v42, v[vgprValuC+42] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v42, v133, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+43], s[sgprBeta], v134, v[vgprValuC+43] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v43, v[vgprValuC+43] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v43, v135, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+44], s[sgprBeta], v136, v[vgprValuC+44] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v44, v[vgprValuC+44] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v44, v137, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+45], s[sgprBeta], v138, v[vgprValuC+45] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v45, v[vgprValuC+45] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v45, v139, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+46], s[sgprBeta], v140, v[vgprValuC+46] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v46, v[vgprValuC+46] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v46, v141, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+47], s[sgprBeta], v142, v[vgprValuC+47] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v47, v[vgprValuC+47] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v47, v143, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+48], s[sgprBeta], v144, v[vgprValuC+48] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v48, v[vgprValuC+48] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v48, v145, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+49], s[sgprBeta], v146, v[vgprValuC+49] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v49, v[vgprValuC+49] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v49, v147, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+50], s[sgprBeta], v148, v[vgprValuC+50] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v50, v[vgprValuC+50] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v50, v149, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+51], s[sgprBeta], v150, v[vgprValuC+51] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v51, v[vgprValuC+51] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v51, v151, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+52], s[sgprBeta], v152, v[vgprValuC+52] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v52, v[vgprValuC+52] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v52, v153, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+53], s[sgprBeta], v154, v[vgprValuC+53] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v53, v[vgprValuC+53] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v53, v155, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+54], s[sgprBeta], v156, v[vgprValuC+54] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v54, v[vgprValuC+54] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v54, v157, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+55], s[sgprBeta], v158, v[vgprValuC+55] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v55, v[vgprValuC+55] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v55, v159, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+56], s[sgprBeta], v160, v[vgprValuC+56] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v56, v[vgprValuC+56] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v56, v161, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+57], s[sgprBeta], v162, v[vgprValuC+57] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v57, v[vgprValuC+57] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v57, v163, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+58], s[sgprBeta], v164, v[vgprValuC+58] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v58, v[vgprValuC+58] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v58, v165, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+59], s[sgprBeta], v166, v[vgprValuC+59] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v59, v[vgprValuC+59] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v59, v167, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+60], s[sgprBeta], v168, v[vgprValuC+60] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v60, v[vgprValuC+60] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v60, v169, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+61], s[sgprBeta], v170, v[vgprValuC+61] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v61, v[vgprValuC+61] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v61, v171, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+62], s[sgprBeta], v172, v[vgprValuC+62] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v62, v[vgprValuC+62] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v62, v173, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+63], s[sgprBeta], v174, v[vgprValuC+63] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v63, v[vgprValuC+63] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v63, v175, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+64], s[sgprBeta], v176, v[vgprValuC+64] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v64, v[vgprValuC+64] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v64, v177, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+65], s[sgprBeta], v178, v[vgprValuC+65] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v65, v[vgprValuC+65] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v65, v179, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+66], s[sgprBeta], v180, v[vgprValuC+66] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v66, v[vgprValuC+66] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v66, v181, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+67], s[sgprBeta], v182, v[vgprValuC+67] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v67, v[vgprValuC+67] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v67, v183, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+68], s[sgprBeta], v184, v[vgprValuC+68] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v68, v[vgprValuC+68] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v68, v185, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,0,14,2:vw1); (0,0,14,3:vw1); (0,0,15,0:vw1); (0,0,15,1:vw1); (0,0,15,2:vw1); (0,0,15,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,14,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v18, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v18, v6, v18, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v17, v18, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v18, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v18, v6, v18, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,14,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v20, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v20, v6, v20, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v19, v20, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v20, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v20, v6, v20, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[62:63], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v22, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v22, v6, v22, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v21, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v22, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v22, v6, v22, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v24, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v24, v6, v24, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v23, v24, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v24, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v24, v6, v24, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v26, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v26, v6, v26, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16 v25, v26, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v26, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v26, v6, v26, s[66:67]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,15,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[62:63], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[66:67], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[66:67], s[62:63], s[66:67]             // in0 && in1
v_add_lshl_u32 v28, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v28, v6, v28, s[66:67]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v27, v28, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v28, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v28, v6, v28, s[66:67]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+12], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+13], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+14], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+15], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+16], acc63          // copy acc to vreg[63]

/* rC *= alpha batchElements=[(0, 0, 14, 2), (0, 0, 14, 3), (0, 0, 15, 0), (0, 0, 15, 1), (0, 0, 15, 2), (0, 0, 15, 3)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+14:vgprValuC+14+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+11], s[sgprBeta], v17, v[vgprValuC+11] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v18, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v19, v[vgprValuC+12] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v20, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v21, v[vgprValuC+13] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+14], s[sgprBeta], v23, v[vgprValuC+14] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v14, v[vgprValuC+14] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v14, v24, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+15], s[sgprBeta], v25, v[vgprValuC+15] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v15, v[vgprValuC+15] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v15, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v27, v[vgprValuC+16] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v16, v[vgprValuC+16] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v16, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_SK_Partials_1:
label_GW_Partials_E0:
s_mov_b64 s[sgprSrdWS:sgprSrdWS+1], s[sgprAddressWS:sgprAddressWS+1]
s_mov_b32 s[sgprSrdWS+2], BufferOOB
s_mov_b32 s[sgprSrdWS+3], Srd127_96

s_mul_i32 s8, 0x10000, s[sgprPersistentWorkGroupIndex] // Offset to correct partials tile (low word)
s_mul_hi_u32 s9, 0x10000, s[sgprPersistentWorkGroupIndex] // partials tile offset (high word) for 64-bit SRD
s_add_u32 s[sgprSrdWS+0], s[sgprSrdWS+0], s8       // add lo to SRD
s_addc_u32 s[sgprSrdWS+1], s[sgprSrdWS+1], s9      // add hi (offset high word + lo carry) to SRD

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=42 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partials Write Batch #0 (d1,d0,vc1,vc0) = */
/*      (0,0,0,0:vw4); (0,0,1,0:vw4); (0,0,2,0:vw4); (0,0,3,0:vw4); (0,0,4,0:vw4); (0,0,5,0:vw4); (0,0,6,0:vw4); (0,0,7,0:vw4); (0,0,8,0:vw4); (0,0,9,0:vw4); (0,0,10,0:vw4); (0,0,11,0:vw4); (0,0,12,0:vw4); (0,0,13,0:vw4); (0,0,14,0:vw4); (0,0,15,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc8           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc12          // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+20], acc16          // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+21], acc20          // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+22], acc24          // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+23], acc28          // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+24], acc32          // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+25], acc36          // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+26], acc40          // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+27], acc44          // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+28], acc48          // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+29], acc52          // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+30], acc56          // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+31], acc60          // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+32], acc1           // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+33], acc5           // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+34], acc9           // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+35], acc13          // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+36], acc17          // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+37], acc21          // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+38], acc25          // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+39], acc29          // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+40], acc33          // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+41], acc37          // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+42], acc41          // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+43], acc45          // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+44], acc49          // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+45], acc53          // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+46], acc57          // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+47], acc61          // copy acc to vreg[31]
v_accvgpr_read_b32 v[vgprValuC+48], acc2           // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+49], acc6           // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+50], acc10          // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+51], acc14          // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+52], acc18          // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+53], acc22          // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+54], acc26          // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+55], acc30          // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+56], acc34          // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+57], acc38          // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+58], acc42          // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+59], acc46          // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+60], acc50          // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+61], acc54          // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+62], acc58          // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+63], acc62          // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+64], acc3           // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+65], acc7           // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+66], acc11          // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+67], acc15          // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+68], acc19          // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+69], acc23          // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+70], acc27          // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+71], acc31          // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+72], acc35          // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+73], acc39          // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+74], acc43          // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+75], acc47          // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+76], acc51          // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+77], acc55          // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+78], acc59          // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+79], acc63          // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* apply mask, calc new C and issue writes */
v_lshlrev_b32 v11, 4, v[vgprSerial]                // v11 = v[vgprSerial] * 16
s_mov_b32 s8, 0                                    // Init sgpr offset
buffer_store_dwordx4 v[16:19], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[20:23], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[24:27], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[28:31], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[32:35], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[36:39], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[40:43], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[44:47], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[48:51], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[52:55], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[56:59], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[60:63], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[64:67], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[68:71], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[72:75], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_add_u32 s8, s8, 4096                             // Inc sgpr offset
buffer_store_dwordx4 v[76:79], v11, s[sgprSrdWS:sgprSrdWS+3], s8 offen offset:0 sc0 sc1 // addStore
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_waitcnt vmcnt(0)                                 // release: wait for partials stores before flag
s_barrier                                          // store all data before setting flag
s_lshl_b32 s8, s[sgprPersistentWorkGroupIndex], 2  // flag offset based on CTA index
v_readfirstlane_b32 s61, v[vgprSerial]             // Wave 0 updates flags
s_cmp_eq_u32 s61, 0                                // Check for wave 0
s_cbranch_scc0 label_SK_SkipFlagSet                // Skip flag set
s_mov_b32 s61, 1                                   // flag data
v_mov_b32 v13, s61                                 // move flag value to vgpr
v_mov_b32 v14, 0                                   // zero vaddr offset
s_mov_b64 s[64:65], s[sgprAddressFlags:sgprAddressFlags+1]
s_mov_b32 s66, BufferOOB
s_mov_b32 s67, Srd127_96
buffer_store_dword v13, v14, s[64:67], s8 offen offset:0 sc0 sc1 // set flag
s_waitcnt vmcnt(0)                                 // release: wait for partials stores before flag
label_SK_SkipFlagSet:
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_PersistentLoopClose:
s_cmp_ge_u32 s[sgprPersistentIteration], s[sgprPersistentIterationEnd] // Check whether assigned work is exhausted
s_cbranch_scc1 label_NoBranch_16                   // Only branch on scc0
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s64, label_PersistentLoopStart, 4        // target branch offset
s_abs_i32 s64, s64                                 // abs offset
s_sub_u32 s62, s62, s64                            // sub target branch offset
s_subb_u32 s63, s63, 0                             // sub high and carry
s_setpc_b64 s[62:63]                               // branch to label_PersistentLoopStart
label_NoBranch_16:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
