
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.protected Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMBSK_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_PGR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCC1
.globl Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMBSK_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_PGR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCC1
.p2align 8
.type Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMBSK_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_PGR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCC1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMBSK_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_PGR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCC1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 88 // accvgpr offset
  .amdhsa_next_free_vgpr 92 // vgprs
  .amdhsa_next_free_sgpr 68 // sgprs
  .amdhsa_group_segment_fixed_size 2560 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
/* Num VGPR   =82 */
/* Num AccVGPR=4 */
/* Num SGPR   =68 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 4 x 1 */
/* SubGroup= 4 x 16 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=4 */
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
  - .name: Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMBSK_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_PGR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCC1
    .symbol: 'Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMBSK_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_PGR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCC1.kd'
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
      - .name:            dstD
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            Synchronizer
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            GSUSync
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            batchOffsetD
        .size:            8
        .offset:          124
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          132
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          140
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          148
        .value_kind:      by_value
        .value_type:      u64
    .group_segment_fixed_size:   2560
    .kernarg_segment_align:      8
    .kernarg_segment_size:       160
    .max_flat_workgroup_size:    64
    .private_segment_fixed_size: 0
    .sgpr_count:                 68
    .sgpr_spill_count:           0
    .vgpr_count:                 82
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_HHS_BH_MT16x16x16_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMBSK_GLS0_HPLR0_ISA950_ICIW0_IU1_K1_LDSSI0_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV0_MIWT1_1_MXLIBL_MXSFNS_MO64_MGRIPM3_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL0_PAP0_PGL0_PGR0_PLR0_PKA0_RAP0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW4_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS0_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_1_WGMXCC1:
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
.set vgprSerial, 14

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuA_X0_I0_D0_PACK, vgprBase+2
.set vgprValuB_X0_I0_BASE, vgprBase+4
.set vgprValuB_X0_I0_D0_PACK, vgprBase+6
.set vgprG2LA_BASE, vgprBase+0
.set vgprG2LB_BASE, vgprBase+4
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
.set sgprAddressA, 20
.set sgprAddressB, 22
.set sgprStridesA, 24
.set sgprStridesB, 26
.set sgprAlpha, 28
.set sgprBeta, 29
.set sgprAddressD, 30
.set sgprAddressC, 32
.set sgprStridesD, 34
.set sgprStridesC, 36
.set sgprAddressTD, 38
.set sgprSynchronizer, 40
.set sgprGSUSync, 42
.set sgprGSU, 43

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
/* MT offset for 64b address (=MT0*MT1*bpeC) */
.set MTOffset, 0x400
.set MTOffsetH32, 0x0

/* Global Offset A */

/* Global Offset B */

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_dword s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_dword s46, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_dword s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_dword s47, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s45, s44, 0x1e                          // Get arg type
s_and_b32 s44, 0x3fffffff, s44                     // Get nums of gemm
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s45, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s45, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[16:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_dwordx4 s[32:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_dwordx2 s[36:37], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_and_b32 s[sgprStaggerU], s46, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s46, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 m0, 0xa00                                // LDS clamp at 2560 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* remap workgroup to XCCs */
s_lshr_b32 s52, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s52, s52                             // Get log(WGMXCC)
s_lshr_b32 s53, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s52, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s49, s47, s52
s_lshl_b32 s49, s49, s52
s_cmp_ge_u32 s[sgprWorkGroup0], s49
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s53, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s49, s[sgprWorkGroup0], s52
s_bfm_b32 s50, s52, 0
s_and_b32 s50, s[sgprWorkGroup0], s50
s_lshr_b32 s51, s47, s52
s_mul_i32 s50, s50, s51
s_add_u32 s[sgprWorkGroup0], s49, s50
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[16:17], s53                        // s49 = s[sgprWorkGroup0] / s53
v_rcp_f64 v[16:17], v[16:17]                       // s49 = s[sgprWorkGroup0] / s53
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup0]          // s49 = s[sgprWorkGroup0] / s53
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s49 = s[sgprWorkGroup0] / s53
v_cvt_u32_f64 v16, v[16:17]                        // s49 = s[sgprWorkGroup0] / s53
v_mul_lo_u32 v17, v16, s53                         // s49 = s[sgprWorkGroup0] / s53
v_sub_u32 v18, s[sgprWorkGroup0], v17              // s49 = s[sgprWorkGroup0] / s53
v_cmpx_ge_u32 exec, v18, s53                       // s49 = s[sgprWorkGroup0] / s53
v_add_u32 v16, v16, 1                              // s49 = s[sgprWorkGroup0] / s53
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v17, v16, s53                         // s49 = s[sgprWorkGroup0] / s53
v_sub_u32 v18, s[sgprWorkGroup0], v17              // s49 = s[sgprWorkGroup0] / s53
v_readfirstlane_b32 s49, v16                       // quotient
v_readfirstlane_b32 s50, v18                       // remainder
s_mul_i32 s49, s49, s53
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s50, s50, s52
/* temp0 = temp0 + temp1 */
s_add_u32 s49, s49, s50
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[16:17], s53                        // s50 = s47 / s53
v_rcp_f64 v[16:17], v[16:17]                       // s50 = s47 / s53
v_cvt_f64_u32 v[18:19], s47                        // s50 = s47 / s53
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s50 = s47 / s53
v_cvt_u32_f64 v16, v[16:17]                        // s50 = s47 / s53
v_mul_lo_u32 v17, v16, s53                         // s50 = s47 / s53
v_sub_u32 v18, s47, v17                            // s50 = s47 / s53
v_cmpx_ge_u32 exec, v18, s53                       // s50 = s47 / s53
v_add_u32 v16, v16, 1                              // s50 = s47 / s53
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s50, v16                       // quotient
s_mul_i32 s50, s50, s53
s_sub_u32 s51, s47, s50
s_cmp_gt_u32 s[sgprWorkGroup0], s50
s_cselect_b32 s50, s51, s53
s_lshr_b32 s50, s50, s52
s_bfm_b32 s51, s52, 0
s_and_b32 s51, s[sgprWorkGroup0], s51
s_mul_i32 s50, s50, s51
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s49, s50
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s45, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s45, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [6...20) to pool */
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
v_lshl_add_u32 v6, v7, 6, v6                       // 5. K offset: lrKOffset = kIdx * mStride(64); 6. offset in wave: lrOffset = bnOffset + lrKOffset
/* lr1J */
v_and_b32 v8, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v7, 15, v8                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v8, 4, v8                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v7, v8, 6, v7                       // 5. K offset: lrKOffset = kIdx * mStride(64); 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v8, 6, v[vgprSerial]                 // 8 = Serial / 64
v_lshrrev_b32 v8, 0, v8                            // LSU offset: Get LSU wave_id
s_mov_b32 s46, 256                                 // LSU offset: stride = lsuStride(16)*(MT0(16) + PAD0(0))
v_mul_lo_u32 v8, s46, v8                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v8, v6            // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v9, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v9, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v6, 6, v[vgprSerial]                 // 6 = Serial / 64
v_lshrrev_b32 v6, 0, v6                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(16)*(MT1(16) + PAD1(0)) (dup assign opt.)
v_mul_lo_u32 v6, s46, v6                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v6, v7            // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v8, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v8, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x280, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 4 */
/* v7 = A-unroll = serial/LVCA */
v_lshrrev_b32 v7, 2, v[vgprSerial]                 // 7 = Serial / 4
v_and_b32 v6, 3, v[vgprSerial]                     // 6 = Serial % 4
/* tile *= glvw */
v_lshlrev_b32 v6, 2, v6                            // v6 = v6 * 4
v_mov_b32 v10, v7                                  // copy for GlobalSplitU
/* LVCB = 4 */
/* v9 = B-unroll = serial%LVCB */
v_lshrrev_b32 v8, 2, v[vgprSerial]                 // 8 = Serial / 4
v_and_b32 v9, 3, v[vgprSerial]                     // 9 = Serial % 4
/* unroll *= glvw */
v_lshlrev_b32 v9, 2, v9                            // v9 = v9 * 4
v_mov_b32 v11, v9                                  // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v10 */
/* lwaUnrollAssignmentB = v11 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v10    // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v12, 7, v[vgprLocalWriteAddrA]       // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrA], v12, 5, v[vgprLocalWriteAddrA] // padding 32 per block 128

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v11    // lwBL**(MTB + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v8, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(MT1J+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v12, 7, v[vgprLocalWriteAddrB]       // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v12, 5, v[vgprLocalWriteAddrB] // padding 32 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x280, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=640
s_waitcnt lgkmcnt(0)                               // wait for 88 bytes of kern args
v_mov_b32 v17, MT0                                 // set MT0 into sgpr
v_mov_b32 v16, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v15, v17                             // v15 = ceil(v16 / v17)
v_rcp_iflag_f32 v15, v15                           // v15 = ceil(v16 / v17)
v_cvt_f32_u32 v18, v16                             // v15 = ceil(v16 / v17)
v_mul_f32 v15, v15, v18                            // v15 = ceil(v16 / v17)
v_cvt_u32_f32 v15, v15                             // v15 = ceil(v16 / v17)
v_mul_u32_u24 v18, v15, v17                        // v15 = ceil(v16 / v17)
v_sub_u32 v18, v16, v18                            // v15 = ceil(v16 / v17)
v_cmp_ne_u32 vcc, v18, 0                           // v15 = ceil(v16 / v17)
v_addc_co_u32 v15, vcc, v15, 0, vcc                // ceil
v_mov_b32 v17, MT1                                 // set MT1 into sgpr
v_mov_b32 v16, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v15     // set back to numWorkGroup0
v_cvt_f32_u32 v15, v17                             // v15 = ceil(v16 / v17)
v_rcp_iflag_f32 v15, v15                           // v15 = ceil(v16 / v17)
v_cvt_f32_u32 v18, v16                             // v15 = ceil(v16 / v17)
v_mul_f32 v15, v15, v18                            // v15 = ceil(v16 / v17)
v_cvt_u32_f32 v15, v15                             // v15 = ceil(v16 / v17)
v_mul_u32_u24 v18, v15, v17                        // v15 = ceil(v16 / v17)
v_sub_u32 v18, v16, v18                            // v15 = ceil(v16 / v17)
v_cmp_ne_u32 vcc, v18, 0                           // v15 = ceil(v16 / v17)
v_addc_co_u32 v15, vcc, v15, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v15     // set back to numWorkGroup1

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s46, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s47, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s46, s46, s47
v_cvt_f32_u32 v12, s46                             // s46 = s[sgprWorkGroup0] / s46
v_rcp_iflag_f32 v12, v12                           // s46 = s[sgprWorkGroup0] / s46
v_cvt_f32_u32 v13, s[sgprWorkGroup0]               // s46 = s[sgprWorkGroup0] / s46
v_mul_f32 v12, v12, v13                            // s46 = s[sgprWorkGroup0] / s46
v_cvt_u32_f32 v12, v12                             // s46 = s[sgprWorkGroup0] / s46
v_mul_u32_u24 v13, v12, s46                        // s46 = s[sgprWorkGroup0] / s46
v_sub_u32 v13, s[sgprWorkGroup0], v13              // s46 = s[sgprWorkGroup0] / s46
v_cmpx_eq_u32 exec, v13, s46                       // s46 = s[sgprWorkGroup0] / s46
v_add_u32 v12, 1, v12                              // s46 = s[sgprWorkGroup0] / s46
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s46                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s46, v12                       // quotient
s_mov_b32 s[sgprWorkGroup2], s46
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s46, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s46, s46, s[sgprWorkGroup2]
s_mul_i32 s46, s46, s47
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s46
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v12, s[sgprNumWorkGroups0]           // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v12, v12                           // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v13, s[sgprWorkGroup0]               // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v12, v12, v13                            // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v12, v12                             // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v13, v12, s[sgprNumWorkGroups0]      // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v13, s[sgprWorkGroup0], v13              // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v13, s[sgprNumWorkGroups0]     // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v12, 1, v12                              // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s46, v12                       // quotient
s_mov_b32 s[sgprWorkGroup1], s46
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s46, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s46
s_branch label_MultiGemmEnd
label_MultiGemm:
s_mov_b32 s15, 108                                 // KernArgAddressOffset
s_mul_i32 s52, s44, 4
s_mov_b64 s[46:47], s[sgprKernArgAddress:sgprKernArgAddress+1]

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s53, 0
s_load_dwordx4 s[16:19], s[46:47], s52
s_cmpk_eq_u32 s44, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s50, s16, 4                             // s50 = s16 / 16
s_and_b32 s48, 15, s16                             // s48 = s16 % 16
s_addc_u32 s50, s50, 0
s_lshr_b32 s51, s17, 4                             // s51 = s17 / 16
s_and_b32 s48, 15, s17                             // s48 = s17 % 16
s_addc_u32 s51, s51, 0
s_mul_i32 s50, s50, s51
s_mul_i32 s50, s50, s18
s_and_b32 s51, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s50, s50, s51
s_add_u32 s53, s53, s50
s_cmp_lt_u32 s[sgprWorkGroup0], s53
s_cbranch_scc1 label_FOUND
/* Check if custom structure pointer is null */
s_add_u32 s52, s52, s15
s_load_dwordx4 s[16:19], s[46:47], s52
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s44
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s50, s16, 4                             // s50 = s16 / 16
s_and_b32 s48, 15, s16                             // s48 = s16 % 16
s_addc_u32 s50, s50, 0
s_lshr_b32 s51, s17, 4                             // s51 = s17 / 16
s_and_b32 s48, 15, s17                             // s48 = s17 % 16
s_addc_u32 s51, s51, 0
s_mul_i32 s50, s50, s51
s_mul_i32 s50, s50, s18
s_and_b32 s46, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s50, s50, s46
s_add_u32 s53, s53, s50

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s47, s14, 1
s_sub_u32 s46, s53, s50
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s46

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s44, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s47, s47, 108                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s47
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_dwordx16 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_dwordx2 s[36:37], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
/* init: add vgpr [6...20) to pool */
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
v_lshl_add_u32 v6, v7, 6, v6                       // 5. K offset: lrKOffset = kIdx * mStride(64); 6. offset in wave: lrOffset = bnOffset + lrKOffset
/* lr1J */
v_and_b32 v8, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v7, 15, v8                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v8, 4, v8                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v7, v8, 6, v7                       // 5. K offset: lrKOffset = kIdx * mStride(64); 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v8, 6, v[vgprSerial]                 // 8 = Serial / 64
v_lshrrev_b32 v8, 0, v8                            // LSU offset: Get LSU wave_id
s_mov_b32 s46, 256                                 // LSU offset: stride = lsuStride(16)*(MT0(16) + PAD0(0))
v_mul_lo_u32 v8, s46, v8                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v8, v6            // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)
v_lshrrev_b32 v9, 7, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrA], v9, 5, v[vgprLocalReadAddrA] // Final Offset: padding 32 per block 128

/* local read addresses: final offsets b */
v_lshrrev_b32 v6, 6, v[vgprSerial]                 // 6 = Serial / 64
v_lshrrev_b32 v6, 0, v6                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(16)*(MT1(16) + PAD1(0)) (dup assign opt.)
v_mul_lo_u32 v6, s46, v6                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v6, v7            // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)
v_lshrrev_b32 v8, 7, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 128
v_lshl_add_u32 v[vgprLocalReadAddrB], v8, 5, v[vgprLocalReadAddrB] // Final Offset: padding 32 per block 128

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x280, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 4 */
/* v7 = A-unroll = serial/LVCA */
v_lshrrev_b32 v7, 2, v[vgprSerial]                 // 7 = Serial / 4
v_and_b32 v6, 3, v[vgprSerial]                     // 6 = Serial % 4
/* tile *= glvw */
v_lshlrev_b32 v6, 2, v6                            // v6 = v6 * 4
v_mov_b32 v10, v7                                  // copy for GlobalSplitU
/* LVCB = 4 */
/* v9 = B-unroll = serial%LVCB */
v_lshrrev_b32 v8, 2, v[vgprSerial]                 // 8 = Serial / 4
v_and_b32 v9, 3, v[vgprSerial]                     // 9 = Serial % 4
/* unroll *= glvw */
v_lshlrev_b32 v9, 2, v9                            // v9 = v9 * 4
v_mov_b32 v11, v9                                  // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v10 */
/* lwaUnrollAssignmentB = v11 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v10    // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA] //  (multiple bpe)
v_lshrrev_b32 v12, 7, v[vgprLocalWriteAddrA]       // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrA], v12, 5, v[vgprLocalWriteAddrA] // padding 32 per block 128

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x10, v11    // lwBL**(MTB + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v8, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(MT1J+PAD))
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB] //  (multiple bpe)
v_lshrrev_b32 v12, 7, v[vgprLocalWriteAddrB]       // padding 32 per block 128
v_lshl_add_u32 v[vgprLocalWriteAddrB], v12, 5, v[vgprLocalWriteAddrB] // padding 32 per block 128
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x280, v[vgprLocalWriteAddrB] // lwFOB = lw1J + lwL*MT1J + LDS_OFFSET_B=640
s_waitcnt lgkmcnt(0)                               // wait for 88 bytes of kern args
v_mov_b32 v17, MT0                                 // set MT0 into sgpr
v_mov_b32 v16, s[sgprSizesFree+0]                  // set Free0 size
v_cvt_f32_u32 v15, v17                             // v15 = ceil(v16 / v17)
v_rcp_iflag_f32 v15, v15                           // v15 = ceil(v16 / v17)
v_cvt_f32_u32 v18, v16                             // v15 = ceil(v16 / v17)
v_mul_f32 v15, v15, v18                            // v15 = ceil(v16 / v17)
v_cvt_u32_f32 v15, v15                             // v15 = ceil(v16 / v17)
v_mul_u32_u24 v18, v15, v17                        // v15 = ceil(v16 / v17)
v_sub_u32 v18, v16, v18                            // v15 = ceil(v16 / v17)
v_cmp_ne_u32 vcc, v18, 0                           // v15 = ceil(v16 / v17)
v_addc_co_u32 v15, vcc, v15, 0, vcc                // ceil
v_mov_b32 v17, MT1                                 // set MT1 into sgpr
v_mov_b32 v16, s[sgprSizesFree+1]                  // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v15     // set back to numWorkGroup0
v_cvt_f32_u32 v15, v17                             // v15 = ceil(v16 / v17)
v_rcp_iflag_f32 v15, v15                           // v15 = ceil(v16 / v17)
v_cvt_f32_u32 v18, v16                             // v15 = ceil(v16 / v17)
v_mul_f32 v15, v15, v18                            // v15 = ceil(v16 / v17)
v_cvt_u32_f32 v15, v15                             // v15 = ceil(v16 / v17)
v_mul_u32_u24 v18, v15, v17                        // v15 = ceil(v16 / v17)
v_sub_u32 v18, v16, v18                            // v15 = ceil(v16 / v17)
v_cmp_ne_u32 vcc, v18, 0                           // v15 = ceil(v16 / v17)
v_addc_co_u32 v15, vcc, v15, 0, vcc                // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v15     // set back to numWorkGroup1

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s46, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s47, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s46, s46, s47
v_cvt_f32_u32 v12, s46                             // s46 = s[sgprWorkGroup0] / s46
v_rcp_iflag_f32 v12, v12                           // s46 = s[sgprWorkGroup0] / s46
v_cvt_f32_u32 v13, s[sgprWorkGroup0]               // s46 = s[sgprWorkGroup0] / s46
v_mul_f32 v12, v12, v13                            // s46 = s[sgprWorkGroup0] / s46
v_cvt_u32_f32 v12, v12                             // s46 = s[sgprWorkGroup0] / s46
v_mul_u32_u24 v13, v12, s46                        // s46 = s[sgprWorkGroup0] / s46
v_sub_u32 v13, s[sgprWorkGroup0], v13              // s46 = s[sgprWorkGroup0] / s46
v_cmpx_eq_u32 exec, v13, s46                       // s46 = s[sgprWorkGroup0] / s46
v_add_u32 v12, 1, v12                              // s46 = s[sgprWorkGroup0] / s46
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s46                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s46, v12                       // quotient
s_mov_b32 s[sgprWorkGroup2], s46
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s46, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s46, s46, s[sgprWorkGroup2]
s_mul_i32 s46, s46, s47
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s46
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v12, s[sgprNumWorkGroups0]           // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v12, v12                           // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v13, s[sgprWorkGroup0]               // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v12, v12, v13                            // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v12, v12                             // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v13, v12, s[sgprNumWorkGroups0]      // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v13, s[sgprWorkGroup0], v13              // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v13, s[sgprNumWorkGroups0]     // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v12, 1, v12                              // s46 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s[sgprNumWorkGroups0]     // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s46, v12                       // quotient
s_mov_b32 s[sgprWorkGroup1], s46
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s46, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s46

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprSrdA, 44
.set sgprSrdB, 48
.set sgprShadowLimitA, 52
.set sgprShadowLimitB, 54
.set sgprStaggerUIter, 56
.set sgprWrapUA, 57
.set sgprWrapUB, 59
.set sgprGlobalReadIncsA, 61
.set sgprGlobalReadIncsB, 62
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 8  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 8  // pre-pad to make room for possible pointer shift
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
s_and_b32 s44, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s44, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
/* Check if custom structure pointer is null */
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s44, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s44, s[sgprGSU], 0xfff                   // Restore GSU
v_cvt_f32_u32 v12, s44                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_rcp_iflag_f32 v12, v12                           // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_cvt_f32_u32 v13, s[sgprWorkGroup1]               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_mul_f32 v12, v12, v13                            // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_cvt_u32_f32 v12, v12                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_mul_u32_u24 v13, v12, s44                        // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_sub_u32 v13, s[sgprWorkGroup1], v13              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_cmpx_eq_u32 exec, v13, s44                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_add_u32 v12, 1, v12                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s44
v_mov_b32 v13, 0                                   // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s44
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v13, s44                       // overflow happened in remainder
v_sub_u32 v12, v12, 1                              // quotient - 1
v_mul_u32_u24 v13, v12, s44                        // re-calculate remainder
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
s_mov_b32 s44, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s44, s44                            // Restore WGM
s_cmp_gt_i32 s44, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s44, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s44, s44                                 // abs(WGM)
v_cvt_f64_u32 v[16:17], s44                        // s45 = s[sgprWorkGroup0] / s44
v_rcp_f64 v[16:17], v[16:17]                       // s45 = s[sgprWorkGroup0] / s44
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup0]          // s45 = s[sgprWorkGroup0] / s44
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s45 = s[sgprWorkGroup0] / s44
v_cvt_u32_f64 v16, v[16:17]                        // s45 = s[sgprWorkGroup0] / s44
v_mul_lo_u32 v17, v16, s44                         // s45 = s[sgprWorkGroup0] / s44
v_sub_u32 v18, s[sgprWorkGroup0], v17              // s45 = s[sgprWorkGroup0] / s44
v_cmpx_ge_u32 exec, v18, s44                       // s45 = s[sgprWorkGroup0] / s44
v_add_u32 v16, v16, 1                              // s45 = s[sgprWorkGroup0] / s44
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s45, v16                       // quotient
s_mul_i32 s48, s45, s44                            // quotient * non-magic divisor
s_sub_u32 s48, s[sgprWorkGroup0], s48              // WorkGroup0=remainder
s_mul_i32 s48, s48, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s48, s48, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[16:17], s44                        // s46 = s[sgprNumWorkGroups0] / s44
v_rcp_f64 v[16:17], v[16:17]                       // s46 = s[sgprNumWorkGroups0] / s44
v_cvt_f64_u32 v[18:19], s[sgprNumWorkGroups0]      // s46 = s[sgprNumWorkGroups0] / s44
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s46 = s[sgprNumWorkGroups0] / s44
v_cvt_u32_f64 v16, v[16:17]                        // s46 = s[sgprNumWorkGroups0] / s44
v_mul_lo_u32 v17, v16, s44                         // s46 = s[sgprNumWorkGroups0] / s44
v_sub_u32 v18, s[sgprNumWorkGroups0], v17          // s46 = s[sgprNumWorkGroups0] / s44
v_cmpx_ge_u32 exec, v18, s44                       // s46 = s[sgprNumWorkGroups0] / s44
v_add_u32 v16, v16, 1                              // s46 = s[sgprNumWorkGroups0] / s44
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s46, v16                       // quotient
s_mul_i32 s47, s44, s46                            // quotient * non-magic divisor
s_sub_u32 s47, s[sgprNumWorkGroups0], s47          // NumWorkGroups0=remainder
s_cmp_eq_u32 s47, 0                                // remainder == 0 ?
s_cmov_b32 s47, s44                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s45, s46                              // blockId >= numFullBlocks ?
s_cselect_b32 s46, s47, s44
v_cvt_f64_u32 v[16:17], s46                        // s[sgprWorkGroup1] = s48 / s46
v_rcp_f64 v[16:17], v[16:17]                       // s[sgprWorkGroup1] = s48 / s46
v_cvt_f64_u32 v[18:19], s48                        // s[sgprWorkGroup1] = s48 / s46
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s[sgprWorkGroup1] = s48 / s46
v_cvt_u32_f64 v16, v[16:17]                        // s[sgprWorkGroup1] = s48 / s46
v_mul_lo_u32 v17, v16, s46                         // s[sgprWorkGroup1] = s48 / s46
v_sub_u32 v18, s48, v17                            // s[sgprWorkGroup1] = s48 / s46
v_cmpx_ge_u32 exec, v18, s46                       // s[sgprWorkGroup1] = s48 / s46
v_add_u32 v16, v16, 1                              // s[sgprWorkGroup1] = s48 / s46
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v17, v16, s46                         // s[sgprWorkGroup1] = s48 / s46
v_sub_u32 v18, s48, v17                            // s[sgprWorkGroup1] = s48 / s46
v_readfirstlane_b32 s[sgprWorkGroup1], v16         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v18         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s46 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s48, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s45, s45, s44                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s45 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s44, s44                                 // WGM
v_cvt_f64_u32 v[16:17], s44                        // s45 = s[sgprWorkGroup1] / s44
v_rcp_f64 v[16:17], v[16:17]                       // s45 = s[sgprWorkGroup1] / s44
v_cvt_f64_u32 v[18:19], s[sgprWorkGroup1]          // s45 = s[sgprWorkGroup1] / s44
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s45 = s[sgprWorkGroup1] / s44
v_cvt_u32_f64 v16, v[16:17]                        // s45 = s[sgprWorkGroup1] / s44
v_mul_lo_u32 v17, v16, s44                         // s45 = s[sgprWorkGroup1] / s44
v_sub_u32 v18, s[sgprWorkGroup1], v17              // s45 = s[sgprWorkGroup1] / s44
v_cmpx_ge_u32 exec, v18, s44                       // s45 = s[sgprWorkGroup1] / s44
v_add_u32 v16, v16, 1                              // s45 = s[sgprWorkGroup1] / s44
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s45, v16                       // quotient
s_mul_i32 s48, s45, s44                            // quotient * non-magic divisor
s_sub_u32 s48, s[sgprWorkGroup1], s48              // WorkGroup1=remainder
s_mul_i32 s48, s48, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s48, s48, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[16:17], s44                        // s46 = s[sgprNumWorkGroups1] / s44
v_rcp_f64 v[16:17], v[16:17]                       // s46 = s[sgprNumWorkGroups1] / s44
v_cvt_f64_u32 v[18:19], s[sgprNumWorkGroups1]      // s46 = s[sgprNumWorkGroups1] / s44
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s46 = s[sgprNumWorkGroups1] / s44
v_cvt_u32_f64 v16, v[16:17]                        // s46 = s[sgprNumWorkGroups1] / s44
v_mul_lo_u32 v17, v16, s44                         // s46 = s[sgprNumWorkGroups1] / s44
v_sub_u32 v18, s[sgprNumWorkGroups1], v17          // s46 = s[sgprNumWorkGroups1] / s44
v_cmpx_ge_u32 exec, v18, s44                       // s46 = s[sgprNumWorkGroups1] / s44
v_add_u32 v16, v16, 1                              // s46 = s[sgprNumWorkGroups1] / s44
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s46, v16                       // quotient
s_mul_i32 s47, s44, s46                            // quotient * non-magic divisor
s_sub_u32 s47, s[sgprNumWorkGroups1], s47          // NumWorkGroups1=remainder
s_cmp_eq_u32 s47, 0                                // remainder == 0 ?
s_cmov_b32 s47, s44                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s45, s46                              // blockId >= numFullBlocks ?
s_cselect_b32 s46, s47, s44
v_cvt_f64_u32 v[16:17], s46                        // s[sgprWorkGroup0] = s48 / s46
v_rcp_f64 v[16:17], v[16:17]                       // s[sgprWorkGroup0] = s48 / s46
v_cvt_f64_u32 v[18:19], s48                        // s[sgprWorkGroup0] = s48 / s46
v_mul_f64 v[16:17], v[16:17], v[18:19]             // s[sgprWorkGroup0] = s48 / s46
v_cvt_u32_f64 v16, v[16:17]                        // s[sgprWorkGroup0] = s48 / s46
v_mul_lo_u32 v17, v16, s46                         // s[sgprWorkGroup0] = s48 / s46
v_sub_u32 v18, s48, v17                            // s[sgprWorkGroup0] = s48 / s46
v_cmpx_ge_u32 exec, v18, s46                       // s[sgprWorkGroup0] = s48 / s46
v_add_u32 v16, v16, 1                              // s[sgprWorkGroup0] = s48 / s46
s_mov_b64 exec, -1                                 // Reset exec
v_mul_lo_u32 v17, v16, s46                         // s[sgprWorkGroup0] = s48 / s46
v_sub_u32 v18, s48, v17                            // s[sgprWorkGroup0] = s48 / s46
v_readfirstlane_b32 s[sgprWorkGroup0], v16         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v18         // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s46 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s48, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s45, s45, s44                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s45 // wg1 += blockId * WGM
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
v_mov_b32 v15, v7                                  // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v16, v9                                  // groBL_0

/* global read addresses: shift a */
s_mul_i32 s61, s[sgprWorkGroup0], 16               // WorkGroup[01] * MT
s_sub_u32 s61, s[sgprSizeI], s61                   // edge = Size0I - WG*MT
s_sub_u32 s61, s61, 4                              // edge -= margin(4)
v_mov_b32 v17, s61                                 // edge vgpr = Size0I- WG*MT - margin(4)
v_min_i32 v12, v17, v12                            // offset = (offset < edge) ? offset(v12) : edge(v17)

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s65, s[sgprWorkGroup0], 16            // WorkGroup[01] * MT
s_mul_i32 s64, s[sgprWorkGroup0], 16               // WorkGroup[01] * MT
s_and_b32 s62, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s63, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s62, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 4 // s[LoopCounterL] = s[sgprSizesSum] / 16
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v17, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v17, v17                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v18, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v17, v17, v18                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v17, v17                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v18, v17, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v18, s[sgprLoopCounterL], v18            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v18, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v17, 1, v17                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v18, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v18, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v17, v17, 1                              // quotient - 1
v_mul_u32_u24 v18, v17, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v18, s[sgprLoopCounterL], v18            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v17       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v18        // remainder
s_mul_i32 s63, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s62, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s63, s63, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s62, s62, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s62, s62, s63                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s63, s62, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s62, s62, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_mul_hi_u32 s63, s62, s[sgprStrideAL]             // tlu=1, scaled unroll-offset by stride
s_mul_i32 s62, s62, s[sgprStrideAL]                // tlu=1, scaled unroll-offset by stride
s_add_u32 s64, s64, s62                            // accum GsuOffset term to tilestart
s_addc_u32 s65, s65, s63                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1 // Init tensor size
s_sub_u32 s62, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s63, constStrideA0I, s62              // stride x (size-1)
s_mul_i32 s62, constStrideA0I, s62                 // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s62 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s63 // sum tensor size
s_sub_u32 s62, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s63, s[sgprStrideAL], s62             // stride x (size-1)
s_mul_i32 s62, s[sgprStrideAL], s62                // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s62 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s63 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s64 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s65 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_branch label_StridedBatchedGemmLoadA
s_mul_i32 s62, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s62, s62, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s63, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdA:sgprSrdA+1], s[62:63], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_dwordx2 s[62:63], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x7c // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s62        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s63       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 8          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[64:65], s[64:65], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s64, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s65, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s63, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s62, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s64, s64, s62                            // accum wg term to tilestart
s_addc_u32 s65, s65, s63                           // accum wg term to tilestart
s_lshl_b64 s[64:65], s[64:65], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s64    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s65   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s65, s[sgprWorkGroup1], 16            // WorkGroup[01] * MT
s_mul_i32 s64, s[sgprWorkGroup1], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s65, s64, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s64, s64, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s62, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s63, 16, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s62, 16, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 4 // s[LoopCounterL] = s[sgprSizesSum] / 16
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v17, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v17, v17                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v18, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v17, v17, v18                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v17, v17                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v18, v17, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v18, s[sgprLoopCounterL], v18            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v18, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v17, 1, v17                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v18, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v18, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v17, v17, 1                              // quotient - 1
v_mul_u32_u24 v18, v17, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v18, s[sgprLoopCounterL], v18            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v17       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v18        // remainder
s_mul_i32 s63, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s62, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s63, s63, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s62, s62, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s62, s62, s63                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s63, s62, 16                          // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s62, s62, 16                             // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_B_End:
s_add_u32 s64, s64, s62                            // accum GsuOffset term to tilestart
s_addc_u32 s65, s65, s63                           // accum GsuOffset term to tilestart
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1 // Init tensor size
s_sub_u32 s62, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s63, constStrideBL, s62               // stride x (size-1)
s_mul_i32 s62, constStrideBL, s62                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s62 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s63 // sum tensor size
s_sub_u32 s62, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s63, s[sgprStrideB1J], s62            // stride x (size-1)
s_mul_i32 s62, s[sgprStrideB1J], s62               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s62 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s63 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s64 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s65 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_branch label_StridedBatchedGemmLoadB
s_mul_i32 s62, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s62, s62, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s63, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdB:sgprSrdB+1], s[62:63], 0 // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_dwordx2 s[62:63], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x84 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s62        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s63       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 8          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[64:65], s[64:65], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s64, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s65, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s63, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s62, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s64, s64, s62                            // accum wg term to tilestart
s_addc_u32 s65, s65, s63                           // accum wg term to tilestart
s_lshl_b64 s[64:65], s[64:65], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s64    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s65   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v17, s[sgprStrideAL], v[15]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc, v[12], v[17+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetA+0+0], 0x4, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_mul_lo_u32 v17, s[sgprStrideB1J], v[13]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc, v[16], v[17+0] // accumulate K lower
v_add_u32 v[vgprGlobalReadOffsetB+0+0], 0x4, v[vgprGlobalReadOffsetB+0+0] // add prepad for pointer shift
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: increments a */
s_and_b32 s65, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsA+0], 32             // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s65, s65, s[sgprGlobalReadIncsA+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s64, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cmov_b32 s65, 32                                 // DepthU*Bpe if GSUC = 1
s_mul_i32 s[sgprGlobalReadIncsA+0], s65, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_and_b32 s65, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsB+0], 32             // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s65, s65, s[sgprGlobalReadIncsB+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s64, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s65 // incrB (unrollIdx)
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...4) from pool */

/* initC: remove ValuA/B vgpr buffer [6...14) from pool */
v_accvgpr_write acc0, 0                            // initC
v_accvgpr_write acc1, 0                            // initC
v_accvgpr_write acc2, 0                            // initC
v_accvgpr_write acc3, 0                            // initC
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_and_b32 s64, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s64, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
v_cvt_f32_u32 v15, s[sgprGSUSumIdx+1]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v15, v15                           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v16, s[sgprLoopCounterL]             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v15, v15, v16                            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v15, v15                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v16, v15, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v16, s[sgprLoopCounterL], v16            // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v16, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v15, 1, v15                              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v16, 0                                   // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v16, s[sgprGSUSumIdx+1]        // overflow happened in remainder
v_sub_u32 v15, v15, 1                              // quotient - 1
v_mul_u32_u24 v16, v15, s[sgprGSUSumIdx+1]         // re-calculate remainder
v_sub_u32 v16, s[sgprLoopCounterL], v16            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v15       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v16        // remainder
s_add_u32 s64, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s64                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s66, s[sgprStaggerU], 0x1f00
s_lshr_b32 s66, s66, 0x8
s_and_b32 s67, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s64, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s65, s64, s66                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s65           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s64, s64, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s65, s64, 1                              // staggerU mask
s_cmp_ge_u32 s64, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s65, 0          // set Mask
s_cmp_eq_u32 s67, 0x0
s_cbranch_scc0 label_StaggerUMapping
s_mov_b32 s64, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping:
s_cmp_eq_u32 s67, 0x2000
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s64, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s67, 0x4000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s64, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s67, 0x6000
s_cbranch_scc0 label_StaggerUMapping_3
s_mul_i32 s65, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s64, s64, s65
s_add_u32 s64, s64, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s67, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s64, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s64 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s66 // shift by StaggerUStride

/* addr += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_u32 s65, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s64, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* addr += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_u32 s65, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s64, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_u32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s65 // limit -= inc)
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
buffer_load_dwordx2 v[vgprG2LA+0:vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx2 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s64, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s65, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s64, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s65, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt vmcnt(0)                                 // 5wait for global read
// Skip barrier: NumThreads=64PGR=0, prior iter done reading lds, LW wait LR, sync

/* local write a */
ds_write_b64 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+1] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0

/* local write b */
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:32 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 32 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:64 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 64 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:96 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 96 sync LDS0
s_waitcnt lgkmcnt(0)                               // 2prefetch wait for local write
// Skip barrier: NumThreads=64After LW code, sync

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:0  */
/*  mfmaIndex:0  */
ds_read_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuB_X0_I0_D1+0], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuB_X0_I0_D1+1], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(0)                               // Wait for dependent lr
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=4 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=4 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/* Tail: add ValuA/B vgpr buffer [6...14) to pool */

/* Tail: add address/G2L vgpr [14...14) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/
/* Check out VGPR (numG2LA,numG2LB,numG2LMXSA,numG2LMXSB,numG2LMetadata) = (2,2,0,0,0) */
.set vgprG2LA_BASE, 6
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB_BASE, 8
.set vgprG2LB, vgprG2LB_BASE+0

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_and_b32 s64, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s65, s[sgprSizesSum], 4                 // s65 = s[sgprSizesSum] / 16
s_and_b32 s66, s[sgprGSU], 0xfff                   // Restore GSU
v_cvt_f32_u32 v10, s66                             // s64 = s65 / s66
v_rcp_iflag_f32 v10, v10                           // s64 = s65 / s66
v_cvt_f32_u32 v11, s65                             // s64 = s65 / s66
v_mul_f32 v10, v10, v11                            // s64 = s65 / s66
v_cvt_u32_f32 v10, v10                             // s64 = s65 / s66
v_mul_u32_u24 v11, v10, s66                        // s64 = s65 / s66
v_sub_u32 v11, s65, v11                            // s64 = s65 / s66
v_cmpx_eq_u32 exec, v11, s66                       // s64 = s65 / s66
v_add_u32 v10, 1, v10                              // s64 = s65 / s66
v_mov_b32 v11, 0                                   // s[sgprGSUSumIdx+1] = s65 % s66
s_mov_b64 exec, -1                                 // Reset exec
v_cmpx_gt_u32 exec, v11, s66                       // overflow happened in remainder
v_sub_u32 v10, v10, 1                              // quotient - 1
v_mul_u32_u24 v11, v10, s66                        // re-calculate remainder
v_sub_u32 v11, s65, v11                            // re-calculate remainder
s_mov_b64 exec, -1                                 // Reset exec
v_readfirstlane_b32 s64, v10                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v11        // remainder
s_sub_u32 s65, s66, 1                              // GSU-1
s_cmp_eq_u32 s64, 0                                // quotient == 0
s_cselect_b32 s64, s[sgprGSUSumIdx+1], s65         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s64                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
//  removeStagger A
s_sub_i32 s64, 2, s[sgprStaggerUIter]
s_cmp_ge_i32 s64, 0
s_cbranch_scc0 label_Negative_0
s_mul_hi_u32 s65, s64, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s64, s64, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_1
label_Negative_0:
s_abs_i32 s64, s64
s_mul_hi_u32 s65, s64, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s64, s64, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s64, s64, 0xffffffff
s_xor_b32 s65, s65, 0xffffffff
s_add_u32 s64, s64, 0x1
s_addc_u32 s65, s65, 0
label_MultiplyDone_1:
s_sub_u32 s64, s64, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s65, s65, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
//  removeStagger B
s_sub_i32 s64, 2, s[sgprStaggerUIter]
s_cmp_ge_i32 s64, 0
s_cbranch_scc0 label_Negative_2
s_mul_hi_u32 s65, s64, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s64, s64, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_3
label_Negative_2:
s_abs_i32 s64, s64
s_mul_hi_u32 s65, s64, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s64, s64, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s64, s64, 0xffffffff
s_xor_b32 s65, s65, 0xffffffff
s_add_u32 s64, s64, 0x1
s_addc_u32 s65, s65, 0
label_MultiplyDone_3:
s_sub_u32 s64, s64, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s65, s65, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s64        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s65       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s64 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s65 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* Tail global read A */
buffer_load_dwordx2 v[vgprG2LA+0:vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* Update M0 for DTLDS */

/* Tail global read B */
buffer_load_dwordx2 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* release sgprs that will not be used */
.set sgprWGM, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF

/* find the last element location for a */

/* find the last element location for b */
// Calculate SizeJ % MacroTile1
s_mul_i32 s20, s[sgprWorkGroup1], 16               // Calculate the remaining dimension along I/J direction.
s_sub_u32 s20, s[sgprSizeJ], s20                   // Calculate the remaining dimension along I/J direction.
s_mul_i32 s20, s20, 2                              // In bytes
s_and_b32 s24, s[sgprSizeL], 15                    // Calculate the remaining dimension along L direction.
s_lshr_b32 s56, s24, 0x4                           // Divided by lsc(16)
s_mul_hi_u32 s22, s20, s24                         // Calculate total number of valid elements.
s_mul_i32 s26, s20, s24                            // Calculate total number of valid elements.
s_cmp_gt_u32 s22, 0
s_cmov_b32 s26, 0xffffffff                         // If valid elements > max(U32), set the value to max
s_sub_u32 s24, s[sgprSizeJ], 1                     // sLoadTileIdx starts from 0
// Calculate SizeJ - 1 % MacroTile1
s_lshr_b32 s20, s24, 4                             // s20 = s24 / 16
s_and_b32 s20, 15, s24                             // s20 = s24 % 16
s_lshr_b32 s20, s20, 0x4                           // Divide lsp to get the load tile index
s_mul_i32 s20, s20, 1                              // Multiply nlc
s_add_i32 s20, s20, s56
s_and_b32 s24, 15, s[sgprSizesSum+0]               // s24 = s[sgprSizesSum+0] % 16
s_and_b32 s24, s24, 3                              // sLoadNum = (SizesSum+0 mod DU) & glvw
s_and_b32 s22, s24, 0x1
s_mov_b32 s27, 0                                   // Set loop count = 0

/* load single element for B */
label_LoadB:
s_cmp_eq_u32 s22, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_MergeB                        // Skip loading B
label_LOAD_B0:
label_LOAD_B0_K1:
s_cmp_ge_u32 s24, 1
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 0 */
buffer_load_short_d16 v12, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
label_LOAD_B0_K3:
s_cmp_ge_u32 s24, 3
s_cbranch_scc0 label_MergeB
/* g2l=0, load component 2 */
buffer_load_short_d16 v13, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
s_branch label_MergeB

/* merge single element for B */
label_MergeB:
s_cmp_eq_u32 s22, 0                                // Valid loading size per thread is multiples of 4 bytes
s_cbranch_scc1 label_CheckOtherLoadB               // Skip mergeing B
label_MERGE_B0:
label_MERGE_B0_K1:
s_cmp_ge_u32 s24, 1
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v12     // HasEccHalf: pack
label_MERGE_B0_K3:
s_cmp_ge_u32 s24, 3
s_cbranch_scc0 label_CheckOtherLoadB
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v13     // HasEccHalf: pack
s_branch label_CheckOtherLoadB

/* reload loop for a: check if there's other load range need to be reloaded */
label_CheckOtherLoadA:

/* reload loop for b: check if there's other load range need to be reloaded */
label_CheckOtherLoadB:
s_cmp_eq_u32 s22, 0                                // Noneed to load single element for B?
s_cbranch_scc1 label_TailGlobalLoadEnd
s_add_u32 s27, s27, 1
s_cmp_eq_u32 s27, 1                                // Have reloaded all subtiles?
s_cbranch_scc1 label_TailGlobalLoadEnd
s_sub_i32 s20, s20, 1                              // Check the upper subtile
s_cmp_lt_i32 s20, 0
s_cselect_b32 s54, 1, 0                            // Back to the last subtile
s_add_i32 s20, s20, s54                            // If currently reload the first subtile,                                   check the last subtile next.
label_B0:
v_mov_b32 v10, v[vgprGlobalReadOffsetB+0]
label_CheckAddrB:
v_sub_u32 v10, v10, 8                              // sub prepad
v_add_u32 v11, v10, 7                              // Calculate load range per thread
v_cmp_lt_i32 s[54:55], v10, s26                    // If loading start address < total valid bytes?
v_cmp_ge_i32 s[56:57], v11, s26                    // If loading end address >= total valid bytes?
s_and_b32 s54, s54, s56                            // Find threads which access the last element
s_and_b32 s55, s55, s57                            // Find thread that access the last element
s_add_u32 s54, s54, s55                            // Find thread that access the last element
s_cmp_lg_u32 s54, 0                                // Have threads access the last element?
s_cbranch_scc1 label_LoadB                         // Reload B

/* global read for tail done */
label_TailGlobalLoadEnd:
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip barrier: NumThreads=64

/* local write a */
ds_write_b64 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+1] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0 sync LDS0

/* local write b */
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+0] offset:32 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 32 sync LDS0
ds_write_b16 v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:64 // lwoB_0_2_0_0 = (2 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 64 sync LDS0
ds_write_b16_d16_hi v[vgprLocalWriteAddrB+0], v[vgprG2LB+1] offset:96 // lwoB_0_3_0_0 = (3 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 96 sync LDS0

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip barrier: NumThreads=64Tail loop LW->LR, sync LDS0
.set vgprG2LA_BASE, UNDEF
.set vgprG2LA, UNDEF
.set vgprG2LB_BASE, UNDEF
.set vgprG2LB, UNDEF
.set vgprValuA_X0_I0_BASE, 6
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X0_I0_D0_PACK, 8
.set vgprValuA_X0_I0_D1, vgprValuA_X0_I0_D0_PACK+0
.set vgprValuB_X0_I0_BASE, 10
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X0_I0_D0_PACK, 12
.set vgprValuB_X0_I0_D1, vgprValuB_X0_I0_D0_PACK+0

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
ds_read_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
ds_read_u16 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuB_X0_I0_D1+0], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0 sync LDS0
ds_read_u16_d16_hi v[vgprValuB_X0_I0_D1+1], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
/* Adding additional 128 pad since cumulative inc has reached 128 */
s_mov_b32 s11, 640                                 // inc
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc, s11, v[vgprLocalReadAddrA+0] // lrA += 512 ((MT+PAD)*bpeDS)

/* local read inc b */
/* Adding additional 128 pad since cumulative inc has reached 128 */
                                                   // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, s11, v[vgprLocalReadAddrB+0] // lrB += 512 ((MT+PAD)*bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0_D1+0] // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0_D1+1] // pack two half Vgpr to one Vgpr
v_and_b32 v15, 63, v[vgprSerial]                   // v15 = v[vgprSerial] % 64
v_lshrrev_b32 v15, 4, v15                          // 15 = 15 / 16
v_lshlrev_b32 v15, 2, v15                          // v15 = v15 * 4
v_cmp_ge_i32 s[20:21], v15, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], 0, s[20:21] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], 0, s[20:21] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0, s[20:21] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0, s[20:21] // set 0 if K_idx >= sizeL
v_sub_u32 v16, s[sgprLoopCounterL], v15            // get distance between size and k index
v_cmp_lt_i32 s[20:21], v16, 4                      // set partial 0 if distance less than input per thread
s_and_b32 s22, s[sgprSizeL], 7                     // if summation is multiple of 8, skip masking
s_cmp_eq_u32 s22, 0
s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_4    // skip mask
s_and_b32 s22, s[sgprLoopCounterL], 3              // get inputs for edge thread
s_sub_u32 s22, 4, s22                              // use shift to fill 0 for outside element
s_lshl_b32 s22, s22, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[18:19], s22, v[vgprValuA_X0_I0+0+0+0+0:vgprValuA_X0_I0+0+0+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v18, s[20:21]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v19, s[20:21]
v_lshlrev_b64 v[18:19], s22, v[vgprValuB_X0_I0+0+0+0+0:vgprValuB_X0_I0+0+0+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v18, s[20:21]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v19, s[20:21]
label_TailLoop_SkipZeroOutMask_4:
s_nop 1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]

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

/* Tail: add MISC Vgpr [0...6) to pool */
label_Summation_End_5:
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
/* load store sgprs */
/* load store sgprs2 */
.set sgprAddressTD, 38
.set sgprSynchronizer, 40
/* Check if custom structure pointer is null */
s_load_dwordx2 s[sgprAddressTD:sgprAddressTD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_dwordx2 s[sgprSynchronizer:sgprSynchronizer+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
label_LoadExternalEpilogueStruct_3:
.set sgprSrdC, 24
.set sgprSrdD, 20
.set sgprSrdTD, 44
.set sgprGSUSync, 11
.set sgprSrdSync, 48

/* Mapping of Acc register -> C Vgpr register */
s_and_b32 s12, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s12, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
label_RegularSrdInitializationD:  /// Regular SRD initialization for non-General Batched GEMM for D
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_GeneralBatchedGemmSrdInitiationD:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationD_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_and_b32 s12, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s12, 1                                // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
label_RegularSrdInitializationC:  /// Regular SRD initialization for non-General Batched GEMM for C
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_GeneralBatchedGemmSrdInitiationC:  /// Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0            // init SRD to 0
label_GeneralBatchedGemmSrdInitiationC_End:  /// End of handling General Batched GEMM SRD initialization
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s54, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s53, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s53, s54, s[sgprStrideC1J]            // ScaleC s54 by Stride
s_mul_i32 s52, s54, s[sgprStrideC1J]               // ScaleC s54 by Stride
s_lshl_b64 s[52:53], s[52:53], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s52        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s53       // add hi to SRD
s_and_b32 s53, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s53, s54, s[sgprStrideD1J]            // ScaleD s54 by Stride
s_mul_i32 s52, s54, s[sgprStrideD1J]               // ScaleD s54 by Stride
s_lshl_b64 s[52:53], s[52:53], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s52        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s53       // add hi to SRD

s_and_b32 s53, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s53, 1                                // GSU == 1 ?
s_cbranch_scc0 label_ArgTypeChecksC
s_branch label_StridedBatchedGemmLoadC
label_ArgTypeChecksC:  /// Checks for ArgType to General Batched or non-General Batched
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s53, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s52, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[52:53], s[52:53], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s52        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s53       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s52, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s52, s52, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s53, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[52:53], s[52:53], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s52        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s53       // Offsetting within the Batch Matrix [Higher half of address]
s_load_dwordx2 s[52:53], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x74 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s52        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s53       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s53, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s53, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s53, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s52, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[52:53], s[52:53], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s52        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s53       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s52, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s52, s52, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s53, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[52:53], s[52:53], 0               // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s52        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s53       // Offsetting within the Batch Matrix [Higher half of address]
s_load_dwordx2 s[52:53], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x6c // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s52        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s53       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

s_and_b32 s12, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s12, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s53, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s52, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s54, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s54, s54, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s55, s54, s[sgprStrideC1J]            // Free1
s_mul_i32 s54, s54, s[sgprStrideC1J]               // Free1
s_add_u32 s52, s52, s54                            // Free1
s_addc_u32 s53, s53, s55                           // Free1
s_sub_u32 s54, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s54, s54, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s55, s54, s[sgprStrideCK]             // Free2
s_mul_i32 s54, s54, s[sgprStrideCK]                // Free2
s_add_u32 s52, s52, s54                            // Free2
s_addc_u32 s53, s53, s55                           // Free2
s_lshl_b64 s[52:53], s[52:53], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s52        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s53       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/* shift vector components d0 */
v_mov_b32 v3, s[sgprWorkGroup0]
v_mul_i32_i24 v3, -0x10, v3                        // wg*MT
v_add_co_u32 v3, vcc, s[sgprSizesFree+0], v3       // wgMT = Size - wg*MT
v_mov_b32 v4, 0x10                                 // MT
v_cmp_lt_u32 s[12:13], v3, v4                      // wgMT < MT
v_cndmask_b32 v3, v4, v3, s[12:13]                 // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v5, 6, v[vgprSerial]                 // 5 = Serial / 64
v_and_b32 v5, 0, v5                                // v5 = v5 % 1
v_lshrrev_b32 v6, 4, v3                            // 6 = 3 / 16
v_and_b32 v6, 0, v6                                // v6 = v6 % 1
v_cmp_eq_u32 s[12:13], v6, v5                      // wave_id == block_belong_to_wave?
v_cndmask_b32 v3, v4, v3, s[12:13]                 // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(1)) */
v_lshrrev_b32 v4, 4, v3                            // 4 = 3 / 16
v_lshlrev_b32 v6, 0, v5                            // v6 = v5 * 1
v_sub_u32 v4, v4, v6

/* gbReg: glvw block id */
v_lshrrev_b32 v6, 2, v3                            // 6 = 3 / 4

/* tgbReg: glvw block id */
v_lshrrev_b32 v7, 4, v[vgprSerial]                 // 7 = Serial / 16
v_and_b32 v7, 3, v7                                // v7 = v7 % 4
v_lshlrev_b32 v7, 2, v7                            // v7 = v7 * 4
v_lshrrev_b32 v7, 2, v7                            // 7 = 7 / 4
v_lshlrev_b32 v5, 2, v5                            // v5 = v5 * 4
v_add_co_u32 v7, vcc, v5, v7                       // tgbReg = (tid_coal * continOut) / GLVW
v_sub_u32 v6, v6, v7

/* vwReg: glvw in which vw block? */
v_and_b32 v5, 3, v3                                // permute register between threads
v_lshrrev_b32 v5, 2, v5                            // permute register between threads

/* rReg : reminder of M_size % GlobalReadVectorWidth */
v_and_b32 v7, 3, v3                                // v7 = v3 % 4
v_cmp_eq_u32 vcc, v7, 0x1                          // wgMT%VW == 1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1 // branch to shift d0 r=1
v_cmp_eq_u32 vcc, v7, 0x2                          // wgMT%VW == 2
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2 // branch to shift d0 r=2
v_cmp_eq_u32 vcc, v7, 0x3                          // wgMT%VW == 3
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3 // branch to shift d0 r=3

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
/* shift d0 r=1 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0_VW0:  /// r1 mb0 vw0
s_mov_b32 s12, 0
v_cmpx_eq_u32 s[12:13], v6, s12                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc3                        // glvw 1 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
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
v_accvgpr_write_b32 acc0, v7
v_accvgpr_write_b32 acc1, v8
v_accvgpr_write_b32 acc2, v9
s_mov_b64 s[12:13], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[12:13]                    // all threads active

/* no shifting */
s_branch label_ShiftVectorComponents0_GLVW0

label_ShiftVectorComponents0_GLVW0:  /// end shift0

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // 4 = Serial / 64
v_lshrrev_b32 v5, 0, v4                            // 5 = 4 / 1
v_lshlrev_b32 v1, 4, v5                            // wave coordination offset 1
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v1, v5, v1, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v5, 0, v4                                // v5 = v4 % 1
v_lshlrev_b32 v5, 4, v5                            // wave coordination offset 0
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v0, 4, v0                            // 0 = 0 / 16
v_lshlrev_b32 v0, 2, v0                            // thread0 * continuous_output
v_add_lshl_u32 v0, v5, v0, 0                       // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s8, 16, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_u32 v0, s8, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 16, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v1, s8, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 16 bytes of kern args.
s_and_b32 s8, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc0 label_NoBranch_6                    // Only branch on scc1
// long branch if GSU == 1
s_getpc_b64 s[52:53]                               // addr of next instr
s_add_i32 s54, label_GSU_3, 4                      // target branch offset
s_add_u32 s52, s52, s54                            // add target branch offset
s_addc_u32 s53, s53, 0                             // add high and carry
s_setpc_b64 s[52:53]                               // branch to label_GSU_3
label_NoBranch_6:
/* calculate SrdTD address */
s_mov_b32 s[sgprSrdTD+2], BufferOOB
s_mov_b32 s[sgprSrdTD+3], Srd127_96                // Set bits 127_96 in post-loop SRD
s_mul_i32 s8, MT1, s[sgprWorkGroup1]
s_mul_hi_u32 s13, s8, s[sgprStrideD1J]
s_mul_i32 s12, s8, s[sgprStrideD1J]
s_lshl_b64 s[12:13], s[12:13], 1                   // scale by bpe
s_add_u32 s[sgprSrdTD+0], s[sgprAddressTD+0], s12  // add lo to SRTD
s_addc_u32 s[sgprSrdTD+1], s[sgprAddressTD+1], s13 // add hi to SRTD
s_mul_hi_u32 s13, s[sgprWorkGroup2], s[sgprStrideDK]
s_mul_i32 s12, s[sgprWorkGroup2], s[sgprStrideDK]
s_lshl_b64 s[12:13], s[12:13], 1                   // scale by bpe
s_branch label_SrdTDInit_GeneralBatched_End
label_SrdTDInit_GeneralBatched:
s_mul_i32 s32, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s32, s32, s[sgprAddressTD+0]             // Offsetting to the location [Lower half of address]
s_addc_u32 s33, s[sgprAddressTD+1], 0              // Offsetting to the location [Higher half of address]
s_load_dwordx2 s[sgprSrdTD:sgprSrdTD+1], s[32:33], 0 // Load the Matrix Address in the Pointer Array
s_load_dwordx2 s[52:53], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x6c // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address and batchOffsetD Load
s_add_u32 s[sgprSrdTD+0], s[sgprSrdTD+0], s52      // Apply batchOffsetD to SrdTD (low)
s_addc_u32 s[sgprSrdTD+1], s[sgprSrdTD+1], s53     // Apply batchOffsetD to SrdTD (high)
label_SrdTDInit_GeneralBatched_End:
s_add_u32 s[sgprSrdTD+0], s[sgprSrdTD+0], s12      // add lo to SRTD
s_addc_u32 s[sgprSrdTD+1], s[sgprSrdTD+1], s13     // add hi to SRTD
label_Reduction_Start:  /// Reduction start
.set sgprGSUStartWGIdx, 8
label_Reduction_B0_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=1 */
// calculate the starting WG index of GSU WGs
s_mul_i32 s12, s[sgprNumWorkGroups0], s[sgprWorkGroup1] // NumWorkGroups0*wg1
s_and_b32 s13, s[sgprGSU], 0xfff                   // Restore GSU
s_add_u32 s12, s12, s[sgprWorkGroup0]              // NumWorkGroups0*wg1+wg0
s_mul_i32 s12, s12, s13                            // (NumWorkGroups0*wg1+wg0)*GSU
s_mul_i32 s[sgprGSUStartWGIdx], s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // NumWgPerBatch
s_mul_i32 s[sgprGSUStartWGIdx], s[sgprGSUStartWGIdx], s[sgprWorkGroup2] // NumWgPerBatch
s_mul_i32 s[sgprGSUStartWGIdx], s[sgprGSUStartWGIdx], s13 // NumWgPerBatch
s_add_u32 s[sgprGSUStartWGIdx], s[sgprGSUStartWGIdx], s12 // starting WG index of each GSU WGs
s_add_u32 s12, s[sgprGSUStartWGIdx], s[sgprGSUSumIdx] // (NumWorkGroups0*wg1+wg0)*GSU+NumWgPerBatch+GSUSumIdx
// add offset to the base address of workspace buffer
s_mul_hi_u32 s13, s12, MTOffset                    // (MT0*MT1*bpeC)*WGIdx
s_mul_i32 s12, s12, MTOffset                       // (MT0*MT1*bpeC)*WGIdx
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s12    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s13   // add hi to SRD

// synchronizer offset cal
s_mul_i32 s52, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s33, s52, s[sgprWorkGroup2]
s_mul_i32 s32, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_add_u32 s32, s32, s[sgprWorkGroup0]
s_add_u32 s32, s32, s33
v_readfirstlane_b32 s33, v[vgprSerial]
s_mul_i32 s52, s52, s[sgprSizeK]                   // cal a wave offset
s_lshr_b32 s33, s33, 0x6
s_mul_i32 s33, s52, s33                            // wave offset at batch
s_add_u32 s32, s33, s32
s_lshl_b32 s32, s32, 0x2
s_add_u32 s[sgprSrdSync+0], s[sgprSynchronizer+0], s32
s_addc_u32 s[sgprSrdSync+1], s[sgprSynchronizer+1], 0x0

s_and_b32 s33, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_gt_i32 s33, 2                                // GSU > 2 ?
s_cbranch_scc1 label_partial_write                 // branch if true
// GSU <= 2, last gsu wg do the reduction
s_sub_u32 s33, s33, 1
s_cmp_eq_u32 s[sgprGSUSumIdx], s33                 // GSUSumIdx == GSU-1 ?
s_cbranch_scc0 label_partial_write                 // branch if false
label_last_gsu_wg_busy_waiting:
s_load_dword s12, s[sgprSrdSync:sgprSrdSync+1], 0 glc // get atomic_dec value
s_waitcnt lgkmcnt(0)                               // wait for atomic_dec value load
s_cmp_eq_u32 s12, 1                                // last GSU WG?
s_cbranch_scc0 label_last_gsu_wg_busy_waiting      // branch if false
s_mov_b32 s12, 0                                   // reset synchronizer
s_store_dword s12, s[sgprSrdSync:sgprSrdSync+1], 0 glc // reset synchronizer
s_branch label_reduction_body
label_partial_write:
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Partial Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/
// accvgpr read
v_accvgpr_read_b32 v76, acc0                       // copy acc to vreg[0]
v_accvgpr_read_b32 v77, acc1                       // copy acc to vreg[1]
v_accvgpr_read_b32 v78, acc2                       // copy acc to vreg[2]
v_accvgpr_read_b32 v79, acc3                       // copy acc to vreg[3]

// write to workspace
v_lshlrev_b32 v11, 4, v[vgprSerial]                // v11 = v[vgprSerial] * 16
s_mov_b32 s12, 0                                   // Init sgpr offset for interleaved wave store
buffer_store_dwordx4 v[76:79], v11, s[sgprSrdD:sgprSrdD+3], s12 offen offset:0 sc0 sc1 // store WS
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Reduction Batch #0 (d1,d0,vc1,vc0) =   */
/*    (0,0,0,0:vw4)                       */
/******************************************/
// check done start
s_waitcnt 0                                        // (Wait all)
s_and_b32 s32, s[sgprGSU], 0xfff                   // Restore GSU
s_sub_u32 s32, s32, 1
s_atomic_dec s32, s[sgprSrdSync:sgprSrdSync+1] glc
s_and_b32 s33, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_gt_i32 s33, 2                                // GSU > 2 ?
s_cbranch_scc0 label_reduction_skip                // branch if false
// GSU > 2, atomic_dec selects the gsu wg to do the reduction
// check synchronizer done
s_waitcnt lgkmcnt(0)                               // Wait for synchronizer
s_cmp_eq_u32 s32, 0x1
s_cbranch_scc1 label_reduction_body                // branch if true
label_reduction_skip:
s_endpgm
// check done end
label_reduction_body:
s_mov_b32 s13, 0                                   // Init sgpr offset for interleaved wave load
v_lshlrev_b32 v11, 4, v[vgprSerial]                // v11 = v[vgprSerial] * 16
v_mov_b32 v6, BufferOOB
// synchronizer sum offset is equal to MTOffset (=MT0*MT1*bpeC)
s_mul_hi_u32 s57, MTOffset, s[sgprGSUStartWGIdx]   // (MT0*MT1*bpeC)*WGIdx
s_mul_i32 s56, MTOffset, s[sgprGSUStartWGIdx]      // (MT0*MT1*bpeC)*WGIdx
s_add_u32 s56, s[sgprAddressD+0], s56              // add lo to SRD
s_addc_u32 s57, s[sgprAddressD+1], s57             // add hi to SRD
s_mov_b64 s[58:59], s[sgprSrdD+2:sgprSrdD+2+1]
s_and_b32 s32, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_gt_i32 s32, 2                                // GSU > 2 ?
s_cbranch_scc1 label_reduction_all_gsu_wg          // branch if true
// GSU <= 2, so we minus 1 from GSUSync at the beginning
s_sub_i32 s32, s32, 1                              // Use GSU-1
label_reduction_all_gsu_wg:
// buffer load start
buffer_load_dwordx4 v[76:79], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 0 element 0
s_mov_b32 s[sgprGSUSync], s32                      // Init GSUSync to GSU for batch 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 0
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_1   // SyncAddbranchhere
buffer_load_dwordx4 v[16:19], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 1 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 1
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_2   // SyncAddbranchhere
buffer_load_dwordx4 v[20:23], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 2 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 2
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_3   // SyncAddbranchhere
buffer_load_dwordx4 v[24:27], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 3 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 3
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_4   // SyncAddbranchhere
buffer_load_dwordx4 v[28:31], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 4 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 4
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_5   // SyncAddbranchhere
buffer_load_dwordx4 v[32:35], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 5 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 5
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_6   // SyncAddbranchhere
buffer_load_dwordx4 v[36:39], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 6 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 6
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_7   // SyncAddbranchhere
buffer_load_dwordx4 v[40:43], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 7 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 7
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_8   // SyncAddbranchhere
buffer_load_dwordx4 v[44:47], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 8 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 8
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_9   // SyncAddbranchhere
buffer_load_dwordx4 v[48:51], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 9 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 9
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_10  // SyncAddbranchhere
buffer_load_dwordx4 v[52:55], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 10 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 10
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_11  // SyncAddbranchhere
buffer_load_dwordx4 v[56:59], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 11 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 11
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_12  // SyncAddbranchhere
buffer_load_dwordx4 v[60:63], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 12 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 12
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_13  // SyncAddbranchhere
buffer_load_dwordx4 v[64:67], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 13 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 13
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_14  // SyncAddbranchhere
buffer_load_dwordx4 v[68:71], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 14 element 0
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 14
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
s_cmp_eq_i32 s[sgprGSUSync], 0
s_cbranch_scc1 label_Synchronizer_read_add_end_15  // SyncAddbranchhere
buffer_load_dwordx4 v[72:75], v11, s[56:59], s13 offen offset:0 sc0 sc1 // load GSU WG 15 element 0
// buffer load end
// buffer add start
label_Synchronizer_read_add:  /// Synchronizer read add
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 0
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[16:19], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 1
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[20:23], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 2
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[24:27], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 3
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[28:31], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 4
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[32:35], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 5
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[36:39], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 6
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[40:43], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[44:45]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[46:47]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 7
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[44:47], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[48:49]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[50:51]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 8
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[48:51], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[52:53]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[54:55]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 9
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[52:55], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[56:57]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[58:59]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 10
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[56:59], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[60:61]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[62:63]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 11
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[60:63], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[64:65]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[66:67]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 12
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[64:67], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[68:69]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[70:71]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 13
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[68:71], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
s_waitcnt vmcnt(14)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[72:73]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[74:75]          // buffer pk
s_sub_i32 s[sgprGSUSync], s[sgprGSUSync], 1        // 14
s_cmp_le_i32 s[sgprGSUSync], -14
s_cbranch_scc1 label_Synchronizer_read_add_skip    // SyncAddbranch
s_add_u32 s56, s56, MTOffset
s_addc_u32 s57, s57, MTOffsetH32
v_cmp_ge_i32 s[52:53], 0, s[sgprGSUSync]
v_cndmask_b32 v4, v11, v6, s[52:53]                // protect if OOB
buffer_load_dwordx4 v[72:75], v4, s[56:59], s13 offen offset:0 sc0 sc1 // prefetch GSU WG element 0
// buffer add end
s_cmp_gt_i32 s[sgprGSUSync], -0xe
s_cbranch_scc1 label_Synchronizer_read_add         // Syncbranchhere

label_Synchronizer_read_add_end_15:  /// Synchronizer read add end_15
s_waitcnt vmcnt(13)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(12)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(11)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(10)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(9)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(8)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_waitcnt vmcnt(7)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_waitcnt vmcnt(6)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[44:45]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[46:47]          // buffer pk
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[48:49]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[50:51]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[52:53]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[54:55]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[56:57]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[58:59]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[60:61]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[62:63]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[64:65]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[66:67]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[68:69]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[70:71]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_14:  /// Synchronizer read add end_14
s_waitcnt vmcnt(12)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(11)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(10)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(9)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(8)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(7)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_waitcnt vmcnt(6)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[44:45]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[46:47]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[48:49]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[50:51]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[52:53]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[54:55]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[56:57]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[58:59]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[60:61]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[62:63]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[64:65]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[66:67]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_13:  /// Synchronizer read add end_13
s_waitcnt vmcnt(11)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(10)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(9)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(8)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(7)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(6)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[44:45]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[46:47]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[48:49]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[50:51]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[52:53]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[54:55]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[56:57]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[58:59]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[60:61]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[62:63]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_12:  /// Synchronizer read add end_12
s_waitcnt vmcnt(10)                                // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(9)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(8)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(7)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(6)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[44:45]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[46:47]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[48:49]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[50:51]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[52:53]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[54:55]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[56:57]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[58:59]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_11:  /// Synchronizer read add end_11
s_waitcnt vmcnt(9)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(8)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(7)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(6)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[44:45]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[46:47]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[48:49]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[50:51]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[52:53]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[54:55]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_10:  /// Synchronizer read add end_10
s_waitcnt vmcnt(8)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(7)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(6)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[44:45]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[46:47]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[48:49]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[50:51]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_9:  /// Synchronizer read add end_9
s_waitcnt vmcnt(7)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(6)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[44:45]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[46:47]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_8:  /// Synchronizer read add end_8
s_waitcnt vmcnt(6)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[40:41]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[42:43]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_7:  /// Synchronizer read add end_7
s_waitcnt vmcnt(5)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[36:37]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[38:39]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_6:  /// Synchronizer read add end_6
s_waitcnt vmcnt(4)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[32:33]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[34:35]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_5:  /// Synchronizer read add end_5
s_waitcnt vmcnt(3)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[28:29]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[30:31]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_4:  /// Synchronizer read add end_4
s_waitcnt vmcnt(2)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[24:25]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[26:27]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_3:  /// Synchronizer read add end_3
s_waitcnt vmcnt(1)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[20:21]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[22:23]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_2:  /// Synchronizer read add end_2
s_waitcnt vmcnt(0)                                 // (wait for buffer ready)
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk
s_branch label_Synchronizer_read_add_skip          // SyncAddbranch

label_Synchronizer_read_add_end_1:  /// Synchronizer read add end_1
label_Synchronizer_read_add_skip:  /// Synchronizer read add skip
// buffer add end2
// synchronizer store end
s_and_b32 s12, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_gt_i32 s12, 2                                // GSU > 2 ?
s_cbranch_scc1 label_accvgpr_write                 // branch if true
// GSU <= 2, do accvgpr_read for the last gsu wg
label_last_gsu_wg_accvgpr_read:
s_waitcnt vmcnt(0)                                 // wait for buffer_load to finish
v_accvgpr_read_b32 v16, acc0                       // copy acc to vreg[0]
v_accvgpr_read_b32 v17, acc1                       // copy acc to vreg[1]
v_accvgpr_read_b32 v18, acc2                       // copy acc to vreg[2]
v_accvgpr_read_b32 v19, acc3                       // copy acc to vreg[3]
v_pk_add_f32 v[76:77], v[76:77], v[16:17]          // buffer pk
v_pk_add_f32 v[78:79], v[78:79], v[18:19]          // buffer pk

label_accvgpr_write:
// accvgpr write
v_accvgpr_write_b32 acc0, v76                      // copy vreg[0] to acc
v_accvgpr_write_b32 acc1, v77                      // copy vreg[1] to acc
v_accvgpr_write_b32 acc2, v78                      // copy vreg[2] to acc
v_accvgpr_write_b32 acc3, v79                      // copy vreg[3] to acc
s_waitcnt vmcnt(0)                                 // wait for buffer_load to finish
s_and_b32 s12, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_gt_i32 s12, 2                                // GSU > 2 ?
s_cbranch_scc1 label_Reduction_End                 // branch if true
// GSU > 2, no need to reset synchronizer
s_waitcnt lgkmcnt(0)                               // wait for reset synchronizer to finish
.set sgprGSUStartWGIdx, UNDEF
label_Reduction_End:  /// Reduction end

s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_MBSK                    // Branch if Beta is not zero

label_GW_B0_MBSK:
label_GW_B0_FD0_MBSK:

/* Edge/NonEdge store path check (M): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s52, 15, s[sgprSizeI]                    // s52 = s[sgprSizeI] % 16
s_add_u32 s53, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s53                // wg0 >= nwg0-1 ?
s_cselect_b32 s52, s52, 0                          // set rem
s_cmpk_gt_u32 s52, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MBSK_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s52, 15, s[sgprSizeJ]                    // s52 = s[sgprSizeJ] % 16
s_add_u32 s53, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s53                // wg1 >= nwg1-1
s_cselect_b32 s52, s52, 0                          // set rem
s_cmpk_gt_u32 s52, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_MBSK_Then       // jump if edges required
label_GW_B0_FD0_VW4_MBSK_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=10 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v12, v3, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+76], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+77], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+78], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+79], acc3           // copy acc to vreg[3]

/* store after Acc, GSU: 2 */
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_pk_f16_f32 v76, v[vgprValuC+76], v[vgprValuC+77] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v77, v[vgprValuC+78], v[vgprValuC+79] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[76:77], v12, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
                                                   // GW end
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_MBSK_NonEdgeEnd:
label_GW_B0_FD0_VW4_MBSK_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=6 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[52:53], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v12, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v12, v6, v12, s[56:57]               // LDTD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+76], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+77], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+78], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+79], acc3           // copy acc to vreg[3]

/* store after Acc, GSU: 2 */
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_pk_f16_f32 v76, v[vgprValuC+76], v[vgprValuC+77] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v77, v[vgprValuC+78], v[vgprValuC+79] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[76:77], v12, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
                                                   // GW end
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW4_MBSK_Else:
label_GW_B0_FD0_VW1_MBSK_Else:
label_GW_B0_FD0_VW1_MBSK_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=30 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[52:53], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v32, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v32, v6, v32, s[56:57]               // LDTD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v34, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v34, v6, v34, s[56:57]               // LDTD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v36, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v36, v6, v36, s[56:57]               // LDTD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v38, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v38, v6, v38, s[56:57]               // LDTD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* store after Acc, GSU: 2 */
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v32, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v34, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v36, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
v_cvt_f16_f32 v15, v[vgprValuC+15] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v15, v38, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
                                                   // GW end
s_branch label_GW_End                              // jump to end
label_GW_B1_MBSK:
label_GW_B1_FD0_MBSK:

/* Edge/NonEdge store path check (M): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s52, 15, s[sgprSizeI]                    // s52 = s[sgprSizeI] % 16
s_add_u32 s53, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s53                // wg0 >= nwg0-1 ?
s_cselect_b32 s52, s52, 0                          // set rem
s_cmpk_gt_u32 s52, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_MBSK_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s52, 15, s[sgprSizeJ]                    // s52 = s[sgprSizeJ] % 16
s_add_u32 s53, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s53                // wg1 >= nwg1-1
s_cselect_b32 s52, s52, 0                          // set rem
s_cmpk_gt_u32 s52, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_MBSK_Then       // jump if edges required
label_GW_B1_FD0_VW4_MBSK_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=4 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v13, v2, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
buffer_load_dwordx2 v[80:81], v13, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v12, v3, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
v_accvgpr_read_b32 v[vgprValuC+76], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+77], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+78], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+79], acc3           // copy acc to vreg[3]

/* store after Acc, GSU: 2 */
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 1 - 1 (beta) vscnt(0) (interleaved)
v_fma_mix_f32 v[vgprValuC+76], s[sgprBeta], v80, v[vgprValuC+76] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+77], s[sgprBeta], v80, v[vgprValuC+77] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+78], s[sgprBeta], v81, v[vgprValuC+78] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+79], s[sgprBeta], v81, v[vgprValuC+79] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v76, v[vgprValuC+76], v[vgprValuC+77] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v77, v[vgprValuC+78], v[vgprValuC+79] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[76:77], v12, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
                                                   // GW end
s_branch label_GW_End                              // jump to end
label_GW_B1_FD0_VW4_MBSK_NonEdgeEnd:
label_GW_B1_FD0_VW4_MBSK_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=4 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[52:53], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v11, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[56:57]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[80:81], v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v12, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v12, v6, v12, s[56:57]               // LDTD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+76], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+77], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+78], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+79], acc3           // copy acc to vreg[3]

/* store after Acc, GSU: 2 */
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+76:vgprValuC+76+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+76:vgprValuC+76+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+78:vgprValuC+78+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+78:vgprValuC+78+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+76], s[sgprBeta], v80, v[vgprValuC+76] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+77], s[sgprBeta], v80, v[vgprValuC+77] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+78], s[sgprBeta], v81, v[vgprValuC+78] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+79], s[sgprBeta], v81, v[vgprValuC+79] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v76, v[vgprValuC+76], v[vgprValuC+77] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v77, v[vgprValuC+78], v[vgprValuC+79] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[76:77], v12, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
                                                   // GW end
s_branch label_GW_End                              // jump to end
label_GW_B1_FD0_VW4_MBSK_Else:
label_GW_B1_FD0_VW1_MBSK_Else:
label_GW_B1_FD0_VW1_MBSK_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=14 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[52:53], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v32, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v32, v6, v32, s[56:57]               // LDC clip if OOB. offset
buffer_load_short_d16 v31, v32, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v33, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v33, v6, v33, s[56:57]               // LDTD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v35, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v35, v6, v35, s[56:57]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v34, v35, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v36, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v36, v6, v36, s[56:57]               // LDTD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v38, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v38, v6, v38, s[56:57]               // LDC clip if OOB. offset
buffer_load_short_d16 v37, v38, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v39, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v39, v6, v39, s[56:57]               // LDTD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1

/* edge Protect */
v_add_lshl_u32 v41, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v41, v6, v41, s[56:57]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v40, v41, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v42, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v42, v6, v42, s[56:57]               // LDTD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* store after Acc, GSU: 2 */
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+11], s[sgprBeta], v31, v[vgprValuC+11] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v33, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v34, v[vgprValuC+12] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v36, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v37, v[vgprValuC+13] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v39, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
v_fma_mix_f32 v[vgprValuC+15], s[sgprBeta], v40, v[vgprValuC+15] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v15, v[vgprValuC+15] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v15, v42, s[sgprSrdTD:sgprSrdTD+3], 0 offen offset:0 sc0 sc1 // store TD not StoreRemapVectorWidth
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
                                                   // GW end
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[52:53]                               // addr of next instr
s_add_i32 s54, label_KernelEnd, 4                  // target branch offset
s_add_u32 s52, s52, s54                            // add target branch offset
s_addc_u32 s53, s53, 0                             // add high and carry
s_setpc_b64 s[52:53]                               // branch to label_KernelEnd
label_GSU_3:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s52, 15, s[sgprSizeI]                    // s52 = s[sgprSizeI] % 16
s_add_u32 s53, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s53                // wg0 >= nwg0-1 ?
s_cselect_b32 s52, s52, 0                          // set rem
s_cmpk_gt_u32 s52, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s52, 15, s[sgprSizeJ]                    // s52 = s[sgprSizeJ] % 16
s_add_u32 s53, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s53                // wg1 >= nwg1-1
s_cselect_b32 s52, s52, 0                          // set rem
s_cmpk_gt_u32 s52, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=26 */
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
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=20 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[52:53], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v11, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[56:57]               // LDD clip if OOB. offset
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
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW4_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=52 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[52:53], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v16, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v16, v6, v16, s[56:57]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v17, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v17, v6, v17, s[56:57]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v18, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v18, v6, v18, s[56:57]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v19, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v19, v6, v19, s[56:57]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v18, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v15, v[vgprValuC+15] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v15, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s52, 15, s[sgprSizeI]                    // s52 = s[sgprSizeI] % 16
s_add_u32 s53, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s53                // wg0 >= nwg0-1 ?
s_cselect_b32 s52, s52, 0                          // set rem
s_cmpk_gt_u32 s52, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s52, 15, s[sgprSizeJ]                    // s52 = s[sgprSizeJ] % 16
s_add_u32 s53, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s53                // wg1 >= nwg1-1
s_cselect_b32 s52, s52, 0                          // set rem
s_cmpk_gt_u32 s52, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW4_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW4_GSU1_NonEdge:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v12, v2, v0, 1                      // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0 (multiple bpe)
buffer_load_dwordx2 v[20:21], v12, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
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
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v20, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v20, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v21, v[vgprValuC+18] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v21, v[vgprValuC+19] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW4_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=14 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[52:53], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v11, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[56:57]               // LDC clip if OOB. offset
buffer_load_dwordx2 v[12:13], v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v11, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v11, v6, v11, s[56:57]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+16:vgprValuC+16+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+18:vgprValuC+18+1] op_sel_hi:[0,1,1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v12, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v12, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+18], s[sgprBeta], v13, v[vgprValuC+18] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+19], s[sgprBeta], v13, v[vgprValuC+19] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_pk_f16_f32 v16, v[vgprValuC+16], v[vgprValuC+17] // convert C to fp16 and pack with neighbor
v_cvt_pk_f16_f32 v17, v[vgprValuC+18], v[vgprValuC+19] // convert C to fp16 and pack with neighbor
buffer_store_dwordx2 v[16:17], v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_FD0_VW4_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=36 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v6, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[52:53], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v17, v2, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v17, v6, v17, s[56:57]               // LDC clip if OOB. offset
buffer_load_short_d16 v16, v17, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v17, v3, v0, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v17, v6, v17, s[56:57]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v4, vcc, v0, 1                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v19, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v19, v6, v19, s[56:57]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v18, v19, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v19, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v19, v6, v19, s[56:57]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v4, vcc, v0, 2                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v21, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v21, v6, v21, s[56:57]               // LDC clip if OOB. offset
buffer_load_short_d16 v20, v21, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v21, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v21, v6, v21, s[56:57]               // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v4, vcc, v0, 3                        // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[52:53], v4, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[56:57], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[56:57], s[52:53], s[56:57]             // in0 && in1
v_add_lshl_u32 v23, v2, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v23, v6, v23, s[56:57]               // LDC clip if OOB. offset
buffer_load_short_d16_hi v22, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v23, v3, v4, 1                      // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v23, v6, v23, s[56:57]               // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+12], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+13], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_pk_mul_f32 v[vgprValuC+12:vgprValuC+12+1], s[sgprAlpha:sgprAlpha+1], v[vgprValuC+12:vgprValuC+12+1] op_sel_hi:[0,1,1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+11], s[sgprBeta], v16, v[vgprValuC+11] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v11, v[vgprValuC+11] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v11, v17, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+12], s[sgprBeta], v18, v[vgprValuC+12] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v12, v[vgprValuC+12] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v12, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+13], s[sgprBeta], v20, v[vgprValuC+13] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v13, v[vgprValuC+13] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v13, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+15], s[sgprBeta], v22, v[vgprValuC+15] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v15, v[vgprValuC+15] dst_sel:WORD_0  // convert C to fp16
buffer_store_short v15, v23, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
