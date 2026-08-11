
/******************************************/
/* Begin Kernel                           */
/******************************************/
/* STINKY_TOTAL_INST_BYTES: 9480 */
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.protected Cijk_Alik_Bljk_BBS_BH_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD2_2_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLITDM_MXSFIMS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1_WGMXCC1
.globl Cijk_Alik_Bljk_BBS_BH_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD2_2_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLITDM_MXSFIMS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1_WGMXCC1
.p2align 8
.type Cijk_Alik_Bljk_BBS_BH_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD2_2_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLITDM_MXSFIMS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1_WGMXCC1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_BBS_BH_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD2_2_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLITDM_MXSFIMS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1_WGMXCC1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 224 // vgprs
  .amdhsa_next_free_sgpr 80 // sgprs
  .amdhsa_group_segment_fixed_size 8192 // lds bytes
  .amdhsa_wavefront_size32 1 // 32-thread wavefronts
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_inst_pref_size 74
  .amdhsa_user_sgpr_count 28
  .amdhsa_user_sgpr_kernarg_preload_length 26
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.end_amdhsa_kernel
.text
/* Num VGPR   =224 */
/* Num AccVGPR=0 */
/* Num SGPR   =80 */

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
/* UseSgprForGRO=True */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Cijk_Alik_Bljk_BBS_BH_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD2_2_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLITDM_MXSFIMS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1_WGMXCC1
    .symbol: 'Cijk_Alik_Bljk_BBS_BH_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD2_2_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLITDM_MXSFIMS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1_WGMXCC1.kd'
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
    .group_segment_fixed_size:   8192
    .kernarg_segment_align:      8
    .kernarg_segment_size:       136
    .max_flat_workgroup_size:    32
    .private_segment_fixed_size: 0
    .sgpr_count:                 80
    .sgpr_spill_count:           0
    .vgpr_count:                 224
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_BBS_BH_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD2_2_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLITDM_MXSFIMS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG16_2_1_WGMXCC1:
label_ASM_Start:  /// Main body of the asm kernel
global_prefetch_b8 v0, s[0:1] th:TH_LOAD_RT scope:SCOPE_SE
v_nop
s_setreg_IMM32_b32 hwreg(26,0,2), 2

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
/* ValuC range: [0-8), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 12
.set vgprGlobalReadOffsetA, 8
.set vgprGlobalReadOffsetB, 9
.set vgprLocalReadAddrA, 10
.set vgprLocalReadAddrB, 11
.set vgprSerial, 46

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+18
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+8
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+8
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB, vgprG2LB_BASE+0

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprWaveIdx, 5
.set sgprMulticastMaskA, 6
.set sgprMulticastMaskB, 7
.set sgprArgType, 8
.set sgprGSUSumIdx, 10
.set sgprGSULog2BpeC, 9
.set sgprGSULog2BpeD, 12
.set sgprStaggerU, 13
.set sgprWGM, 14
.set sgprLoopCounterL, 15
.set sgprOrigLoopCounter, 16
.set sgprSrdD, 20
.set sgprSrdC, 24
.set sgprNumWorkGroups0, 17
.set sgprNumWorkGroups1, 18
.set sgprSizesFree, 32
.set sgprSizesSum, 35
.set sgprAddressD, 36
.set sgprAddressC, 38
.set sgprAddressA, 40
.set sgprAddressB, 42
.set sgprStridesD, 44
.set sgprStridesC, 46
.set sgprStridesA, 48
.set sgprStridesB, 50
.set sgprAlpha, 52
.set sgprBeta, 53
.set sgprGSU, 54

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
.set DepthU, 64
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 1
.set SrdShiftLeftB, 1
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0xfffff000

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

/* Global Offset B */

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Init workgroup id from ttmp with cluster remap */
label_Preload_Offset_Start:
s_setreg_IMM32_b32 hwreg(26,0,2), 2
s_and_b32 s55, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s56, s2, 0x1e                           // Get arg type
s_mov_b32 s57, s3                                  // Preload internal args
s_cmp_eq_u32 s56, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance2
s_cmp_eq_u32 s56, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance2:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_mov_b64 s[32:33], s[6:7]                         // move preload data to correct sgpr
s_mov_b64 s[34:35], s[8:9]                         // move preload data to correct sgpr
s_mov_b64 s[36:37], s[10:11]                       // move preload data to correct sgpr
s_mov_b64 s[38:39], s[12:13]                       // move preload data to correct sgpr
s_mov_b64 s[40:41], s[14:15]                       // move preload data to correct sgpr
s_mov_b64 s[42:43], s[16:17]                       // move preload data to correct sgpr
s_mov_b64 s[44:45], s[18:19]                       // move preload data to correct sgpr
s_mov_b64 s[46:47], s[20:21]                       // move preload data to correct sgpr
s_mov_b64 s[48:49], s[22:23]                       // move preload data to correct sgpr
s_mov_b64 s[50:51], s[24:25]                       // move preload data to correct sgpr
s_mov_b64 s[52:53], s[26:27]                       // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
s_mov_b32 s58, s5                                  // Load num of WGs
s_and_b32 s[sgprStaggerU], s57, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s57, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s56
s_mov_b32 m0, 0x2000                               // LDS clamp at 8192 bytes
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero
v_readfirstlane_b32 s57, v[vgprSerial]             // first tId
s_lshr_b32 s[sgprWaveIdx], s57, 5                  // wId=fTid // wavelen
s_getreg_b32 s60, hwreg(28,6,4)                    // cluster_id
s_cmp_eq_u32 s60, 0                                // cluster_id == 0x0 ?
s_cbranch_scc0 label_EnableCluster
s_mov_b32 s[sgprWorkGroup0], ttmp9
s_and_b32 s[sgprWorkGroup1], 0xffff, ttmp7
s_lshr_b32 s[sgprWorkGroup2], ttmp7, 0x10
s_branch label_RemapWorkGroupDone
label_EnableCluster:
s_mov_b32 s60, ttmp6                               // Read TTMP6 register
s_mov_b32 s61, ttmp7                               // Read TTMP7 register,                                                                        cluster_z | cluster_y.
s_bfe_u32 s62, s60, 262148                         // Etract wg_y.
s_bfe_u32 s63, s60, 262160                         // Etract nwg_y. Value is nwg_y - 1
s_add_u32 s63, s63, 1
s_and_b32 s64, s61, 0xffff                         // Etract cluster_y.
s_mul_i32 s[sgprWorkGroup1], s64, s63              // cluster_y * nwg_y
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s62 // WorkGroup1 = (cluster_y * nwg_y) + wg_y
s_bfe_u32 s[sgprWorkGroup2], s61, 1048592          // Etract cluster_z.
s_and_b32 s61, s60, 0xf                            // Etract wg_x.
s_bfe_u32 s63, s60, 262156                         // Etract nwg_x. Value is nwg_x - 1
s_add_u32 s63, s63, 1
s_mul_i32 s[sgprWorkGroup0], ttmp9, s63            // cluster_x * nwg_x
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s61 // WorkGroup0 = (cluster_x * nwg_x) + wg_x
s_bfe_u32 s64, s60, 262152                         // Etract wg_z.
s_bfe_u32 s60, s60, 262164                         // Etract nwg_z. Value is nwg_z - 1
s_add_u32 s60, s60, 1
s_mul_i32 s[sgprWorkGroup2], s[sgprWorkGroup2], s60 // cluster_z * nwg_z
s_add_u32 s[sgprWorkGroup2], s[sgprWorkGroup2], s64 // WorkGroup2 = (cluster_z * nwg_z) + wg_z
label_RemapWorkGroupDone:
/* Calculate multicast mask */
/* reduce multicast mask to real WGs in cluster */
s_lshr_b32 s17, s[sgprSizeI], 4                    // s17 = s[sgprSizeI] / 16
s_and_b32 s66, 15, s[sgprSizeI]                    // s66 = s[sgprSizeI] % 16
s_addc_u32 s17, s17, 0
s_sub_u32 s60, s[sgprWorkGroup0], s61              // clusterBaseX
s_sub_u32 s17, s17, s60                            // tilesM - clusterBaseX
s_min_u32 s17, s17, 2                              // validX = min(.., cx)
s_bfm_b32 s64, s17, 0                              // maskRow bits = (1<<validX)-1
s_lshr_b32 s17, s[sgprSizeJ], 4                    // s17 = s[sgprSizeJ] / 16
s_and_b32 s66, 15, s[sgprSizeJ]                    // s66 = s[sgprSizeJ] % 16
s_addc_u32 s17, s17, 0
s_and_b32 s18, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s17, s17, s18                            // tilesN * GSU (raw y extent)
s_sub_u32 s60, s[sgprWorkGroup1], s62              // clusterBaseY
s_sub_u32 s17, s17, s60                            // tilesN*GSU - clusterBaseY
s_min_u32 s17, s17, 2                              // validY = min(.., cy)
s_mul_i32 s17, s17, 2                              // validY*cx
s_bfm_b32 s60, s17, 0                              // maskCol bits = (1<<(validY*cx))-1
s_mov_b32 s[sgprMulticastMaskA], 0x5               // Setting maskA
s_and_b32 s[sgprMulticastMaskA], s[sgprMulticastMaskA], s60 // reduce to real WGs
s_lshl_b32 s[sgprMulticastMaskA], s[sgprMulticastMaskA], s61 // Setting maskA
s_mul_i32 s62, s62, s63                            // Shift factor: wg_y * nwg_x
s_mov_b32 s[sgprMulticastMaskB], 0x3               // Setting maskB
s_and_b32 s[sgprMulticastMaskB], s[sgprMulticastMaskB], s64 // reduce to real WGs
s_lshl_b32 s[sgprMulticastMaskB], s[sgprMulticastMaskB], s62 // Setting maskB

/* remap workgroup to XCCs */
s_branch label_skip_WGMXCC
s_lshr_b32 s64, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s64, s64                             // Get log(WGMXCC)
s_lshr_b32 s65, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s64, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s61, s58, s64
s_lshl_b32 s61, s61, s64
s_cmp_ge_u32 s[sgprWorkGroup0], s61
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s65, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s61, s[sgprWorkGroup0], s64
s_bfm_b32 s62, s64, 0
s_and_b32 s62, s[sgprWorkGroup0], s62
s_lshr_b32 s63, s58, s64
s_mul_i32 s62, s62, s63
s_add_u32 s[sgprWorkGroup0], s61, s62
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[0:1], s65                          // s61 = s[sgprWorkGroup0] / s65
v_rcp_f64 v[0:1], v[0:1]                           // s61 = s[sgprWorkGroup0] / s65
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s61 = s[sgprWorkGroup0] / s65
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s61 = s[sgprWorkGroup0] / s65
v_cvt_u32_f64 v0, v[0:1]                           // s61 = s[sgprWorkGroup0] / s65
v_mul_lo_u32 v1, v0, s65                           // s61 = s[sgprWorkGroup0] / s65
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s61 = s[sgprWorkGroup0] / s65
v_cmp_ge_u32 vcc_lo, v2, s65                       // s61 = s[sgprWorkGroup0] / s65
s_mov_b32 exec_lo, vcc_lo                          // s61 = s[sgprWorkGroup0] / s65
v_add_nc_u32 v0, v0, 1                             // s61 = s[sgprWorkGroup0] / s65
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s65                           // s61 = s[sgprWorkGroup0] / s65
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s61 = s[sgprWorkGroup0] / s65
v_readfirstlane_b32 s61, v0                        // quotient
v_readfirstlane_b32 s62, v2                        // remainder
s_mul_i32 s61, s61, s65
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s62, s62, s64
/* temp0 = temp0 + temp1 */
s_add_u32 s61, s61, s62
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[0:1], s65                          // s62 = s58 / s65
v_rcp_f64 v[0:1], v[0:1]                           // s62 = s58 / s65
v_cvt_f64_u32 v[2:3], s58                          // s62 = s58 / s65
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s62 = s58 / s65
v_cvt_u32_f64 v0, v[0:1]                           // s62 = s58 / s65
v_mul_lo_u32 v1, v0, s65                           // s62 = s58 / s65
v_sub_nc_u32 v2, s58, v1                           // s62 = s58 / s65
v_cmp_ge_u32 vcc_lo, v2, s65                       // s62 = s58 / s65
s_mov_b32 exec_lo, vcc_lo                          // s62 = s58 / s65
v_add_nc_u32 v0, v0, 1                             // s62 = s58 / s65
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s62, v0                        // quotient
s_mul_i32 s62, s62, s65
s_sub_u32 s63, s58, s62
s_cmp_gt_u32 s[sgprWorkGroup0], s62
s_cselect_b32 s62, s63, s65
s_lshr_b32 s62, s62, s64
s_bfm_b32 s63, s64, 0
s_and_b32 s63, s[sgprWorkGroup0], s63
s_mul_i32 s62, s62, s63
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s61, s62
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s56, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s56, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [12...58) to pool */
/* init: add vgpr [0...8) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 6, v0                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v0, v1, 3, v0                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v1, v2, 3, v1                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 0, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s20, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s20, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 0, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s20, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x800, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

/* local write addresses: first offset b */
v_mov_b32 v2, MT0                                  // set MT0 into sgpr
v_mov_b32 v1, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
v_mov_b32 v2, MT1                                  // set MT1 into sgpr
v_mov_b32 v1, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v0      // set back to numWorkGroup0
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v0      // set back to numWorkGroup1
s_wait_kmcnt 0                                     // wait for -16/0 bytes of kern args

/* Early stop padded work-groups in a boundary cluster (grid rounded up to ClusterDim) */
s_cmp_ge_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // padded if WorkGroup0 >= tilesM
s_cbranch_scc1 label_ClusterPad_EarlyStop
s_and_b32 s20, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s20, s[sgprNumWorkGroups1], s20          // tilesN * GSU
s_cmp_ge_u32 s[sgprWorkGroup1], s20                // padded if WorkGroup1 >= tilesN*GSU
s_cbranch_scc1 label_ClusterPad_EarlyStop
s_branch label_ClusterPad_NoEarlyStop
label_ClusterPad_EarlyStop:
s_endpgm                                           // padded work-group: exit before any load/barrier
label_ClusterPad_NoEarlyStop:
s_branch label_MultiGemmEnd                        // Already using 3D WorkGroups, skip remap

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s20, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s21, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s20, s20, s21
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s20                              // s20 = s[sgprWorkGroup0] / s20
v_rcp_iflag_f32 v0, v0                             // s20 = s[sgprWorkGroup0] / s20
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s20 = s[sgprWorkGroup0] / s20
v_mul_f32 v0, v0, v1                               // s20 = s[sgprWorkGroup0] / s20
v_cvt_u32_f32 v0, v0                               // s20 = s[sgprWorkGroup0] / s20
v_mul_u32_u24 v1, v0, s20                          // s20 = s[sgprWorkGroup0] / s20
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s20 = s[sgprWorkGroup0] / s20
v_cmp_eq_u32 vcc_lo, v1, s20                       // s20 = s[sgprWorkGroup0] / s20
s_mov_b32 exec_lo, vcc_lo                          // s20 = s[sgprWorkGroup0] / s20
v_add_nc_u32 v0, 1, v0                             // s20 = s[sgprWorkGroup0] / s20
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s20                       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s20, v0                        // quotient
s_mov_b32 s[sgprWorkGroup2], s20
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s20, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s20, s20, s[sgprWorkGroup2]
s_mul_i32 s20, s20, s21
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s20
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v0, s[sgprNumWorkGroups0]            // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v0, v0                             // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v0, v0, v1                               // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v0, v0                               // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups0]        // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo                          // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v0, 1, v0                             // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s20, v0                        // quotient
s_mov_b32 s[sgprWorkGroup1], s20
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s20, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s20
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s18, 88                                  // KernArgAddressOffset
s_mul_i32 s26, s55, 4
s_mov_b64 s[20:21], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s18, 228
s_mov_b32 s26, 0
s_mov_b64 s[20:21], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s17, 1
s_mov_b32 s27, 0
s_load_b128 s[32:35], s[20:21], s26
s_mov_b32 s19, 1
s_cmp_eq_u32 s55, s19                              // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_wait_kmcnt 0
s_lshr_b32 s24, s32, 4                             // s24 = s32 / 16
s_and_b32 s22, 15, s32                             // s22 = s32 % 16
s_addc_u32 s24, s24, 0
s_lshr_b32 s25, s33, 4                             // s25 = s33 / 16
s_and_b32 s22, 15, s33                             // s22 = s33 % 16
s_addc_u32 s25, s25, 0
s_mul_i32 s24, s24, s25
s_mul_i32 s24, s24, s34
s_and_b32 s25, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s24, s24, s25
s_add_u32 s27, s27, s24
s_cmp_lt_u32 s[sgprWorkGroup0], s27
s_cbranch_scc1 label_FOUND
s_add_u32 s26, s26, s18
s_load_b128 s[32:35], s[20:21], s26
s_add_u32 s17, s17, 1
s_cmp_lt_u32 s17, s55
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_wait_kmcnt 0
s_lshr_b32 s24, s32, 4                             // s24 = s32 / 16
s_and_b32 s22, 15, s32                             // s22 = s32 % 16
s_addc_u32 s24, s24, 0
s_lshr_b32 s25, s33, 4                             // s25 = s33 / 16
s_and_b32 s22, 15, s33                             // s22 = s33 % 16
s_addc_u32 s25, s25, 0
s_mul_i32 s24, s24, s25
s_mul_i32 s24, s24, s34
s_and_b32 s20, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s24, s24, s20
s_add_u32 s27, s27, s24

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s21, s17, 1
s_sub_u32 s20, s27, s24
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s20
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s55, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s21, s21, 88                             // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s21
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[36:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[52:53], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s21, s21, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s21
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b512 s[36:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b32 s52, s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
// Read Beta
s_load_b32 s53, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
label_LoadExternalStructEnd:
/* init: add vgpr [12...58) to pool */
/* init: add vgpr [0...8) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 6, v0                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v0, v1, 3, v0                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 6, v1                            // 1. N offset: nOffset = nIdx * nStride(64)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v1, v2, 3, v1                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 0, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s20, 64                                  // LSU offset: stride = lsuStride(64) when umlds==True
v_mul_lo_u32 v2, s20, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA] //  (multiple bpe)

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 0, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(64) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s20, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB] //  (multiple bpe)

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x800, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

/* local write addresses: first offset b */
v_mov_b32 v2, MT0                                  // set MT0 into sgpr
v_mov_b32 v1, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
v_mov_b32 v2, MT1                                  // set MT1 into sgpr
v_mov_b32 v1, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v0      // set back to numWorkGroup0
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v0      // set back to numWorkGroup1
s_wait_kmcnt 0                                     // wait for -16/0 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* Early stop padded work-groups in a boundary cluster (grid rounded up to ClusterDim) */
s_cmp_ge_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // padded if WorkGroup0 >= tilesM
s_cbranch_scc1 label_ClusterPad_EarlyStop_1
s_and_b32 s19, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s19, s[sgprNumWorkGroups1], s19          // tilesN * GSU
s_cmp_ge_u32 s[sgprWorkGroup1], s19                // padded if WorkGroup1 >= tilesN*GSU
s_cbranch_scc1 label_ClusterPad_EarlyStop_1
s_branch label_ClusterPad_NoEarlyStop_1
label_ClusterPad_EarlyStop_1:
s_endpgm                                           // padded work-group: exit before any load/barrier
label_ClusterPad_NoEarlyStop_1:
s_branch label_MultiGemmEnd                        // Already using 3D WorkGroups, skip remap

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s20, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s21, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s20, s20, s21
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s20                              // s20 = s[sgprWorkGroup0] / s20
v_rcp_iflag_f32 v0, v0                             // s20 = s[sgprWorkGroup0] / s20
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s20 = s[sgprWorkGroup0] / s20
v_mul_f32 v0, v0, v1                               // s20 = s[sgprWorkGroup0] / s20
v_cvt_u32_f32 v0, v0                               // s20 = s[sgprWorkGroup0] / s20
v_mul_u32_u24 v1, v0, s20                          // s20 = s[sgprWorkGroup0] / s20
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s20 = s[sgprWorkGroup0] / s20
v_cmp_eq_u32 vcc_lo, v1, s20                       // s20 = s[sgprWorkGroup0] / s20
s_mov_b32 exec_lo, vcc_lo                          // s20 = s[sgprWorkGroup0] / s20
v_add_nc_u32 v0, 1, v0                             // s20 = s[sgprWorkGroup0] / s20
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s20                       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s20, v0                        // quotient
s_mov_b32 s[sgprWorkGroup2], s20
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s20, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s20, s20, s[sgprWorkGroup2]
s_mul_i32 s20, s20, s21
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s20
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v0, s[sgprNumWorkGroups0]            // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v0, v0                             // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v0, v0, v1                               // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v0, v0                               // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups0]        // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo                          // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v0, 1, v0                             // s20 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s20, v0                        // quotient
s_mov_b32 s[sgprWorkGroup1], s20
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s20, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s20

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprtdmAGroup0, 28
.set sgprtdmAGroup1, 56
.set sgprtdmBGroup0, 64
.set sgprtdmBGroup1, 68
.set sgprGlobalReadIncsA, 19
.set sgprGlobalReadIncsB, 55
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
label_Skip_Address_Prepad_For_Pointer_Array:  /// Skip pre-padding of address for pointer array case

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:
s_setreg_IMM32_b32 hwreg(26,4,1), 1                // Disable WMMA arb stall
s_and_b32 s19, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s19, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
s_and_b32 s19, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s19, s[sgprGSU], 0x3fff                  // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s19                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_rcp_iflag_f32 v0, v0                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_cvt_f32_u32 v1, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_mul_f32 v0, v0, v1                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_cvt_u32_f32 v0, v0                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_mul_u32_u24 v1, v0, s19                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_cmp_eq_u32 vcc_lo, v1, s19                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_add_nc_u32 v0, 1, v0                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s19
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s19
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s19                       // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s19                          // re-calculate remainder
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v1           // remainder
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s[sgprNumWorkGroups1]            // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v0, v0                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_f32_u32 v1, s[sgprWorkGroup1]                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v0, v0, v1                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_u32_f32 v0, v0                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups1]        // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprNumWorkGroups1]     // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_nc_u32 v0, 1, v0                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v1, 0                                    // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups1]     // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups1]        // re-calculate remainder
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprGSUSumIdx], v0           // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v1          // remainder
label_GSUWGMRR_End:
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 1
label_GSU_End:
s_mov_b32 s19, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s19, s19                            // Restore WGM
s_branch label_WGM
s_cmp_gt_i32 s19, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s19, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s19, s19                                 // abs(WGM)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[0:1], s19                          // s22 = s[sgprWorkGroup0] / s19
v_rcp_f64 v[0:1], v[0:1]                           // s22 = s[sgprWorkGroup0] / s19
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s22 = s[sgprWorkGroup0] / s19
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s22 = s[sgprWorkGroup0] / s19
v_cvt_u32_f64 v0, v[0:1]                           // s22 = s[sgprWorkGroup0] / s19
v_mul_lo_u32 v1, v0, s19                           // s22 = s[sgprWorkGroup0] / s19
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s22 = s[sgprWorkGroup0] / s19
v_cmp_ge_u32 vcc_lo, v2, s19                       // s22 = s[sgprWorkGroup0] / s19
s_mov_b32 exec_lo, vcc_lo                          // s22 = s[sgprWorkGroup0] / s19
v_add_nc_u32 v0, v0, 1                             // s22 = s[sgprWorkGroup0] / s19
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s22, v0                        // quotient
s_mul_i32 s23, s22, s19                            // quotient * non-magic divisor
s_sub_u32 s23, s[sgprWorkGroup0], s23              // WorkGroup0=remainder
s_mul_i32 s23, s23, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s23, s23, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[0:1], s19                          // s20 = s[sgprNumWorkGroups0] / s19
v_rcp_f64 v[0:1], v[0:1]                           // s20 = s[sgprNumWorkGroups0] / s19
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups0]        // s20 = s[sgprNumWorkGroups0] / s19
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s20 = s[sgprNumWorkGroups0] / s19
v_cvt_u32_f64 v0, v[0:1]                           // s20 = s[sgprNumWorkGroups0] / s19
v_mul_lo_u32 v1, v0, s19                           // s20 = s[sgprNumWorkGroups0] / s19
v_sub_nc_u32 v2, s[sgprNumWorkGroups0], v1         // s20 = s[sgprNumWorkGroups0] / s19
v_cmp_ge_u32 vcc_lo, v2, s19                       // s20 = s[sgprNumWorkGroups0] / s19
s_mov_b32 exec_lo, vcc_lo                          // s20 = s[sgprNumWorkGroups0] / s19
v_add_nc_u32 v0, v0, 1                             // s20 = s[sgprNumWorkGroups0] / s19
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s20, v0                        // quotient
s_mul_i32 s21, s19, s20                            // quotient * non-magic divisor
s_sub_u32 s21, s[sgprNumWorkGroups0], s21          // NumWorkGroups0=remainder
s_cmp_eq_u32 s21, 0                                // remainder == 0 ?
s_cmov_b32 s21, s19                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s22, s20                              // blockId >= numFullBlocks ?
s_cselect_b32 s20, s21, s19
v_cvt_f64_u32 v[0:1], s20                          // s[sgprWorkGroup1] = s23 / s20
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup1] = s23 / s20
v_cvt_f64_u32 v[2:3], s23                          // s[sgprWorkGroup1] = s23 / s20
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup1] = s23 / s20
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup1] = s23 / s20
v_mul_lo_u32 v1, v0, s20                           // s[sgprWorkGroup1] = s23 / s20
v_sub_nc_u32 v2, s23, v1                           // s[sgprWorkGroup1] = s23 / s20
v_cmp_ge_u32 vcc_lo, v2, s20                       // s[sgprWorkGroup1] = s23 / s20
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s23 / s20
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup1] = s23 / s20
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s20                           // s[sgprWorkGroup1] = s23 / s20
v_sub_nc_u32 v2, s23, v1                           // s[sgprWorkGroup1] = s23 / s20
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v2          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s20 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s23, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s22, s22, s19                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s22 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s19, s19                                 // WGM
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[0:1], s19                          // s22 = s[sgprWorkGroup1] / s19
v_rcp_f64 v[0:1], v[0:1]                           // s22 = s[sgprWorkGroup1] / s19
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup1]            // s22 = s[sgprWorkGroup1] / s19
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s22 = s[sgprWorkGroup1] / s19
v_cvt_u32_f64 v0, v[0:1]                           // s22 = s[sgprWorkGroup1] / s19
v_mul_lo_u32 v1, v0, s19                           // s22 = s[sgprWorkGroup1] / s19
v_sub_nc_u32 v2, s[sgprWorkGroup1], v1             // s22 = s[sgprWorkGroup1] / s19
v_cmp_ge_u32 vcc_lo, v2, s19                       // s22 = s[sgprWorkGroup1] / s19
s_mov_b32 exec_lo, vcc_lo                          // s22 = s[sgprWorkGroup1] / s19
v_add_nc_u32 v0, v0, 1                             // s22 = s[sgprWorkGroup1] / s19
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s22, v0                        // quotient
s_mul_i32 s23, s22, s19                            // quotient * non-magic divisor
s_sub_u32 s23, s[sgprWorkGroup1], s23              // WorkGroup1=remainder
s_mul_i32 s23, s23, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s23, s23, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[0:1], s19                          // s20 = s[sgprNumWorkGroups1] / s19
v_rcp_f64 v[0:1], v[0:1]                           // s20 = s[sgprNumWorkGroups1] / s19
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups1]        // s20 = s[sgprNumWorkGroups1] / s19
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s20 = s[sgprNumWorkGroups1] / s19
v_cvt_u32_f64 v0, v[0:1]                           // s20 = s[sgprNumWorkGroups1] / s19
v_mul_lo_u32 v1, v0, s19                           // s20 = s[sgprNumWorkGroups1] / s19
v_sub_nc_u32 v2, s[sgprNumWorkGroups1], v1         // s20 = s[sgprNumWorkGroups1] / s19
v_cmp_ge_u32 vcc_lo, v2, s19                       // s20 = s[sgprNumWorkGroups1] / s19
s_mov_b32 exec_lo, vcc_lo                          // s20 = s[sgprNumWorkGroups1] / s19
v_add_nc_u32 v0, v0, 1                             // s20 = s[sgprNumWorkGroups1] / s19
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s20, v0                        // quotient
s_mul_i32 s21, s19, s20                            // quotient * non-magic divisor
s_sub_u32 s21, s[sgprNumWorkGroups1], s21          // NumWorkGroups1=remainder
s_cmp_eq_u32 s21, 0                                // remainder == 0 ?
s_cmov_b32 s21, s19                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s22, s20                              // blockId >= numFullBlocks ?
s_cselect_b32 s20, s21, s19
v_cvt_f64_u32 v[0:1], s20                          // s[sgprWorkGroup0] = s23 / s20
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup0] = s23 / s20
v_cvt_f64_u32 v[2:3], s23                          // s[sgprWorkGroup0] = s23 / s20
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup0] = s23 / s20
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup0] = s23 / s20
v_mul_lo_u32 v1, v0, s20                           // s[sgprWorkGroup0] = s23 / s20
v_sub_nc_u32 v2, s23, v1                           // s[sgprWorkGroup0] = s23 / s20
v_cmp_ge_u32 vcc_lo, v2, s20                       // s[sgprWorkGroup0] = s23 / s20
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s23 / s20
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup0] = s23 / s20
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s20                           // s[sgprWorkGroup0] = s23 / s20
v_sub_nc_u32 v2, s23, v1                           // s[sgprWorkGroup0] = s23 / s20
v_readfirstlane_b32 s[sgprWorkGroup0], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s20 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s23, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s22, s22, s19                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s22 // wg1 += blockId * WGM
label_WGM:
s_mov_b64 s[20:21], 0
s_mul_i32 s20, s[sgprStrideA0I], 32                // stride * MT(16) * bpe(2.0)
s_mul_hi_u32 s21, s20, s[sgprWorkGroup0]           // *= wgId
s_mul_i32 s20, s20, s[sgprWorkGroup0]              // *= wgId
s_mul_i32 s22, s[sgprWaveIdx], 32                  // woffset = wId * mt // numWaves * bpe // tdmSplit
s_mul_i32 s22, s22, s[sgprStrideA0I]               // woffset *= stride
s_add_u32 s20, s20, s22                            // += woffset
s_addc_u32 s21, s21, 0                             // += woffset carry
s_mul_i32 s22, s[sgprGSUSumIdx], 128               // gsuOffset = GSUSumIdx * DepthU(64) * bpe(2.0)
s_add_u32 s20, s20, s22                            // += gsuOffset
s_add_u32 s[sgprAddressA], s20, s[sgprAddressA]    // += baseAddr(lo)
s_addc_u32 s[sgprAddressA+1], s21, s[sgprAddressA+1] // += baseAddr(hi)
s_mul_hi_u32 s21, s[sgprStrideAK], s[sgprWorkGroup2] // Batch: Stride*WG
s_mul_i32 s20, s[sgprStrideAK], s[sgprWorkGroup2]  // Batch: Stride*WG
s_lshl_b64 s[20:21], s[20:21], 1                   // scale by bpe (multiple bpe)
s_add_u32 s[sgprAddressA], s20, s[sgprAddressA]    // += baseAddr(lo)
s_addc_u32 s[sgprAddressA+1], s21, s[sgprAddressA+1] // += baseAddr(hi)
s_mov_b32 s[sgprtdmAGroup0+0], 1
s_mov_b32 s[sgprtdmAGroup0+1], 0
s_mov_b32 s[sgprtdmAGroup0+2], 0
s_mov_b32 s[sgprtdmAGroup0+3], 0
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
s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0x10000 // Set data_size to 1
s_mov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprAddressA:sgprAddressA+1]
s_or_b32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0x80000000 // set type field to 2(image)
s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]
s_mul_i32 s20, s[sgprWaveIdx], 2048                // woffset = WaveIdx * (mt // numWaves * du * bpe // dim1Divisor)
s_add_u32 s20, s20, 0                              // ldsOffset = woffset + ldsConstOffset
s_mov_b32 s[sgprtdmAGroup0+1], s20
s_and_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0xfff7ffff // clear iterate_enable (D# Group 1 bit 19)
s_and_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], 0xffff
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff0000
s_lshl_b32 s19, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], s19
s_lshr_b32 s19, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s19
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff
s_and_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0xffff0000
s_lshl_b32 s19, s[sgprSizeI], 0x10
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s19
s_lshr_b32 s19, s[sgprSizeI], 0x10
s_or_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], s19
s_and_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0xffff
s_or_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0x400000 // set tile0 to 64
s_and_b32 s[sgprtdmAGroup1+4], s[sgprtdmAGroup1+4], 0xffff0000
s_or_b32 s[sgprtdmAGroup1+4], s[sgprtdmAGroup1+4], 0x10 // set tile1 to 16
s_mov_b32 s[sgprtdmAGroup1+5], s[sgprStrideA0I]
s_mov_b64 s[20:21], 0
s_mul_i32 s20, s[sgprStrideB1J], 32                // stride * MT(16) * bpe(2.0)
s_mul_hi_u32 s21, s20, s[sgprWorkGroup1]           // *= wgId
s_mul_i32 s20, s20, s[sgprWorkGroup1]              // *= wgId
s_mul_i32 s22, s[sgprWaveIdx], 32                  // woffset = wId * mt // numWaves * bpe // tdmSplit
s_mul_i32 s22, s22, s[sgprStrideB1J]               // woffset *= stride
s_add_u32 s20, s20, s22                            // += woffset
s_addc_u32 s21, s21, 0                             // += woffset carry
s_mul_i32 s22, s[sgprGSUSumIdx], 128               // gsuOffset = GSUSumIdx * DepthU(64) * bpe(2.0)
s_add_u32 s20, s20, s22                            // += gsuOffset
s_add_u32 s[sgprAddressB], s20, s[sgprAddressB]    // += baseAddr(lo)
s_addc_u32 s[sgprAddressB+1], s21, s[sgprAddressB+1] // += baseAddr(hi)
s_mul_hi_u32 s21, s[sgprStrideBK], s[sgprWorkGroup2] // Batch: Stride*WG
s_mul_i32 s20, s[sgprStrideBK], s[sgprWorkGroup2]  // Batch: Stride*WG
s_lshl_b64 s[20:21], s[20:21], 1                   // scale by bpe (multiple bpe)
s_add_u32 s[sgprAddressB], s20, s[sgprAddressB]    // += baseAddr(lo)
s_addc_u32 s[sgprAddressB+1], s21, s[sgprAddressB+1] // += baseAddr(hi)
s_mov_b32 s[sgprtdmBGroup0+0], 1
s_mov_b32 s[sgprtdmBGroup0+1], 0
s_mov_b32 s[sgprtdmBGroup0+2], 0
s_mov_b32 s[sgprtdmBGroup0+3], 0
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
s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0x10000 // Set data_size to 1
s_mov_b64 s[sgprtdmBGroup0+2:sgprtdmBGroup0+2+1], s[sgprAddressB:sgprAddressB+1]
s_or_b32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0x80000000 // set type field to 2(image)
s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]
s_mul_i32 s20, s[sgprWaveIdx], 2048                // woffset = WaveIdx * (mt // numWaves * du * bpe // dim1Divisor)
s_add_u32 s20, s20, 2048                           // ldsOffset = woffset + ldsConstOffset
s_mov_b32 s[sgprtdmBGroup0+1], s20
s_and_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0xfff7ffff // clear iterate_enable (D# Group 1 bit 19)
s_and_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], 0xffff
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff0000
s_lshl_b32 s19, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], s19
s_lshr_b32 s19, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s19
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff0000
s_lshl_b32 s19, s[sgprSizeJ], 0x10
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s19
s_lshr_b32 s19, s[sgprSizeJ], 0x10
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], s19
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0x400000 // set tile0 to 64
s_and_b32 s[sgprtdmBGroup1+4], s[sgprtdmBGroup1+4], 0xffff0000
s_or_b32 s[sgprtdmBGroup1+4], s[sgprtdmBGroup1+4], 0x10 // set tile1 to 16
s_mov_b32 s[sgprtdmBGroup1+5], s[sgprStrideB1J]
.set sgprMulticastMaskA, UNDEF
.set sgprMulticastMaskB, UNDEF
s_and_b32 s7, s[sgprGSU], 0x3fff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsA+0], 128            // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s7, s7, s[sgprGlobalReadIncsA+0]         // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s6, s[sgprGSU], 0x8000                   // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], s7 // incrA (unrollIdx)
s_and_b32 s7, s[sgprGSU], 0x3fff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsB+0], 128            // GSU*DepthU*Bpe*MI_dim(1)
s_mul_i32 s7, s7, s[sgprGlobalReadIncsB+0]         // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s6, s[sgprGSU], 0x8000                   // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s7 // incrB (unrollIdx)
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 6 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 64
s_and_b32 s20, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s20, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v0, 1, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_add_u32 s20, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s20                // numIterMyWg++ if needed
label_GSU_1:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // gate: only signal when LoopCounterL != 0
s_cbranch_scc1 label_skipCBPreSignal_LCL_PYP6OcN8X0CHgOIq // skip cluster barrier when LoopCounterL == 0
s_barrier_signal -1
s_barrier_wait -1                                  // sync workgroup before cluster signal
s_cmp_eq_u32 s[sgprWaveIdx], 0                     // Check for waveID 0
s_cbranch_scc0 label_skipCBPreSignal_ASQcRDPcobkn1egt // Execute cluster barrier signal for waveID 0
s_barrier_signal -3                                // cluster_barrier signal
label_skipCBPreSignal_ASQcRDPcobkn1egt:
label_skipCBPreSignal_LCL_PYP6OcN8X0CHgOIq:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
s_barrier_wait -3                                  // cluster_barrier wait
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS0
tensor_load_to_lds s[sgprtdmBGroup0:sgprtdmBGroup0+3], s[sgprtdmBGroup1:sgprtdmBGroup1+7] // sync LDS0
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprGlobalReadIncsA] // TDM increment lo
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0 // TDM increment hi (carry)
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s[sgprGlobalReadIncsB] // TDM increment lo
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0 // TDM increment hi (carry)
label_ShadowInitStart:
s_and_b32 s6, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s6, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
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
s_and_b32 s6, s[sgprSrdD+2], 127
s_lshl_b32 s6, s6, 25
s_and_b32 s[sgprSrdD+1], s[sgprSrdD+1], 33554431
s_or_b32 s[sgprSrdD+1], s[sgprSrdD+1], s6
s_lshr_b32 s[sgprSrdD+2], s[sgprSrdD+2], 7
s_and_b32 s6, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s6, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
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
s_and_b32 s6, s[sgprSrdC+2], 127
s_lshl_b32 s6, s6, 25
s_and_b32 s[sgprSrdC+1], s[sgprSrdC+1], 33554431
s_or_b32 s[sgprSrdC+1], s[sgprSrdC+1], s6
s_lshr_b32 s[sgprSrdC+2], s[sgprSrdC+2], 7
s_mul_i32 s78, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s77, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_hi_u32 s77, s78, s[sgprStrideC1J]            // ScaleC s78 by Stride
s_mul_i32 s76, s78, s[sgprStrideC1J]               // ScaleC s78 by Stride
s_lshl_b64 s[76:77], s[76:77], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s76        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s77       // add hi to SRD
s_and_b32 s77, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_hi_u32 s77, s78, s[sgprStrideD1J]            // ScaleD s78 by Stride
s_mul_i32 s76, s78, s[sgprStrideD1J]               // ScaleD s78 by Stride
s_lshl_b64 s[76:77], s[76:77], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s76        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s77       // add hi to SRD
s_and_b32 s77, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s77, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s77, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s76, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[76:77], s[76:77], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s76        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s77       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s76, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s76, s76, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s77, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[78:79], s[76:77], 0                   // Load the Matrix Address in the Pointer Array
s_wait_kmcnt 0
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s78        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s79       // Offsetting within the Batch Matrix [Higher half of address]
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s77, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s77, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s77, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s76, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[76:77], s[76:77], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s76        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s77       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s76, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s76, s76, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s77, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[78:79], s[76:77], 0                   // Load the Matrix Address in the Pointer Array
s_wait_kmcnt 0
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s78        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s79       // Offsetting within the Batch Matrix [Higher half of address]
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD
s_and_b32 s6, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s6, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
s_mul_hi_u32 s77, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s76, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s78, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s78, s78, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s79, s78, s[sgprStrideC1J]            // Free1
s_mul_i32 s78, s78, s[sgprStrideC1J]               // Free1
s_add_u32 s76, s76, s78                            // Free1
s_addc_u32 s77, s77, s79                           // Free1
s_sub_u32 s78, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s78, s78, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s79, s78, s[sgprStrideCK]             // Free2
s_mul_i32 s78, s78, s[sgprStrideCK]                // Free2
s_add_u32 s76, s76, s78                            // Free2
s_addc_u32 s77, s77, s79                           // Free2
s_lshl_b64 s[76:77], s[76:77], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s76        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s77       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc0 label_skipInitCVmov                 // skip v_mov initC (WMMA initC will run in main loop)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v[vgprValuC+0], 0                        // initC
v_mov_b32 v[vgprValuC+1], 0                        // initC
v_mov_b32 v[vgprValuC+2], 0                        // initC
v_mov_b32 v[vgprValuC+3], 0                        // initC
v_mov_b32 v[vgprValuC+4], 0                        // initC
v_mov_b32 v[vgprValuC+5], 0                        // initC
v_mov_b32 v[vgprValuC+6], 0                        // initC
v_mov_b32 v[vgprValuC+7], 0                        // initC
label_skipInitCVmov:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc0 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc1
s_getpc_b64 s[76:77]                               // addr of next instr
s_add_i32 s78, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s76, s76, s78                            // add target branch offset
s_addc_u32 s77, s77, 0                             // add high and carry
s_setpc_b64 s[76:77]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_T8JHFHKM7BO5OHXW:
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x1000
s_xor_b32 s[sgprtdmBGroup0+1], s[sgprtdmBGroup0+1], 0x1000
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS1
tensor_load_to_lds s[sgprtdmBGroup0:sgprtdmBGroup0+3], s[sgprtdmBGroup1:sgprtdmBGroup1+7] // sync LDS1
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
s_wait_tensorcnt 0
label_skipPGR2_2:
s_wait_tensorcnt 2
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
s_wait_alu depctr_va_vdst(14)
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL == 1 (PGR>=2, not Suppress: single-loop -> toPGR1)
s_cbranch_scc1 label_toPGR1                        // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
label_InitCIterWmma_label_LoopBeginL_0:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x1000
s_xor_b32 s[sgprtdmBGroup0+1], s[sgprtdmBGroup0+1], 0x1000
s_wait_alu depctr_vm_vsrc(2)
v_xor_b32 v[vgprLocalReadAddrA], 0x1000, v[vgprLocalReadAddrA] // swap Red Blk
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrB], 0x1000, v[vgprLocalReadAddrB] // swap Red Blk
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprGlobalReadIncsA] // TDM increment lo
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0 // TDM increment hi (carry)
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s[sgprGlobalReadIncsB] // TDM increment lo
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0 // TDM increment hi (carry)
s_wait_dscnt 4
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], 0 matrix_a_reuse matrix_b_reuse // left value = v[0+0:7+0]
s_branch label_InitCIterWmma_target_0
.align 16
label_LoopBeginL:
s_nop 0                                            // <This is 1-cycle>
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0 <This is 2-cycle>
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 3-cycle>
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 5-cycle>
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 7-cycle>
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 9-cycle>
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x1000 // <This is 10-cycle>
s_xor_b32 s[sgprtdmBGroup0+1], s[sgprtdmBGroup0+1], 0x1000 // <This is 11-cycle>
s_wait_alu depctr_vm_vsrc(2)                       // <This is 12-cycle>
v_xor_b32 v[vgprLocalReadAddrA], 0x1000, v[vgprLocalReadAddrA] // swap Red Blk <This is 13-cycle>
s_wait_alu depctr_vm_vsrc(0)                       // <This is 14-cycle>
v_xor_b32 v[vgprLocalReadAddrB], 0x1000, v[vgprLocalReadAddrB] // swap Red Blk <This is 15-cycle>
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprGlobalReadIncsA] // TDM increment lo <This is 16-cycle>
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0 // TDM increment hi (carry) <This is 17-cycle>
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s[sgprGlobalReadIncsB] // TDM increment lo <This is 18-cycle>
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0 // TDM increment hi (carry) <This is 19-cycle>
s_wait_dscnt 4                                     // <This is 20-cycle>
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0] <This is 21-cycle>
label_InitCIterWmma_target_0:
s_wait_tensorcnt 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
s_wait_alu depctr_va_vdst(1)
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_wait_alu depctr_va_vdst(0)
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_wait_dscnt 4
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS0
tensor_load_to_lds s[sgprtdmBGroup0:sgprtdmBGroup0+3], s[sgprtdmBGroup1:sgprtdmBGroup1+7] // sync LDS0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:
s_wait_tensorcnt 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x1000
s_xor_b32 s[sgprtdmBGroup0+1], s[sgprtdmBGroup0+1], 0x1000
s_wait_alu depctr_vm_vsrc(2)
v_xor_b32 v[vgprLocalReadAddrA], 0x1000, v[vgprLocalReadAddrA] // swap Red Blk
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrB], 0x1000, v[vgprLocalReadAddrB] // swap Red Blk
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprGlobalReadIncsA] // TDM increment lo
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0 // TDM increment hi (carry)
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s[sgprGlobalReadIncsB] // TDM increment lo
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0 // TDM increment hi (carry)
s_wait_dscnt 4
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
s_wait_alu depctr_va_vdst(1)
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_alu depctr_va_vdst(0)
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 4
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
label_toPGR1:
s_and_b32 s6, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s6, 1                                 // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1
s_mov_b32 s6, 0
s_cmp_eq_u32 s[sgprBeta], s6                       // Beta == 0
s_cbranch_scc0 label_OptNLL_End                    // Branch if Beta is not zero
s_cmp_eq_u32 s[sgprAlpha], 1.0                     // Alpha == 1.0 ?
s_cbranch_scc0 label_OptNLL_End                    // branch if alpha != 1
s_and_b32 s76, 15, s[sgprSizeI]                    // s76 = s[sgprSizeI] % 16
s_add_u32 s77, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s77                // wg0 >= nwg0-1 ?
s_cselect_b32 s76, s76, 0                          // set rem
s_mov_b32 s6, 0
s_cmp_gt_u32 s76, s6                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s76, 15, s[sgprSizeJ]                    // s76 = s[sgprSizeJ] % 16
s_add_u32 s77, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s77                // wg1 >= nwg1-1
s_cselect_b32 s76, s76, 0                          // set rem
s_mov_b32 s6, 0
s_cmp_gt_u32 s76, s6                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s77, 63, s[sgprSizesSum+0]               // s77 = s[sgprSizesSum+0] % 64
s_cmp_eq_u32 s77, 0                                // numIterL == 0
s_cbranch_scc0 label_OptNLL_End                    // skip if tail loop required
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_alu depctr_vm_vsrc(2)
v_xor_b32 v[vgprLocalReadAddrA], 0x1000, v[vgprLocalReadAddrA] // swap Red Blk
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrB], 0x1000, v[vgprLocalReadAddrB] // swap Red Blk
s_wait_dscnt 4
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
s_wait_dscnt 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
label_toPGR1end_OptNLL:
label_Summation_End_OptNLL:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v12, 5, v[vgprSerial]                // 12 = Serial / 32
v_lshrrev_b32 v13, 0, v12                          // 13 = 12 / 1
v_lshlrev_b32 v9, 4, v13                           // wave coordination offset 1
v_and_b32 v13, 15, v[vgprSerial]                   // v13 = v[vgprSerial] % 16
v_add_lshl_u32 v9, v13, v9, 0                      // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v10, v9, s[sgprStrideC1J]             //  offset 1
v_mul_lo_u32 v11, v9, s[sgprStrideD1J]             //  offset 1
v_and_b32 v13, 0, v12                              // v13 = v12 % 1
v_lshlrev_b32 v13, 4, v13                          // wave coordination offset 0
v_and_b32 v8, 31, v[vgprSerial]                    // v8 = v[vgprSerial] % 32
v_lshrrev_b32 v8, 4, v8                            // 8 = 8 / 16
v_lshlrev_b32 v8, 3, v8                            // thread0 * continuous_output
v_add_lshl_u32 v8, v13, v8, 0                      // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s6, 16, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v8, s6, v8                            // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s6, 16, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v9, s6, v9                            // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
label_GW_B0_OptNLL_MB:
label_GW_B0_FD0_OptNLL_MB:
label_GW_B0_FD0_VW8_OptNLL_MB_Then:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_add_lshl_u32 v19, v11, v8, 1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8 (multiple bpe)
v_mov_b32 v[vgprValuC+24], v[vgprValuC+0]          // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+25], v[vgprValuC+1]          // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+26], v[vgprValuC+2]          // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+27], v[vgprValuC+3]          // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+28], v[vgprValuC+4]          // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+29], v[vgprValuC+5]          // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+30], v[vgprValuC+6]          // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+31], v[vgprValuC+7]          // copy MI out reg to vreg[7]
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v26, v[vgprValuC+28], v[vgprValuC+29] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v27, v[vgprValuC+30], v[vgprValuC+31] // convert C to bf16 and Pack with neighbor
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[24:27], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_branch label_GW_End                              // jump to end
label_GW_End:
s_endpgm                                           // Kernel End
label_OptNLL_End:
label_GSU_3:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+3], v[vgprLocalReadAddrB+0] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+3], v[vgprLocalReadAddrB+0] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_alu depctr_vm_vsrc(2)
v_xor_b32 v[vgprLocalReadAddrA], 0x1000, v[vgprLocalReadAddrA] // swap Red Blk
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrB], 0x1000, v[vgprLocalReadAddrB] // swap Red Blk
s_wait_dscnt 4
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
s_wait_dscnt 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+7], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:

/* Tail: add ValuA/B vgpr buffer [12...46) to pool */

/* Tail: add address/G2L vgpr [46...46) to pool */

/******************************************/
/* Tail Loop                              */
/******************************************/

/* local write reset offsets a */

/* local write reset offsets b */

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 63, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 64
s_and_b32 s76, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s77, s[sgprSizesSum], 6                 // s77 = s[sgprSizesSum] / 64
s_and_b32 s78, s[sgprGSU], 0x3fff                  // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v12, s78                             // s76 = s77 / s78
v_rcp_iflag_f32 v12, v12                           // s76 = s77 / s78
v_cvt_f32_u32 v13, s77                             // s76 = s77 / s78
v_mul_f32 v12, v12, v13                            // s76 = s77 / s78
v_cvt_u32_f32 v12, v12                             // s76 = s77 / s78
v_mul_u32_u24 v13, v12, s78                        // s76 = s77 / s78
v_sub_nc_u32 v13, s77, v13                         // s76 = s77 / s78
v_cmp_eq_u32 vcc_lo, v13, s78                      // s76 = s77 / s78
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s76 = s77 / s78
v_add_nc_u32 v12, 1, v12                           // s76 = s77 / s78
v_mov_b32 v13, 0                                   // s[sgprGSUSumIdx+1] = s77 % s78
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v13, s78                      // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v12, v12, 1                           // quotient - 1
v_mul_u32_u24 v13, v12, s78                        // re-calculate remainder
v_sub_nc_u32 v13, s77, v13                         // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s76, v12                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v13        // remainder
s_sub_u32 s77, s78, 1                              // GSU-1
s_cmp_eq_u32 s76, 0                                // quotient == 0
s_cselect_b32 s76, s[sgprGSUSumIdx+1], s77         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s76                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0
// Skip barrier: NumThreads=32Barrier before tail TDM loads (WAR hazard with NLL LDS reads)
s_and_b32 s6, s[sgprSizeL], 63
// TDM reset tensor dim for tail
s_and_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], 0xffff
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff0000
s_lshl_b32 s7, s6, 0x10
s_or_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], s7
s_lshr_b32 s7, s6, 0x10
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s7
s_and_b32 s6, s[sgprSizeL], 63
// TDM reset tensor dim for tail
s_and_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], 0xffff
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff0000
s_lshl_b32 s7, s6, 0x10
s_or_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], s7
s_lshr_b32 s7, s6, 0x10
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s7

/* Update M0 for DTLDS */

/* Tail global read A */
s_barrier_wait -3                                  // cluster barrier wait
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS0

/* Update M0 for DTLDS */

/* Tail global read B */
tensor_load_to_lds s[sgprtdmBGroup0:sgprtdmBGroup0+3], s[sgprtdmBGroup1:sgprtdmBGroup1+7] // sync LDS0
s_wait_loadcnt 0
s_wait_tensorcnt 0                                 // 2wait for global read
// Skip barrier: NumThreads=32

/* Recalc local read offsets */
s_wait_dscnt 0                                     // 5wait for local write
// Skip barrier: NumThreads=32Tail loop LW->LR, sync LDS0
.set vgprValuA_X0_I0_BASE, 12
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+8
.set vgprValuB_X0_I0_BASE, 28
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+8

/* Tail: local read init pointers a */

/* localReadInitPointers */

/* Tail: local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
.align 16
label_TailLoopBeginL:

/* local read a */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
s_wait_alu depctr_va_vdst(1)
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read b */
s_wait_alu depctr_va_vdst(0)
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0

/* local read inc a */
s_mov_b32 s6, 64                                   // inc
s_wait_alu depctr_vm_vsrc(2)
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc_lo, s6, v[vgprLocalReadAddrA+0] // lrA += 64 (bpeDS)

/* local read inc b */
                                                   // inc (dup assign opt.)
s_wait_alu depctr_vm_vsrc(0)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, s6, v[vgprLocalReadAddrB+0] // lrB += 64 (bpeDS)
s_wait_dscnt 0                                     // 4wait for local read
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]

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
.set vgprValuB_X0_I0_BASE, UNDEF
.set vgprValuB_X0_I0, UNDEF
.set vgprValuB_X1_I0, UNDEF

/* Tail: add MISC Vgpr [8...12) to pool */
label_Summation_End_S4FDBQ587JJL6NOU:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprtdmAGroup0, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprtdmAGroup1, UNDEF
.set sgprtdmBGroup0, UNDEF
.set sgprtdmBGroup1, UNDEF
/* load store sgprs */

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_nop
v_nop
v_nop
v_nop
v_lshrrev_b32 v12, 5, v[vgprSerial]                // 12 = Serial / 32
v_lshrrev_b32 v13, 0, v12                          // 13 = 12 / 1
v_lshlrev_b32 v9, 4, v13                           // wave coordination offset 1
v_and_b32 v13, 15, v[vgprSerial]                   // v13 = v[vgprSerial] % 16
v_add_lshl_u32 v9, v13, v9, 0                      // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v10, v9, s[sgprStrideC1J]             //  offset 1
v_mul_lo_u32 v11, v9, s[sgprStrideD1J]             //  offset 1
v_and_b32 v13, 0, v12                              // v13 = v12 % 1
v_lshlrev_b32 v13, 4, v13                          // wave coordination offset 0
v_and_b32 v8, 31, v[vgprSerial]                    // v8 = v[vgprSerial] % 32
v_lshrrev_b32 v8, 4, v8                            // 8 = 8 / 16
v_lshlrev_b32 v8, 3, v8                            // thread0 * continuous_output
v_add_lshl_u32 v8, v13, v8, 0                      // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s6, 16, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_nc_u32 v8, s6, v8                            // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s6, 16, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v9, s6, v9                            // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_and_b32 s6, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s6, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:

/* Edge/NonEdge store path check (M): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 15, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 16
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_mov_b32 s6, 0
s_cmp_gt_u32 s28, s6                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW8_MB_Else         // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 15, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 16
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_mov_b32 s6, 0
s_cmp_gt_u32 s28, s6                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW8_MB_Then         // jump if edges required
label_GW_B0_FD0_VW8_MB_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=24 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw8)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_add_lshl_u32 v19, v11, v8, 2                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+24], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+30], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+7]          // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
s_wait_alu depctr_va_vdst(4)
buffer_store_b128 v[24:27], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[28:31], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW8_MB_NonEdgeEnd:
label_GW_B0_FD0_VW8_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=20 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw8)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s14, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v19, v11, v8, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v19, v14, v19, s16                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_mov_b32 v[vgprValuC+24], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+30], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+7]          // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
s_wait_alu depctr_va_vdst(4)
buffer_store_b128 v[24:27], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[28:31], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW8_MB_Else:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=100 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s14, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v27, v11, v8, 2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v27, v14, v27, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v12, vcc_lo, v8, 1                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v28, v11, v12, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v28, v14, v28, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v12, vcc_lo, v8, 2                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v29, v11, v12, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v29, v14, v29, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v12, vcc_lo, v8, 3                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v30, v11, v12, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v30, v14, v30, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
v_add_co_u32 v12, vcc_lo, v8, 4                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v31, v11, v12, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v31, v14, v31, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
v_add_co_u32 v12, vcc_lo, v8, 5                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v32, v11, v12, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v32, v14, v32, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
v_add_co_u32 v12, vcc_lo, v8, 6                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v33, v11, v12, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v33, v14, v33, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
v_add_co_u32 v12, vcc_lo, v8, 7                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v34, v11, v12, 2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v34, v14, v34, s16                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7)] */
v_mov_b32 v[vgprValuC+19], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+20], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+21], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+22], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+23], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+24], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+7]          // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
s_wait_alu depctr_va_vdst(7)
buffer_store_b32 v19, v27, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b32 v20, v28, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(5)
buffer_store_b32 v21, v29, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(4)
buffer_store_b32 v22, v30, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(3)
buffer_store_b32 v23, v31, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b32 v24, v32, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(1)
buffer_store_b32 v25, v33, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b32 v26, v34, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
s_getpc_b64 s[14:15]                               // addr of next instr
s_add_i32 s16, label_KernelEnd, 4                  // target branch offset
s_add_u32 s14, s14, s16                            // add target branch offset
s_addc_u32 s15, s15, 0                             // add high and carry
s_setpc_b64 s[14:15]                               // branch to label_KernelEnd
label_GSU_4:
s_mov_b32 s7, 0
s_cmp_eq_u32 s[sgprBeta], s7                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 15, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 16
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_mov_b32 s6, 0
s_cmp_gt_u32 s28, s6                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW8_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 15, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 16
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_mov_b32 s6, 0
s_cmp_gt_u32 s28, s6                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW8_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW8_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=24 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw8)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_add_lshl_u32 v19, v11, v8, 1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v26, v[vgprValuC+28], v[vgprValuC+29] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v27, v[vgprValuC+30], v[vgprValuC+31] // convert C to bf16 and Pack with neighbor
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[24:27], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW8_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW8_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=20 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw8)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s14, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v19, v11, v8, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v19, v14, v19, s16                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v26, v[vgprValuC+28], v[vgprValuC+29] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v27, v[vgprValuC+30], v[vgprValuC+31] // convert C to bf16 and Pack with neighbor
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[24:27], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW8_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=100 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s14, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v27, v11, v8, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v27, v14, v27, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v12, vcc_lo, v8, 1                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v28, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v28, v14, v28, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v12, vcc_lo, v8, 2                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v29, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v29, v14, v29, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v12, vcc_lo, v8, 3                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v30, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v30, v14, v30, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
v_add_co_u32 v12, vcc_lo, v8, 4                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v31, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v31, v14, v31, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
v_add_co_u32 v12, vcc_lo, v8, 5                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v32, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v32, v14, v32, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
v_add_co_u32 v12, vcc_lo, v8, 6                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v33, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v33, v14, v33, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
v_add_co_u32 v12, vcc_lo, v8, 7                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v34, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v34, v14, v34, s16                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7)] */
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
v_cvt_pk_bf16_f32 v19, v[vgprValuC+19], v[vgprValuC+19] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v19, v27, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_pk_bf16_f32 v20, v[vgprValuC+20], v[vgprValuC+20] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v20, v28, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v21, v29, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v22, v30, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v23, v31, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v24, v32, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v25, v33, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v26, v34, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 15, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 16
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rem
s_mov_b32 s6, 0
s_cmp_gt_u32 s28, s6                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW8_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 16 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s28, 15, s[sgprSizeJ]                    // s28 = s[sgprSizeJ] % 16
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rem
s_mov_b32 s6, 0
s_cmp_gt_u32 s28, s6                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW8_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW8_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=14 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw8)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_add_lshl_u32 v20, v10, v8, 1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8 (multiple bpe)
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[32:35], v20, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
v_add_lshl_u32 v19, v11, v8, 1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=8, coord0Vgpr=8 (multiple bpe)

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16

s_wait_loadcnt 0                                   // vlcnt(0) = 1 - 1 (beta) (interleaved)
v_cvt_f32_bf16 v12, v32.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v32.h op_sel:[1]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v33.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v33.h op_sel:[1]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v34.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v34.h op_sel:[1]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v35.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v35.h op_sel:[1]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v26, v[vgprValuC+28], v[vgprValuC+29] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v27, v[vgprValuC+30], v[vgprValuC+31] // convert C to bf16 and Pack with neighbor
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[24:27], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW8_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW8_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=14 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw8)                       */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s14, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v19, v10, v8, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v19, v14, v19, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[20:23], v19, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v19, v11, v8, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v19, v14, v19, s16                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0)] */
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
s_wait_loadcnt 0                                   // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16
v_cvt_f32_bf16 v12, v20.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v20.h op_sel:[1]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v21.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v21.h op_sel:[1]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+27], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v22.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+28], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v22.h op_sel:[1]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+29], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v23.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+30], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_f32_bf16 v12, v23.h op_sel:[1]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+31], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+25] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v25, v[vgprValuC+26], v[vgprValuC+27] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v26, v[vgprValuC+28], v[vgprValuC+29] // convert C to bf16 and Pack with neighbor
v_cvt_pk_bf16_f32 v27, v[vgprValuC+30], v[vgprValuC+31] // convert C to bf16 and Pack with neighbor
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[24:27], v19, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW8_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=68 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,0,1:vw1); (0,0,0,2:vw1); (0,0,0,3:vw1); (0,0,0,4:vw1); (0,0,0,5:vw1); (0,0,0,6:vw1); (0,0,0,7:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v14, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s14, v8, s[sgprSizeI]                 // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v28, v10, v8, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v28, v14, v28, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_d16_b16 v27, v28, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v28, v11, v8, 1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v28, v14, v28, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
v_add_co_u32 v12, vcc_lo, v8, 1                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v30, v10, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v30, v14, v30, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_d16_b16 v29, v30, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v30, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v30, v14, v30, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
v_add_co_u32 v12, vcc_lo, v8, 2                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v32, v10, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v32, v14, v32, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_d16_b16 v31, v32, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v32, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v32, v14, v32, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
v_add_co_u32 v12, vcc_lo, v8, 3                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v34, v10, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v34, v14, v34, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_d16_b16 v33, v34, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v34, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v34, v14, v34, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,4) */
v_add_co_u32 v12, vcc_lo, v8, 4                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v36, v10, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v36, v14, v36, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_d16_b16 v35, v36, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v36, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v36, v14, v36, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,5) */
v_add_co_u32 v12, vcc_lo, v8, 5                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v38, v10, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v38, v14, v38, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_d16_b16 v37, v38, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v38, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v38, v14, v38, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,6) */
v_add_co_u32 v12, vcc_lo, v8, 6                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v40, v10, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v40, v14, v40, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_d16_b16 v39, v40, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v40, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v40, v14, v40, s16                   // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,0,0,7) */
v_add_co_u32 v12, vcc_lo, v8, 7                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s14, v12, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s16, v9, s[sgprSizeJ]                 // coord1 < size1
s_and_b32 s16, s14, s16                            // in0 && in1
v_add_lshl_u32 v42, v10, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v42, v14, v42, s16                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_d16_b16 v41, v42, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v42, v11, v12, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v42, v14, v42, s16                   // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 0, 0, 4), (0, 0, 0, 5), (0, 0, 0, 6), (0, 0, 0, 7)] */
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
s_wait_loadcnt 0                                   // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v16, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v17, 0x7fff0000                          // fp32 Nan
v_mov_b32 v18, 0x7fff                              // rounding bias for bfloat16
v_cvt_f32_bf16 v12, v27.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+19], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v19, v[vgprValuC+19], v[vgprValuC+19] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v19, v28, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_f32_bf16 v12, v29.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+20], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v20, v[vgprValuC+20], v[vgprValuC+20] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v20, v30, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_f32_bf16 v12, v31.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+21], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v21, v[vgprValuC+21], v[vgprValuC+21] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v21, v32, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_f32_bf16 v12, v33.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+22], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v22, v[vgprValuC+22], v[vgprValuC+22] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v22, v34, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_f32_bf16 v12, v35.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+23], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v23, v[vgprValuC+23], v[vgprValuC+23] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v23, v36, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_f32_bf16 v12, v37.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+24], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v24, v[vgprValuC+24], v[vgprValuC+24] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v24, v38, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_f32_bf16 v12, v39.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+25], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v25, v[vgprValuC+25], v[vgprValuC+25] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v25, v40, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_cvt_f32_bf16 v12, v41.l op_sel:[0]               // cvt bf16 to f32
v_fmac_f32 v[vgprValuC+26], v12, s[sgprBeta]       // finalSum = sum*alpha + C*beta
v_cvt_pk_bf16_f32 v26, v[vgprValuC+26], v[vgprValuC+26] // convert C to bf16 in gwvw==1
s_wait_alu depctr_va_vdst(0)
buffer_store_b16 v26, v42, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_End_2:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
