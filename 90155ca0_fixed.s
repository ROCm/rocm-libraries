
/******************************************/
/* Begin Kernel                           */
/******************************************/
/* STINKY_TOTAL_INST_BYTES: 34948 */
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.protected Cijk_Alik_C_Bjlk_C_C_B_UserArgs_MT64x64x8_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW0_IU1_IA0_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV1_MIWT2_2_MXLITDM_MXSFIMS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA0_RAP0_SGROB0_SIA4_SS0_SPO0_SRVW0_SSO0_SVW2_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1_WGMXCC1
.globl Cijk_Alik_C_Bjlk_C_C_B_UserArgs_MT64x64x8_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW0_IU1_IA0_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV1_MIWT2_2_MXLITDM_MXSFIMS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA0_RAP0_SGROB0_SIA4_SS0_SPO0_SRVW0_SSO0_SVW2_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1_WGMXCC1
.p2align 8
.type Cijk_Alik_C_Bjlk_C_C_B_UserArgs_MT64x64x8_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW0_IU1_IA0_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV1_MIWT2_2_MXLITDM_MXSFIMS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA0_RAP0_SGROB0_SIA4_SS0_SPO0_SRVW0_SSO0_SVW2_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1_WGMXCC1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_C_Bjlk_C_C_B_UserArgs_MT64x64x8_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW0_IU1_IA0_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV1_MIWT2_2_MXLITDM_MXSFIMS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA0_RAP0_SGROB0_SIA4_SS0_SPO0_SRVW0_SSO0_SVW2_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1_WGMXCC1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 197 // vgprs
  .amdhsa_next_free_sgpr 74 // sgprs
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_inst_pref_size 255
.end_amdhsa_kernel
.text
/* Num VGPR   =197 */
/* Num AccVGPR=0 */
/* Num SGPR   =74 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 16 x 2 */
/* SubGroup= 4 x 32 */
/* VectorWidthA=2 */
/* VectorWidthB=2 */
/* GlobalReadVectorWidthA=1, GlobalReadVectorWidthB=1 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=True */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 3
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Cijk_Alik_C_Bjlk_C_C_B_UserArgs_MT64x64x8_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW0_IU1_IA0_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV1_MIWT2_2_MXLITDM_MXSFIMS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA0_RAP0_SGROB0_SIA4_SS0_SPO0_SRVW0_SSO0_SVW2_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1_WGMXCC1
    .symbol: 'Cijk_Alik_C_Bjlk_C_C_B_UserArgs_MT64x64x8_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW0_IU1_IA0_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV1_MIWT2_2_MXLITDM_MXSFIMS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA0_RAP0_SGROB0_SIA4_SS0_SPO0_SRVW0_SSO0_SVW2_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1_WGMXCC1.kd'
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
        .value_type:      f32c
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f32c
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
        .size:            8
        .offset:          64
        .value_kind:      by_value
        .value_type:      f32c
      - .name:            beta
        .size:            8
        .offset:          72
        .value_kind:      by_value
        .value_type:      f32c
      - .name:            D
        .size:            8
        .offset:          80
        .value_kind:      global_buffer
        .value_type:      f32c
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          88
        .value_kind:      global_buffer
        .value_type:      f32c
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      u32
      - .name:            batchOffsetD
        .size:            8
        .offset:          112
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          120
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          128
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          136
        .value_kind:      by_value
        .value_type:      u64
    .group_segment_fixed_size:   16384
    .kernarg_segment_align:      8
    .kernarg_segment_size:       144
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 74
    .sgpr_spill_count:           0
    .vgpr_count:                 197
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Alik_C_Bjlk_C_C_B_UserArgs_MT64x64x8_MI16x16x1_SN_LDSB0_AFC0_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW0_IU1_IA0_K1_LDSSI0_LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPMn1_LRVWn1_LWPMn1_MIAV1_MIWT2_2_MXLITDM_MXSFIMS_MO64_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTG0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVG0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA0_RAP0_SGROB0_SIA4_SS0_SPO0_SRVW0_SSO0_SVW2_SNLL0_SIP1_SGRO0_TDMI3_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THG0_THMXSA0_THMXSB0_THM0_THWS0_TPSN_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS32_WG32_4_1_WGMXCC1:
label_ASM_Start:  /// Main body of the asm kernel
s_mov_b64 s[64:65], 0
v_nop
global_prefetch_b8 v0, s[64:65] th:TH_LOAD_RT scope:SCOPE_SE
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
/* ValuC range: [0-64), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 68
.set vgprGlobalReadOffsetA, 64
.set vgprGlobalReadOffsetB, 65
.set vgprLocalReadAddrA, 66
.set vgprLocalReadAddrB, 67
.set vgprSerial, 104

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
.set sgprArgType, 6
.set sgprGSUSumIdx, 8
.set sgprGSULog2BpeC, 7
.set sgprGSULog2BpeD, 10
.set sgprStaggerU, 11
.set sgprWGM, 12
.set sgprLoopCounterL, 13
.set sgprOrigLoopCounter, 14
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprNumWorkGroups0, 15
.set sgprNumWorkGroups1, 24
.set sgprSizesFree, 28
.set sgprSizesSum, 31
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprStridesA, 36
.set sgprStridesB, 38
.set sgprAlpha, 40
.set sgprBeta, 42
.set sgprAddressD, 44
.set sgprAddressC, 46
.set sgprStridesD, 48
.set sgprStridesC, 50
.set sgprGSU, 52

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
.set constStrideB1J, 1
.set sgprStrideBL, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 64
.set MT1, 64
.set DepthU, 8
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

/* Load num of Gemms */
s_load_b32 s20, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s22, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s23, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_wait_kmcnt 0                                     // load args
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
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_b256 s[44:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_mov_b64 s[26:27], s[sgprKernArgAddress:sgprKernArgAddress+1] // XNACK protect for SMEM
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[26:27], 16
s_wait_kmcnt 0                                     // wait for args to load
label_LoadArgsEnd:
s_and_b32 s[sgprStaggerU], s22, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s22, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s21
s_mov_b32 m0, 0x4000                               // LDS clamp at 16384 bytes
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero
v_readfirstlane_b32 s22, v0                        // first tId
s_delay_alu instid0(VALU_DEP_1)
s_lshr_b32 s[sgprWaveIdx], s22, 5                  // wId=fTid // wavelen

/* Init workgroup id from ttmp */
s_mov_b32 s[sgprWorkGroup0], ttmp9
s_and_b32 s[sgprWorkGroup1], 0xffff, ttmp7
s_lshr_b32 s[sgprWorkGroup2], ttmp7, 0x10

/* remap workgroup to XCCs */
s_lshr_b32 s58, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s58, s58                             // Get log(WGMXCC)
s_lshr_b32 s59, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s58, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s55, s23, s58
s_lshl_b32 s55, s55, s58
s_cmp_ge_u32 s[sgprWorkGroup0], s55
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s59, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s55, s[sgprWorkGroup0], s58
s_bfm_b32 s56, s58, 0
s_and_b32 s56, s[sgprWorkGroup0], s56
s_lshr_b32 s57, s23, s58
s_mul_i32 s56, s56, s57
s_add_u32 s[sgprWorkGroup0], s55, s56
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[0:1], s59                          // s55 = s[sgprWorkGroup0] / s59
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_rcp_f64 v[0:1], v[0:1]                           // s55 = s[sgprWorkGroup0] / s59
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s55 = s[sgprWorkGroup0] / s59
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s55 = s[sgprWorkGroup0] / s59
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f64 v0, v[0:1]                           // s55 = s[sgprWorkGroup0] / s59
v_mul_lo_u32 v1, v0, s59                           // s55 = s[sgprWorkGroup0] / s59
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s55 = s[sgprWorkGroup0] / s59
v_cmp_ge_u32 vcc_lo, v2, s59                       // s55 = s[sgprWorkGroup0] / s59
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s55 = s[sgprWorkGroup0] / s59
v_add_nc_u32 v0, v0, 1                             // s55 = s[sgprWorkGroup0] / s59
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_mul_lo_u32 v1, v0, s59                           // s55 = s[sgprWorkGroup0] / s59
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s55 = s[sgprWorkGroup0] / s59
v_readfirstlane_b32 s55, v0                        // quotient
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_readfirstlane_b32 s56, v2                        // remainder
s_mul_i32 s55, s55, s59
/* temp1 = (wg%CU_Count)//WGMXCC */
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
s_lshr_b32 s56, s56, s58
/* temp0 = temp0 + temp1 */
s_add_u32 s55, s55, s56
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[0:1], s59                          // s56 = s23 / s59
v_rcp_f64 v[0:1], v[0:1]                           // s56 = s23 / s59
v_cvt_f64_u32 v[2:3], s23                          // s56 = s23 / s59
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s56 = s23 / s59
v_cvt_u32_f64 v0, v[0:1]                           // s56 = s23 / s59
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_mul_lo_u32 v1, v0, s59                           // s56 = s23 / s59
v_sub_nc_u32 v2, s23, v1                           // s56 = s23 / s59
s_delay_alu instid0(VALU_DEP_1)
v_cmp_ge_u32 vcc_lo, v2, s59                       // s56 = s23 / s59
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s56 = s23 / s59
v_add_nc_u32 v0, v0, 1                             // s56 = s23 / s59
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_readfirstlane_b32 s56, v0                        // quotient
s_mul_i32 s56, s56, s59
s_sub_u32 s57, s23, s56
s_cmp_gt_u32 s[sgprWorkGroup0], s56
s_cselect_b32 s56, s57, s59
s_lshr_b32 s56, s56, s58
s_bfm_b32 s57, s58, 0
s_and_b32 s57, s[sgprWorkGroup0], s57
s_mul_i32 s56, s56, s57
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s55, s56
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [68...170) to pool */
/* init: add vgpr [0...64) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 3, v0                            // 1. N offset: nOffset = nIdx * nStride(8)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshlrev_b32 v0, 1, v0                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v0, v1, 1, v0                       // 5. K offset: lrKOffset = kIdx * mStride(2); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 8, v0                       // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(256); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v1, 1, v1                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshl_add_u32 v1, v2, 7, v1                       // 5. K offset: lrKOffset = kIdx * mStride(128); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshl_add_u32 v1, v3, 5, v1                       // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(32); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 8                                   // LSU offset: stride = lsuStride(8) when umlds==True
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshlrev_b32 v[vgprLocalReadAddrA], 3, v[vgprLocalReadAddrA] //  (multiple bpe)

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 512                                 // LSU offset: stride = lsuStride(8)*(MT1(64) + PAD1(0))
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_lshlrev_b32 v[vgprLocalReadAddrB], 3, v[vgprLocalReadAddrB] //  (multiple bpe)

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x1000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

/* local write addresses: first offset b */
s_wait_kmcnt 0                                     // wait for 96/0 bytes of kern args over preload
v_mov_b32 v2, MT0                                  // set MT0 into sgpr
v_mov_b32 v1, s[sgprSizesFree+0]                   // set Free0 size
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
v_mov_b32 v2, MT1                                  // set MT1 into sgpr
v_mov_b32 v1, s[sgprSizesFree+1]                   // set Free1 size
s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
v_readfirstlane_b32 s[sgprNumWorkGroups0], v0      // set back to numWorkGroup0
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
s_nop 0                                            // 1 wait states
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_readfirstlane_b32 s[sgprNumWorkGroups1], v0      // set back to numWorkGroup1

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v0, s16                              // s16 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(VALU_DEP_1)
v_rcp_iflag_f32 v0, v0                             // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v0, v0, v1                               // s16 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v1, v0, s16                          // s16 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v1, s16                       // s16 = s[sgprWorkGroup0] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v0, 1, v0                             // s16 = s[sgprWorkGroup0] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s16                       // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_readfirstlane_b32 s16, v0                        // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v0, s[sgprNumWorkGroups0]            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_delay_alu instid0(VALU_DEP_1)
v_rcp_iflag_f32 v0, v0                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v0, v0, v1                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups0]        // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v0, 1, v0                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_readfirstlane_b32 s16, v0                        // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s16, s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s24, 96                                  // KernArgAddressOffset
s_mul_i32 s60, s20, 4
s_mov_b64 s[54:55], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s24, 228
s_mov_b32 s60, 0
s_mov_b64 s[54:55], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s15, 1
s_mov_b32 s61, 0
s_load_b128 s[28:31], s[54:55], s60
s_mov_b32 s16, 1
s_cmp_eq_u32 s20, s16                              // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_wait_kmcnt 0
s_lshr_b32 s58, s28, 6                             // s58 = s28 / 64
s_and_b32 s56, 63, s28                             // s56 = s28 % 64
s_addc_u32 s58, s58, 0
s_lshr_b32 s59, s29, 6                             // s59 = s29 / 64
s_and_b32 s56, 63, s29                             // s56 = s29 % 64
s_addc_u32 s59, s59, 0
s_mul_i32 s58, s58, s59
s_mul_i32 s58, s58, s30
s_and_b32 s59, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s58, s58, s59
s_add_u32 s61, s61, s58
s_cmp_lt_u32 s[sgprWorkGroup0], s61
s_cbranch_scc1 label_FOUND
s_add_u32 s60, s60, s24
s_load_b128 s[28:31], s[54:55], s60
s_add_u32 s15, s15, 1
s_cmp_lt_u32 s15, s20
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_wait_kmcnt 0
s_lshr_b32 s58, s28, 6                             // s58 = s28 / 64
s_and_b32 s56, 63, s28                             // s56 = s28 % 64
s_addc_u32 s58, s58, 0
s_lshr_b32 s59, s29, 6                             // s59 = s29 / 64
s_and_b32 s56, 63, s29                             // s56 = s29 % 64
s_addc_u32 s59, s59, 0
s_mul_i32 s58, s58, s59
s_mul_i32 s58, s58, s30
s_and_b32 s54, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s58, s58, s54
s_add_u32 s61, s61, s58

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s55, s15, 1
s_sub_u32 s54, s61, s58
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s54
/* Check if custom structure pointer is null */
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 2                                // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s55, s55, 96                             // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s55
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[32:47], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b128 s[48:51], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s55, s55, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s55
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b64 s[sgprAddressD:sgprAddressD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_b64 s[sgprAddressC:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18
s_load_b64 s[sgprAddressA:sgprAddressA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20
s_load_b64 s[sgprAddressB:sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28
s_load_b64 s[sgprStridesD:sgprStridesD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_b64 s[sgprStridesC:sgprStridesC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38
s_load_b64 s[sgprStridesA:sgprStridesA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_b64 s[sgprStridesB:sgprStridesB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48
s_load_b64 s[sgprAlpha:sgprAlpha+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
// Read Beta
s_load_b64 s[42:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
label_LoadExternalStructEnd:
/* init: add vgpr [68...170) to pool */
/* init: add vgpr [0...64) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v1, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 3, v0                            // 1. N offset: nOffset = nIdx * nStride(8)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshlrev_b32 v0, 1, v0                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v0, v1, 1, v0                       // 5. K offset: lrKOffset = kIdx * mStride(2); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v4, 5, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 8, v0                       // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(256); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 31, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(32)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v1, 1, v1                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshl_add_u32 v1, v2, 7, v1                       // 5. K offset: lrKOffset = kIdx * mStride(128); 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v3, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshl_add_u32 v1, v3, 5, v1                       // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(32); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // 2 = Serial / 32
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 8                                   // LSU offset: stride = lsuStride(8) when umlds==True
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0         // Final Offset: offset = (lro0+lsuoffset)*bpeDS
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshlrev_b32 v[vgprLocalReadAddrA], 3, v[vgprLocalReadAddrA] //  (multiple bpe)

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 512                                 // LSU offset: stride = lsuStride(8)*(MT1(64) + PAD1(0))
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1         // Final Offset: offset = (lro1+lsuoffset)*bpeDS
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_lshlrev_b32 v[vgprLocalReadAddrB], 3, v[vgprLocalReadAddrB] //  (multiple bpe)

/* local read addresses: declare addresses a */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x1000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

/* local write addresses: first offset b */
s_wait_kmcnt 0                                     // wait for 96/0 bytes of kern args over preload
v_mov_b32 v2, MT0                                  // set MT0 into sgpr
v_mov_b32 v1, s[sgprSizesFree+0]                   // set Free0 size
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
v_mov_b32 v2, MT1                                  // set MT1 into sgpr
v_mov_b32 v1, s[sgprSizesFree+1]                   // set Free1 size
s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
v_readfirstlane_b32 s[sgprNumWorkGroups0], v0      // set back to numWorkGroup0
v_cvt_f32_u32 v0, v2                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
v_rcp_iflag_f32 v0, v0                             // v0 = ceil(v1 / v2)
v_cvt_f32_u32 v3, v1                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v0, v0, v3                               // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // v0 = ceil(v1 / v2)
v_mul_u32_u24 v3, v0, v2                           // v0 = ceil(v1 / v2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v3, v1, v3                            // v0 = ceil(v1 / v2)
v_cmp_ne_u32 vcc_lo, v3, 0                         // v0 = ceil(v1 / v2)
v_add_co_ci_u32 v0, vcc_lo, v0, 0, vcc_lo          // ceil
s_nop 0                                            // 1 wait states
s_delay_alu instid0(VALU_DEP_1)
v_readfirstlane_b32 s[sgprNumWorkGroups1], v0      // set back to numWorkGroup1

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_i32 s16, s16, s17
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s16                              // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v0, v0                             // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v0, v0, v1                               // s16 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v1, v0, s16                          // s16 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v1, s16                       // s16 = s[sgprWorkGroup0] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v0, 1, v0                             // s16 = s[sgprWorkGroup0] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s16                       // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_readfirstlane_b32 s16, v0                        // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v0, s[sgprNumWorkGroups0]            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_delay_alu instid0(VALU_DEP_1)
v_rcp_iflag_f32 v0, v0                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v0, v0, v1                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups0]        // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v0, 1, v0                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_readfirstlane_b32 s16, v0                        // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s16, s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprtdmAGroup0, 56
.set sgprtdmAGroup1, 60
.set sgprtdmBGroup0, sgprtdmAGroup0+0
.set sgprtdmBGroup1, sgprtdmAGroup1+0
.set sgprtdmABIncs, 25
.set sgprStaggerUIter, 26
.set sgprWrapUA, 54
.set sgprWrapUB, 68
.set sgprGlobalReadIncsA, 27
.set sgprGlobalReadIncsB, 53
s_and_b32 s16, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s16, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
label_Skip_Address_Prepad_For_Pointer_Array:  /// Skip pre-padding of address for pointer array case

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha].real == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha].real != 0
v_cmp_eq_f32 vcc_lo, s[sgprAlpha+1], 0.0           // s[Alpha].imag == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha].imag != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:
s_setreg_IMM32_b32 hwreg(26,2,1), 1                // Disable WMMA arb stall
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
s_and_b32 s16, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v1, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_f32_u32 v0, s16                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
v_rcp_iflag_f32 v0, v0                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_f32 v0, v0, v1                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_u32_u24 v1, v0, s16                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cmp_eq_u32 vcc_lo, v1, s16                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_add_nc_u32 v0, 1, v0                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
v_cmp_gt_u32 vcc_lo, v1, s16                       // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s16                          // re-calculate remainder
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
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
v_cvt_f32_u32 v1, s[sgprWorkGroup1]                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
v_rcp_iflag_f32 v0, v0                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v0, v0, v1                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups1]        // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprNumWorkGroups1]     // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_nc_u32 v0, 1, v0                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v1, 0                                    // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups1]     // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups1]        // re-calculate remainder
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprGSUSumIdx], v0           // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v1          // remainder
label_GSUWGMRR_End:
s_mov_b32 s[sgprGSULog2BpeC], 3
s_mov_b32 s[sgprGSULog2BpeD], 3
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 3
s_mov_b32 s[sgprGSULog2BpeD], 3
label_GSU_End:
s_sext_i32_i16 s16, s[sgprWGM]                     // Restore WGM
s_cmp_gt_i32 s16, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s16, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s16, s16                                 // abs(WGM)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[0:1], s16                          // s17 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_rcp_f64 v[0:1], v[0:1]                           // s17 = s[sgprWorkGroup0] / s16
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s17 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f64 v0, v[0:1]                           // s17 = s[sgprWorkGroup0] / s16
v_mul_lo_u32 v1, v0, s16                           // s17 = s[sgprWorkGroup0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s17 = s[sgprWorkGroup0] / s16
v_cmp_ge_u32 vcc_lo, v2, s16                       // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups0]        // s18 = s[sgprNumWorkGroups0] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s17 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v0, v0, 1                             // s17 = s[sgprWorkGroup0] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
v_readfirstlane_b32 s17, v0                        // quotient
v_cvt_f64_u32 v[0:1], s16                          // s18 = s[sgprNumWorkGroups0] / s16
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
v_rcp_f64 v[0:1], v[0:1]                           // s18 = s[sgprNumWorkGroups0] / s16
s_sub_u32 s20, s[sgprWorkGroup0], s20              // WorkGroup0=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s20, s20, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s18 = s[sgprNumWorkGroups0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f64 v0, v[0:1]                           // s18 = s[sgprNumWorkGroups0] / s16
v_mul_lo_u32 v1, v0, s16                           // s18 = s[sgprNumWorkGroups0] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v2, s[sgprNumWorkGroups0], v1         // s18 = s[sgprNumWorkGroups0] / s16
v_cmp_ge_u32 vcc_lo, v2, s16                       // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_f64_u32 v[2:3], s20                          // s[sgprWorkGroup1] = s20 / s18
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s18 = s[sgprNumWorkGroups0] / s16
v_add_nc_u32 v0, v0, 1                             // s18 = s[sgprNumWorkGroups0] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_readfirstlane_b32 s18, v0                        // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups0], s19          // NumWorkGroups0=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[0:1], s18                          // s[sgprWorkGroup1] = s20 / s18
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup1] = s20 / s18
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup1] = s20 / s18
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup1] = s20 / s18
v_mul_lo_u32 v1, v0, s18                           // s[sgprWorkGroup1] = s20 / s18
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v2, s20, v1                           // s[sgprWorkGroup1] = s20 / s18
v_cmp_ge_u32 vcc_lo, v2, s18                       // s[sgprWorkGroup1] = s20 / s18
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s20 / s18
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup1] = s20 / s18
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
v_mul_lo_u32 v1, v0, s18                           // s[sgprWorkGroup1] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_sub_nc_u32 v2, s20, v1                           // s[sgprWorkGroup1] = s20 / s18
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
v_readfirstlane_b32 s[sgprWorkGroup0], v2          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0] // WorkGroup0=remainder
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup1]            // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[0:1], s16                          // s17 = s[sgprWorkGroup1] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_rcp_f64 v[0:1], v[0:1]                           // s17 = s[sgprWorkGroup1] / s16
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s17 = s[sgprWorkGroup1] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f64 v0, v[0:1]                           // s17 = s[sgprWorkGroup1] / s16
v_mul_lo_u32 v1, v0, s16                           // s17 = s[sgprWorkGroup1] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v2, s[sgprWorkGroup1], v1             // s17 = s[sgprWorkGroup1] / s16
v_cmp_ge_u32 vcc_lo, v2, s16                       // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups1]        // s18 = s[sgprNumWorkGroups1] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s17 = s[sgprWorkGroup1] / s16
v_add_nc_u32 v0, v0, 1                             // s17 = s[sgprWorkGroup1] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
v_readfirstlane_b32 s17, v0                        // quotient
v_cvt_f64_u32 v[0:1], s16                          // s18 = s[sgprNumWorkGroups1] / s16
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
v_rcp_f64 v[0:1], v[0:1]                           // s18 = s[sgprNumWorkGroups1] / s16
s_sub_u32 s20, s[sgprWorkGroup1], s20              // WorkGroup1=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s20, s20, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s18 = s[sgprNumWorkGroups1] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f64 v0, v[0:1]                           // s18 = s[sgprNumWorkGroups1] / s16
v_mul_lo_u32 v1, v0, s16                           // s18 = s[sgprNumWorkGroups1] / s16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v2, s[sgprNumWorkGroups1], v1         // s18 = s[sgprNumWorkGroups1] / s16
v_cmp_ge_u32 vcc_lo, v2, s16                       // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_f64_u32 v[2:3], s20                          // s[sgprWorkGroup0] = s20 / s18
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s18 = s[sgprNumWorkGroups1] / s16
v_add_nc_u32 v0, v0, 1                             // s18 = s[sgprNumWorkGroups1] / s16
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_readfirstlane_b32 s18, v0                        // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups1], s19          // NumWorkGroups1=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[0:1], s18                          // s[sgprWorkGroup0] = s20 / s18
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup0] = s20 / s18
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup0] = s20 / s18
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup0] = s20 / s18
v_mul_lo_u32 v1, v0, s18                           // s[sgprWorkGroup0] = s20 / s18
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v2, s20, v1                           // s[sgprWorkGroup0] = s20 / s18
v_cmp_ge_u32 vcc_lo, v2, s18                       // s[sgprWorkGroup0] = s20 / s18
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s20 / s18
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup0] = s20 / s18
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
v_mul_lo_u32 v1, v0, s18                           // s[sgprWorkGroup0] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup0], v0          // quotient
v_sub_nc_u32 v2, s20, v1                           // s[sgprWorkGroup0] = s20 / s18
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s20, s[sgprWorkGroup1] // WorkGroup1=remainder
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s17 // wg1 += blockId * WGM
label_WGM:
label_TDMInitA:
s_bitcmp1_b32 s[sgprWaveIdx], 0                    // Check parity of wId
s_cbranch_scc1 label_TDMInitB                      // Jump to B if wId is odd
s_mov_b32 s[sgprtdmAGroup0+0], 1
s_mov_b32 s[sgprtdmAGroup0+1], 0
s_mov_b32 s[sgprtdmAGroup0+2], 0
s_mov_b32 s[sgprtdmAGroup0+3], 0
s_mov_b32 s[sgprtdmAGroup1+0], 0
s_mov_b32 s[sgprtdmAGroup1+1], 0
s_mov_b32 s[sgprtdmAGroup1+2], 0
s_mov_b32 s[sgprtdmAGroup1+3], 0
s_mov_b32 s[sgprtdmAGroup1+4], 0
s_mov_b32 s[sgprtdmAGroup1+5], 0
s_mov_b32 s[sgprtdmAGroup1+6], 0
s_mov_b32 s[sgprtdmAGroup1+7], 0
s_or_b32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0x80000000 // set type field to 2(image)
s_and_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0xfffcffff // Reset data_size
s_mov_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprAddressA:sgprAddressA+1]
s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0x30000 // Set data_size to 3
s_or_b32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], 0x80000000 // set type field to 2(image)
s_and_b32 s18, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s18, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_TDMGeneralBatchedAddressAReady // Keep the direct matrix address
s_mov_b32 s16, 1                                   // check summation size
s_mul_i32 s16, s[sgprSizesSum+0], s16              // check summation size
s_cmp_eq_u32 s16, 0                                // skip pointer dereference when summation size is zero
s_cbranch_scc1 label_TDMGeneralBatchedAddressAReady
v_cmp_eq_f32 s16, s[sgprAlpha], 0.0                // s[Alpha].real == 0.0f ?
v_cmp_eq_f32 vcc_lo, s[sgprAlpha+1], 0.0           // s[Alpha].imag == 0.0f ?
s_delay_alu instid0(VALU_DEP_2)
s_and_b32 s16, vcc_lo, s16                         // s[Alpha] == 0 ?
s_cmp_eq_u32 s16, 0                                // branch if s[Alpha] == 0
s_cbranch_scc0 label_TDMGeneralBatchedAddressAReady // branch if s[Alpha] == 0
s_mul_i32 s16, s[sgprWorkGroup2], 8                // offset of A pointer-array entry
s_add_u32 s16, s16, s[sgprAddressA+0]              // address of A[batch] (low)
s_addc_u32 s17, s[sgprAddressA+1], 0               // address of A[batch] (high)
s_load_b64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[16:17], 0 // load A matrix address from pointer array
s_wait_kmcnt 0                                     // wait for A matrix address
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70 // load batchOffsetA from kernel args
s_wait_kmcnt 0                                     // wait for batchOffsetA
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s16 // apply batchOffsetA (low)
s_addc_u32 s[sgprtdmAGroup0+2+1], s[sgprtdmAGroup0+2+1], s17 // apply batchOffsetA (high)
s_or_b32 s[sgprtdmAGroup0+2+1], s[sgprtdmAGroup0+2+1], 0x80000000 // restore type field to 2(image)
label_TDMGeneralBatchedAddressAReady:  /// TDM A matrix address is ready
s_lshr_b32 s16, s[sgprWaveIdx], 1                  // componentId=WaveIdx >> 1
s_mov_b32 s[sgprtdmAGroup1+5], s[sgprStrideA0I]
s_mov_b32 s[sgprtdmAGroup1+6], 0
s_mul_i32 s16, s16, 2048                           // woffset = wId * (mt // numComp * du * bpe // dim1Divisor)
s_add_u32 s16, s16, 0                              // ldsOffset = woffset + ldsConstOffset
s_mov_b32 s[sgprtdmAGroup0+1], s16
s_mul_i32 s16, 64, s[sgprWorkGroup0]
s_sub_i32 s16, s[sgprSizeI], s16
s_and_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0xfff7ffff // clear iterate_enable (D# Group 1 bit 19)
s_and_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], 0xffff
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff0000
s_lshl_b32 s17, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], s17
s_lshr_b32 s17, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s17
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s17, v[vgprSerial]             // first tId
s_delay_alu instid0(VALU_DEP_1)
s_lshr_b32 s17, s17, 6                             // wId=fTid // wavelen // 2
s_mul_i32 s17, s17, 32                             // woffset = wId * (mt // numComp // dim1Divisor)
s_sub_u32 s16, s16, s17                            // consider multiple waves
s_cmov_b32 s16, 0                                  // set to 0 for waves that no enough data to load
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff
s_and_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0xffff0000
s_lshl_b32 s18, s16, 0x10
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s18
s_lshr_b32 s18, s16, 0x10
s_or_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], s18
s_and_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0xffff
s_or_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0x80000 // set tile0 to 8
s_and_b32 s[sgprtdmAGroup1+4], s[sgprtdmAGroup1+4], 0xffff0000
s_or_b32 s[sgprtdmAGroup1+4], s[sgprtdmAGroup1+4], 0x20 // set tile1 to 32
s_branch label_TDMInitABEnd
label_TDMInitB:
s_mov_b32 s[sgprtdmBGroup0+0], 1
s_mov_b32 s[sgprtdmBGroup0+1], 0
s_mov_b32 s[sgprtdmBGroup0+2], 0
s_mov_b32 s[sgprtdmBGroup0+3], 0
s_mov_b32 s[sgprtdmBGroup1+0], 0
s_mov_b32 s[sgprtdmBGroup1+1], 0
s_mov_b32 s[sgprtdmBGroup1+2], 0
s_mov_b32 s[sgprtdmBGroup1+3], 0
s_mov_b32 s[sgprtdmBGroup1+4], 0
s_mov_b32 s[sgprtdmBGroup1+5], 0
s_mov_b32 s[sgprtdmBGroup1+6], 0
s_mov_b32 s[sgprtdmBGroup1+7], 0
s_or_b32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0x80000000 // set type field to 2(image)
s_and_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0xfffcffff // Reset data_size
s_mov_b64 s[sgprtdmBGroup0+2:sgprtdmBGroup0+2+1], s[sgprAddressB:sgprAddressB+1]
s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0x30000 // Set data_size to 3
s_or_b32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0x80000000 // set type field to 2(image)
s_and_b32 s18, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s18, 3                                // ArgType == 3 for General Batched GEMM
s_cbranch_scc0 label_TDMGeneralBatchedAddressBReady // Keep the direct matrix address
s_mov_b32 s16, 1                                   // check summation size
s_mul_i32 s16, s[sgprSizesSum+0], s16              // check summation size
s_cmp_eq_u32 s16, 0                                // skip pointer dereference when summation size is zero
s_cbranch_scc1 label_TDMGeneralBatchedAddressBReady
v_cmp_eq_f32 s16, s[sgprAlpha], 0.0                // s[Alpha].real == 0.0f ?
v_cmp_eq_f32 vcc_lo, s[sgprAlpha+1], 0.0           // s[Alpha].imag == 0.0f ?
s_delay_alu instid0(VALU_DEP_2)
s_and_b32 s16, vcc_lo, s16                         // s[Alpha] == 0 ?
s_cmp_eq_u32 s16, 0                                // branch if s[Alpha] == 0
s_cbranch_scc0 label_TDMGeneralBatchedAddressBReady // branch if s[Alpha] == 0
s_mul_i32 s16, s[sgprWorkGroup2], 8                // offset of B pointer-array entry
s_add_u32 s16, s16, s[sgprAddressB+0]              // address of B[batch] (low)
s_addc_u32 s17, s[sgprAddressB+1], 0               // address of B[batch] (high)
s_load_b64 s[sgprtdmBGroup0+2:sgprtdmBGroup0+2+1], s[16:17], 0 // load B matrix address from pointer array
s_wait_kmcnt 0                                     // wait for B matrix address
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // load batchOffsetB from kernel args
s_wait_kmcnt 0                                     // wait for batchOffsetB
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s16 // apply batchOffsetB (low)
s_addc_u32 s[sgprtdmBGroup0+2+1], s[sgprtdmBGroup0+2+1], s17 // apply batchOffsetB (high)
s_or_b32 s[sgprtdmBGroup0+2+1], s[sgprtdmBGroup0+2+1], 0x80000000 // restore type field to 2(image)
label_TDMGeneralBatchedAddressBReady:  /// TDM B matrix address is ready
s_lshr_b32 s16, s[sgprWaveIdx], 1                  // componentId=WaveIdx >> 1
s_mov_b32 s[sgprtdmBGroup1+5], s[sgprStrideBL]
s_mov_b32 s[sgprtdmBGroup1+6], 0
s_mul_i32 s16, s16, 2048                           // woffset = wId * (mt // numComp * du * bpe // dim1Divisor)
s_add_u32 s16, s16, 4096                           // ldsOffset = woffset + ldsConstOffset
s_mov_b32 s[sgprtdmBGroup0+1], s16
s_mul_i32 s16, 64, s[sgprWorkGroup1]
s_sub_i32 s16, s[sgprSizeJ], s16
s_and_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0xfff7ffff // clear iterate_enable (D# Group 1 bit 19)
s_and_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], 0xffff
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff0000
s_lshl_b32 s17, s16, 0x10
s_or_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], s17
s_lshr_b32 s17, s16, 0x10
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s17
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff0000
s_lshl_b32 s18, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s18
s_lshr_b32 s18, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], s18
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0x400000 // set tile0 to 64
s_and_b32 s[sgprtdmBGroup1+4], s[sgprtdmBGroup1+4], 0xffff0000
s_or_b32 s[sgprtdmBGroup1+4], s[sgprtdmBGroup1+4], 0x4 // set tile1 to 4
label_TDMInitABEnd:
label_TDMGlobalOffsetA:
s_bitcmp1_b32 s[sgprWaveIdx], 0                    // Check parity of wId
s_cbranch_scc1 label_TDMGlobalOffsetB              // Jump to B if wId is odd
s_mov_b64 s[16:17], 0
s_lshr_b32 s18, s[sgprWaveIdx], 1                  // wCompId = WaveIdx >> 1
s_mul_i32 s16, s[sgprStrideA0I], 512               // tileStride * MT(64) * bpe(8)
s_mul_i32 s18, s18, 256                            // woffset = wCompId * mt // numComp(2) * bpe(8) // tdmSplit(1)
s_mul_hi_u32 s17, s16, s[sgprWorkGroup0]           // *= wgId
s_mul_i32 s18, s18, s[sgprStrideA0I]               // woffset *= tdmSeparateStride
s_mul_i32 s16, s16, s[sgprWorkGroup0]              // *= wgId
s_add_u32 s16, s16, s18                            // += woffset
s_addc_u32 s17, s17, 0                             // += woffset carry
s_and_b32 s18, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_TDM_GSUC_A                    // branch if GSUC == 1
s_mov_b32 s18, s[sgprGSUSumIdx]                    // gsuIterOffset = GSUSumIdx
s_branch label_TDM_GSUC_A_End
label_TDM_GSUC_A:
s_lshr_b32 s22, s[sgprSizesSum], 3                 // numIterL = SizesSum / DepthU(8)
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v1, s22                              // s22 = s22 / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s22 = s22 / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
v_rcp_iflag_f32 v0, v0                             // s22 = s22 / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s22 = s22 / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s22 = s22 / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s22 = s22 / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s22, v1                           // s22 = s22 / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // s22 = s22 / s[sgprGSUSumIdx+1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s22 = s22 / s[sgprGSUSumIdx+1]
v_add_nc_u32 v0, 1, v0                             // s22 = s22 / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s22 % s[sgprGSUSumIdx+1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
v_cmp_gt_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // re-calculate remainder
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
v_sub_nc_u32 v1, s22, v1                           // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s22, v0                        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_mul_i32 s21, s22, s[sgprGSUSumIdx]               // quotient*GSUSumIdx
s_add_u32 s20, 1, s22                              // quotient+1
s_add_u32 s21, s21, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s20, s20, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s20, s20, s21                        // (quotient+1)*GSUSumIdx if needed
s_mov_b32 s18, s20                                 // gsuIterOffset = accumulatedNumOfLoopCounterL
label_TDM_GSUC_A_End:
s_mul_i32 s18, s18, 64                             // gsuOffset = gsuIterOffset * DepthU(8) * bpe(8)
s_add_u32 s16, s16, s18                            // += gsuOffset
s_addc_u32 s17, s17, 0                             // += gsuOffset carry
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s16 // += tileOffset(lo)
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], s17 // += tileOffset(hi)
s_and_b32 s19, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s19, 3                                // ArgType == 3 for General Batched GEMM
s_cselect_b32 s18, 0, s[sgprWorkGroup2]            // general batch uses an already-dereferenced matrix base
s_mul_hi_u32 s17, s[sgprStrideAK], s18             // Batch: Stride*WG
s_mul_i32 s16, s[sgprStrideAK], s18                // Batch: Stride*WG
s_lshl_b64 s[16:17], s[16:17], 3                   // scale by bpe (multiple bpe)
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s16 // += batchOffset(lo)
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], s17 // += batchOffset(hi)
s_branch label_TDMGlobalOffsetABEnd
label_TDMGlobalOffsetB:
s_mov_b64 s[16:17], 0
s_lshr_b32 s18, s[sgprWaveIdx], 1                  // wCompId = WaveIdx >> 1
s_mul_i32 s16, constStrideB1J, 512                 // tileStride * MT(64) * bpe(8)
s_mul_i32 s18, s18, 32                             // woffset = wCompId * mt // numComp(2) * bpe(8) // tdmSplit(1)
s_mul_hi_u32 s17, s16, s[sgprWorkGroup1]           // *= wgId
s_mul_i32 s18, s18, s[sgprStrideBL]                // woffset *= tdmSeparateStride
s_mul_i32 s16, s16, s[sgprWorkGroup1]              // *= wgId
s_add_u32 s16, s16, s18                            // += woffset
s_addc_u32 s17, s17, 0                             // += woffset carry
s_and_b32 s18, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_TDM_GSUC_B                    // branch if GSUC == 1
s_mov_b32 s18, s[sgprGSUSumIdx]                    // gsuIterOffset = GSUSumIdx
s_branch label_TDM_GSUC_B_End
label_TDM_GSUC_B:
s_lshr_b32 s22, s[sgprSizesSum], 3                 // numIterL = SizesSum / DepthU(8)
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v1, s22                              // s22 = s22 / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s22 = s22 / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
v_rcp_iflag_f32 v0, v0                             // s22 = s22 / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s22 = s22 / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s22 = s22 / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s22 = s22 / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s22, v1                           // s22 = s22 / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // s22 = s22 / s[sgprGSUSumIdx+1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s22 = s22 / s[sgprGSUSumIdx+1]
v_add_nc_u32 v0, 1, v0                             // s22 = s22 / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s22 % s[sgprGSUSumIdx+1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
v_cmp_gt_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // re-calculate remainder
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
v_sub_nc_u32 v1, s22, v1                           // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s22, v0                        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_mul_i32 s21, s22, s[sgprGSUSumIdx]               // quotient*GSUSumIdx
s_add_u32 s20, 1, s22                              // quotient+1
s_add_u32 s21, s21, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s20, s20, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s20, s20, s21                        // (quotient+1)*GSUSumIdx if needed
s_mov_b32 s18, s20                                 // gsuIterOffset = accumulatedNumOfLoopCounterL
label_TDM_GSUC_B_End:
s_mul_i32 s18, s18, 64                             // gsuOffset = gsuIterOffset * DepthU(8) * bpe(8)
s_mul_i32 s18, s18, s[sgprStrideBL]                // tlu=1, scale GSU offset by unroll stride
s_add_u32 s16, s16, s18                            // += gsuOffset
s_addc_u32 s17, s17, 0                             // += gsuOffset carry
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s16 // += tileOffset(lo)
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], s17 // += tileOffset(hi)
s_and_b32 s19, s[sgprArgType], 0xff                // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s19, 3                                // ArgType == 3 for General Batched GEMM
s_cselect_b32 s18, 0, s[sgprWorkGroup2]            // general batch uses an already-dereferenced matrix base
s_mul_hi_u32 s17, s[sgprStrideBK], s18             // Batch: Stride*WG
s_mul_i32 s16, s[sgprStrideBK], s18                // Batch: Stride*WG
s_lshl_b64 s[16:17], s[16:17], 3                   // scale by bpe (multiple bpe)
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s16 // += batchOffset(lo)
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], s17 // += batchOffset(hi)
label_TDMGlobalOffsetABEnd:
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_mov_b32 s[sgprGlobalReadIncsA+0], 64             // GSU*DepthU*Bpe*MI_dim(1)
s_mov_b32 s[sgprGlobalReadIncsB+0], 64             // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_mul_i32 s17, s17, s[sgprGlobalReadIncsA+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_cselect_b32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], s17 // incrA (unrollIdx)
s_and_b32 s17, s[sgprGSU], 0xfff                   // Restore GSU
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_mul_i32 s17, s17, s[sgprGlobalReadIncsB+0]       // GSU*DepthU*Bpe*MI_dim(1)
s_cmov_b32 s17, 64                                 // DepthU*Bpe if GSUC = 1
s_bitcmp1_b32 s[sgprWaveIdx], 0                    // Check parity of wId
s_mul_i32 s[sgprGlobalReadIncsB+0], s17, s[sgprStrideBL] // incrB unrollIdx)
s_cselect_b32 s[sgprtdmABIncs], s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsA+0]
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 3 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 8
s_and_b32 s16, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff    // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v0, 1, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
v_cmp_gt_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // re-calculate remainder
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
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
s_bitcmp1_b32 s[sgprWaveIdx], 0                    // check wave parity
s_mul_hi_u32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_cbranch_scc1 label_SkipStaggerA                  // skip: odd waves do not carry A
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s16 // TDM addr += stagger offset (lo)
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], s17 // TDM addr += stagger offset (hi)
label_SkipStaggerA:
s_mul_hi_u32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mov_b32 s21, 0
s_mul_hi_u32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_bitcmp1_b32 s[sgprWaveIdx], 0                    // check wave parity
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mov_b32 s20, s[sgprGlobalReadIncsA+0]
s_sub_u64 s[sgprWrapUA:sgprWrapUA+1], s[20:21], s[sgprWrapUA:sgprWrapUA+1] // increment-WrapU
s_cbranch_scc0 label_SkipStaggerB                  // skip: even waves do not carry B
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s16 // TDM addr += stagger offset (lo)
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], s17 // TDM addr += stagger offset (hi)
label_SkipStaggerB:
s_mul_hi_u32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mov_b32 s21, 0
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_bitcmp1_b32 s[sgprWaveIdx], 0                    // check wave parity
s_mov_b32 s20, s[sgprGlobalReadIncsB+0]
s_sub_u64 s[sgprWrapUB:sgprWrapUB+1], s[20:21], s[sgprWrapUB:sgprWrapUB+1] // increment-WrapU
s_cselect_b32 s[sgprWrapUA+0], s[sgprWrapUB+0], s[sgprWrapUA+0] // hoist: WrapUA = parity ? WrapUB : WrapUA (lo)
s_cselect_b32 s[sgprWrapUA+1], s[sgprWrapUB+1], s[sgprWrapUA+1] // hoist: WrapUA = parity ? WrapUB : WrapUA (hi)
s_and_b32 s[sgprArgType], s[sgprArgType], 0xfffffeff // clear ArgType bit 8 (TDM wave-parity)
s_bitcmp1_b32 s[sgprWaveIdx], 0                    // TDM wave-parity -> ArgType bit 8
s_cbranch_scc0 label_SkipPackTdmParityArgType      // even wave: leave bit 8 clear
s_or_b32 s[sgprArgType], s[sgprArgType], 0x100     // odd wave: ArgType bit 8 = 1
label_SkipPackTdmParityArgType:  /// even wave: ArgType bit 8 stays 0
.set sgprWaveIdx, UNDEF
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS0
s_add_u32 s16, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s16              // Is this wrapIter? (pf)
s_cselect_b32 s16, s[sgprWrapUA+0], s[sgprtdmABIncs] // select WrapU or normal inc (lo)
s_cselect_b32 s17, s[sgprWrapUA+1], 0              // select WrapU or normal inc (hi)
s_add_u64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[16:17] // TDM addr += inc (with wrap, 64-bit)
label_ShadowInitStart:
s_and_b32 s5, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s5, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckD                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationD_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckD:  /// Check if ArgType is for General Batched GEMM for D
s_and_b32 s5, s[sgprArgType], 0xff                 // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s5, 3                                 // ArgType == 3 for General Batched GEMM
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
s_and_b32 s5, s[sgprSrdD+2], 127
s_lshl_b32 s5, s5, 25
s_and_b32 s[sgprSrdD+1], s[sgprSrdD+1], 33554431
s_or_b32 s[sgprSrdD+1], s[sgprSrdD+1], s5
s_lshr_b32 s[sgprSrdD+2], s[sgprSrdD+2], 7
s_and_b32 s5, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s5, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_ArgTypeCheckC                 // Handling General Batched GEMM SRD initialization
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_branch label_GeneralBatchedGemmSrdInitiationC_End // End of handling General Batched GEMM SRD initialization
label_ArgTypeCheckC:  /// Check if ArgType is for General Batched GEMM for C
s_and_b32 s5, s[sgprArgType], 0xff                 // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s5, 3                                 // ArgType == 3 for General Batched GEMM
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
s_mul_i32 s72, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_and_b32 s5, s[sgprSrdC+2], 127
s_mul_i32 s70, s72, s[sgprStrideC1J]               // ScaleC s72 by Stride
s_lshl_b32 s5, s5, 25
s_and_b32 s[sgprSrdC+1], s[sgprSrdC+1], 33554431
s_or_b32 s[sgprSrdC+1], s[sgprSrdC+1], s5
s_lshr_b32 s[sgprSrdC+2], s[sgprSrdC+2], 7
s_and_b32 s71, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s71, s72, s[sgprStrideC1J]            // ScaleC s72 by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s70        // add lo to SRD
s_mul_i32 s70, s72, s[sgprStrideD1J]               // ScaleD s72 by Stride
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s71       // add hi to SRD
s_and_b32 s71, s[sgprGSU], 0xfff                   // Restore GSU
s_mul_hi_u32 s71, s72, s[sgprStrideD1J]            // ScaleD s72 by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi to SRD
s_and_b32 s71, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s71, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_and_b32 s5, s[sgprArgType], 0xff                 // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s5, 3                                 // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s71, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s70, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s71       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s70, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s70, s70, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s71, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[72:73], s[70:71], 0                   // Load the Matrix Address in the Pointer Array
s_wait_kmcnt 0                                     // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s72        // Offsetting within the Batch Matrix [Lower half of address]
s_load_b64 s[70:71], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68 // Load batchOffsetC from kernel args
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s73       // Offsetting within the Batch Matrix [Higher half of address]
s_wait_kmcnt 0                                     // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s70        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s71       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_and_b32 s71, s[sgprGSU], 0xfff                   // Restore GSU
s_cmp_eq_u32 s71, 1                                // GSU == 1 ?
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_and_b32 s5, s[sgprArgType], 0xff                 // mask ArgType domain (bits 8+ = TDM wave id)
s_cmp_eq_u32 s5, 3                                 // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s71, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s70, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi to SRD
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s70, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s70, s70, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s71, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[72:73], s[70:71], 0                   // Load the Matrix Address in the Pointer Array
s_wait_kmcnt 0                                     // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s72        // Offsetting within the Batch Matrix [Lower half of address]
s_load_b64 s[70:71], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60 // Load batchOffsetD from kernel args
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s73       // Offsetting within the Batch Matrix [Higher half of address]
s_wait_kmcnt 0                                     // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // Add matrix address to SRD (high)
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD
s_and_b32 s5, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s5, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
s_mul_hi_u32 s71, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s72, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s70, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s72, s72, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s73, s72, s[sgprStrideC1J]            // Free1
s_mul_i32 s72, s72, s[sgprStrideC1J]               // Free1
s_add_u32 s70, s70, s72                            // Free1
s_addc_u32 s71, s71, s73                           // Free1
s_sub_u32 s72, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s72, s72, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s73, s72, s[sgprStrideCK]             // Free2
s_mul_i32 s72, s72, s[sgprStrideCK]                // Free2
s_add_u32 s70, s70, s72                            // Free2
s_addc_u32 s71, s71, s73                           // Free2
s_lshl_b64 s[70:71], s[70:71], 3                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
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
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc0 label_NoBranch_0                    // Only branch on scc1
s_getpc_b64 s[70:71]                               // addr of next instr
s_add_i32 s72, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s70, s70, s72                            // add target branch offset
s_addc_u32 s71, s71, 0                             // add high and carry
s_setpc_b64 s[70:71]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_0:
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x2000
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS1
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
s_wait_tensorcnt 0
label_skipPGR2_2:
s_wait_tensorcnt 1
s_barrier_signal -1
s_barrier_wait -1                                  // auto token transition barrier, sync LDS0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:8 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:520 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL == 1 (PGR>=2, not Suppress: single-loop -> toPGR1)
s_cbranch_scc1 label_toPGR1                        // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:
s_wait_dscnt 5                                     // <This is 1-cycle>
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0 <This is 2-cycle>
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0] <This is 3-cycle>
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1] <This is 4-cycle>
s_wait_dscnt 3                                     // <This is 5-cycle>
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0] <This is 6-cycle>
s_wait_dscnt 2                                     // <This is 7-cycle>
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1] <This is 8-cycle>
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x2000 // <This is 9-cycle>
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter? <This is 10-cycle>
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0] <This is 11-cycle>
s_cselect_b32 s46, s[sgprWrapUA+0], s[sgprtdmABIncs] // select WrapU or normal inc (lo) <This is 12-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br <This is 13-cycle>
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=4 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 13-cycle>
ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:2048 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 13-cycle>
ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:2560 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 13-cycle>
s_cselect_b32 s47, s[sgprWrapUA+1], 0              // select WrapU or normal inc (hi) <This is 13-cycle>
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL <This is 13-cycle>
s_add_u64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[46:47] // TDM addr += inc (with wrap, 64-bit) <This is 13-cycle>
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2 <This is 13-cycle>
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0] <This is 13-cycle>
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1] <This is 13-cycle>
v_add_f32 v106, -v110, 0                           // Ai=-Ai <This is 13-cycle>
v_add_f32 v118, -v108, 0                           // Ar=-Ar <This is 13-cycle>
v_add_f32 v119, -v109, 0                           // Ar=-Ar <This is 13-cycle>
v_add_f32 v116, -v110, 0                           // Ai=-Ai <This is 13-cycle>
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0] <This is 13-cycle>
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1] <This is 13-cycle>
s_wait_alu depctr_va_vdst(14)                      // <This is 13-cycle>
ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:2056 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 13-cycle>
ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:2568 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 13-cycle>
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=4 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0 <This is 13-cycle>
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1] <This is 14-cycle>
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0] <This is 15-cycle>
s_wait_dscnt 6                                     // <This is 15-cycle>
s_wait_tensorcnt 0                                 // <This is 15-cycle>
s_barrier_signal -1                                // <This is 15-cycle>
v_add_f32 v107, -v111, 0                           // Ai=-Ai <This is 16-cycle>
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2) // <This is 16-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[106:107], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi <This is 29-cycle>
v_add_f32 v117, -v111, 0                           // Ai=-Ai <This is 29-cycle>
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1] <This is 29-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br <This is 45-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[118:119], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi <This is 61-cycle>
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0] <This is 61-cycle>
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1] <This is 61-cycle>
s_delay_alu instid0(VALU_DEP_1)                    // <This is 61-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br <This is 77-cycle>
v_add_f32 v106, -v110, 0                           // Ai=-Ai <This is 77-cycle>
v_add_f32 v107, -v111, 0                           // Ai=-Ai <This is 77-cycle>
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0] <This is 77-cycle>
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1] <This is 77-cycle>
s_wait_alu depctr_vm_vsrc(1)                       // <This is 77-cycle>
v_xor_b32 v[vgprLocalReadAddrB], 0x2000, v[vgprLocalReadAddrB] // swap Red Blk <This is 77-cycle>
s_wait_alu depctr_vm_vsrc(0)                       // <This is 77-cycle>
v_xor_b32 v[vgprLocalReadAddrA], 0x2000, v[vgprLocalReadAddrA] // swap Red Blk <This is 77-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[106:107], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi <This is 93-cycle>
v_add_f32 v116, -v110, 0                           // Ai=-Ai <This is 93-cycle>
v_add_f32 v117, -v111, 0                           // Ai=-Ai <This is 93-cycle>
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0] <This is 93-cycle>
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1] <This is 93-cycle>
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1) // <This is 93-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br <This is 109-cycle>
v_add_f32 v118, -v108, 0                           // Ar=-Ar <This is 109-cycle>
v_add_f32 v119, -v109, 0                           // Ar=-Ar <This is 109-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[118:119], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi <This is 125-cycle>
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0] <This is 125-cycle>
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0] <This is 125-cycle>
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1] <This is 125-cycle>
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1] <This is 125-cycle>
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1) // <This is 125-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br <This is 141-cycle>
v_add_f32 v106, -v110, 0                           // Ai=-Ai <This is 141-cycle>
v_add_f32 v107, -v111, 0                           // Ai=-Ai <This is 141-cycle>
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0] <This is 141-cycle>
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1] <This is 141-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[106:107], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi <This is 157-cycle>
v_add_f32 v116, -v110, 0                           // Ai=-Ai <This is 157-cycle>
v_add_f32 v117, -v111, 0                           // Ai=-Ai <This is 157-cycle>
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0] <This is 157-cycle>
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1] <This is 157-cycle>
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1) // <This is 157-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br <This is 173-cycle>
v_add_f32 v118, -v108, 0                           // Ar=-Ar <This is 173-cycle>
v_add_f32 v119, -v109, 0                           // Ar=-Ar <This is 173-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[118:119], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi <This is 189-cycle>
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0] <This is 189-cycle>
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0] <This is 189-cycle>
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1] <This is 189-cycle>
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1] <This is 189-cycle>
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1) // <This is 189-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br <This is 205-cycle>
v_add_f32 v106, -v110, 0                           // Ai=-Ai <This is 205-cycle>
v_add_f32 v107, -v111, 0                           // Ai=-Ai <This is 205-cycle>
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0] <This is 205-cycle>
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1] <This is 205-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[106:107], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi <This is 221-cycle>
v_add_f32 v116, -v110, 0                           // Ai=-Ai <This is 221-cycle>
v_add_f32 v117, -v111, 0                           // Ai=-Ai <This is 221-cycle>
s_wait_dscnt 5                                     // <This is 221-cycle>
v_mov_b32 v110, v[vgprValuA_X1_I0+0+0+0+1]         // deint A im[0] <This is 221-cycle>
v_mov_b32 v111, v[vgprValuA_X1_I0+0+0+0+3]         // deint A im[1] <This is 221-cycle>
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1) // <This is 221-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br <This is 237-cycle>
v_add_f32 v118, -v108, 0                           // Ar=-Ar <This is 237-cycle>
v_add_f32 v119, -v109, 0                           // Ar=-Ar <This is 237-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[118:119], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi <This is 253-cycle>
v_mov_b32 v108, v[vgprValuA_X1_I0+0+0+0+0]         // deint A re[0] <This is 253-cycle>
s_wait_dscnt 4                                     // <This is 253-cycle>
v_mov_b32 v112, v[vgprValuB_X1_I0+0+0+0+0]         // deint B re[0] <This is 253-cycle>
s_wait_dscnt 3                                     // <This is 253-cycle>
v_mov_b32 v113, v[vgprValuB_X1_I0+0+0+0+2]         // deint B re[1] <This is 253-cycle>
v_mov_b32 v109, v[vgprValuA_X1_I0+0+0+0+2]         // deint A re[1] <This is 253-cycle>
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1) // <This is 253-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br <This is 269-cycle>
v_add_f32 v106, -v110, 0                           // Ai=-Ai <This is 269-cycle>
v_add_f32 v107, -v111, 0                           // Ai=-Ai <This is 269-cycle>
v_mov_b32 v114, v[vgprValuB_X1_I0+0+0+0+1]         // deint B im[0] <This is 269-cycle>
v_mov_b32 v115, v[vgprValuB_X1_I0+0+0+0+3]         // deint B im[1] <This is 269-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[106:107], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi <This is 285-cycle>
v_add_f32 v116, -v110, 0                           // Ai=-Ai <This is 285-cycle>
v_add_f32 v117, -v111, 0                           // Ai=-Ai <This is 285-cycle>
v_mov_b32 v110, v[vgprValuA_X1_I0+0+0+0+1]         // deint A im[0] <This is 285-cycle>
v_mov_b32 v111, v[vgprValuA_X1_I0+0+0+0+3]         // deint A im[1] <This is 285-cycle>
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1) // <This is 285-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br <This is 301-cycle>
v_add_f32 v118, -v108, 0                           // Ar=-Ar <This is 301-cycle>
v_add_f32 v119, -v109, 0                           // Ar=-Ar <This is 301-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[118:119], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi <This is 317-cycle>
v_mov_b32 v108, v[vgprValuA_X1_I0+0+0+0+0]         // deint A re[0] <This is 317-cycle>
s_wait_dscnt 2                                     // <This is 317-cycle>
v_mov_b32 v112, v[vgprValuB_X1_I0+4+0+0+0]         // deint B re[0] <This is 317-cycle>
s_wait_dscnt 1                                     // <This is 317-cycle>
v_mov_b32 v113, v[vgprValuB_X1_I0+4+0+0+2]         // deint B re[1] <This is 317-cycle>
v_mov_b32 v109, v[vgprValuA_X1_I0+0+0+0+2]         // deint A re[1] <This is 317-cycle>
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1) // <This is 317-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br <This is 333-cycle>
v_add_f32 v106, -v110, 0                           // Ai=-Ai <This is 333-cycle>
v_add_f32 v107, -v111, 0                           // Ai=-Ai <This is 333-cycle>
v_mov_b32 v114, v[vgprValuB_X1_I0+4+0+0+1]         // deint B im[0] <This is 333-cycle>
v_mov_b32 v115, v[vgprValuB_X1_I0+4+0+0+3]         // deint B im[1] <This is 333-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[106:107], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi <This is 349-cycle>
v_add_f32 v116, -v110, 0                           // Ai=-Ai <This is 349-cycle>
v_add_f32 v117, -v111, 0                           // Ai=-Ai <This is 349-cycle>
s_wait_dscnt 0                                     // <This is 349-cycle>
v_mov_b32 v110, v[vgprValuA_X1_I0+4+0+0+1]         // deint A im[0] <This is 349-cycle>
v_mov_b32 v111, v[vgprValuA_X1_I0+4+0+0+3]         // deint A im[1] <This is 349-cycle>
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1) // <This is 349-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br <This is 365-cycle>
v_add_f32 v118, -v108, 0                           // Ar=-Ar <This is 365-cycle>
v_add_f32 v119, -v109, 0                           // Ar=-Ar <This is 365-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[118:119], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi <This is 381-cycle>
v_mov_b32 v108, v[vgprValuA_X1_I0+4+0+0+0]         // deint A re[0] <This is 381-cycle>
v_mov_b32 v112, v[vgprValuB_X1_I0+0+0+0+0]         // deint B re[0] <This is 381-cycle>
v_mov_b32 v113, v[vgprValuB_X1_I0+0+0+0+2]         // deint B re[1] <This is 381-cycle>
v_mov_b32 v109, v[vgprValuA_X1_I0+4+0+0+2]         // deint A re[1] <This is 381-cycle>
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1) // <This is 381-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br <This is 397-cycle>
v_add_f32 v106, -v110, 0                           // Ai=-Ai <This is 397-cycle>
v_add_f32 v107, -v111, 0                           // Ai=-Ai <This is 397-cycle>
v_mov_b32 v114, v[vgprValuB_X1_I0+0+0+0+1]         // deint B im[0] <This is 397-cycle>
v_mov_b32 v115, v[vgprValuB_X1_I0+0+0+0+3]         // deint B im[1] <This is 397-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[106:107], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi <This is 413-cycle>
v_add_f32 v116, -v110, 0                           // Ai=-Ai <This is 413-cycle>
v_add_f32 v117, -v111, 0                           // Ai=-Ai <This is 413-cycle>
v_mov_b32 v110, v[vgprValuA_X1_I0+4+0+0+1]         // deint A im[0] <This is 413-cycle>
v_mov_b32 v111, v[vgprValuA_X1_I0+4+0+0+3]         // deint A im[1] <This is 413-cycle>
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1) // <This is 413-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br <This is 429-cycle>
v_add_f32 v118, -v108, 0                           // Ar=-Ar <This is 429-cycle>
v_add_f32 v119, -v109, 0                           // Ar=-Ar <This is 429-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[118:119], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi <This is 445-cycle>
v_mov_b32 v108, v[vgprValuA_X1_I0+4+0+0+0]         // deint A re[0] <This is 445-cycle>
v_mov_b32 v112, v[vgprValuB_X1_I0+4+0+0+0]         // deint B re[0] <This is 445-cycle>
v_mov_b32 v113, v[vgprValuB_X1_I0+4+0+0+2]         // deint B re[1] <This is 445-cycle>
v_mov_b32 v109, v[vgprValuA_X1_I0+4+0+0+2]         // deint A re[1] <This is 445-cycle>
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1) // <This is 445-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br <This is 461-cycle>
v_add_f32 v106, -v110, 0                           // Ai=-Ai <This is 461-cycle>
v_add_f32 v107, -v111, 0                           // Ai=-Ai <This is 461-cycle>
v_mov_b32 v114, v[vgprValuB_X1_I0+4+0+0+1]         // deint B im[0] <This is 461-cycle>
v_mov_b32 v115, v[vgprValuB_X1_I0+4+0+0+3]         // deint B im[1] <This is 461-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[106:107], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi <This is 477-cycle>
v_add_f32 v116, -v110, 0                           // Ai=-Ai <This is 477-cycle>
v_add_f32 v117, -v111, 0                           // Ai=-Ai <This is 477-cycle>
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1) // <This is 477-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br <This is 493-cycle>
v_add_f32 v118, -v108, 0                           // Ar=-Ar <This is 493-cycle>
v_add_f32 v119, -v109, 0                           // Ar=-Ar <This is 493-cycle>
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[118:119], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi <This is 509-cycle>
s_barrier_wait -1                                  // auto token transition barrier, sync LDS1 <This is 509-cycle>
s_barrier_signal -1                                // <This is 509-cycle>
s_barrier_wait -1                                  // auto token transition barrier, sync LDS0 <This is 509-cycle>
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS0 <This is 509-cycle>
s_wait_alu depctr_va_vdst(14)                      // <This is 509-cycle>
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1 <This is 509-cycle>
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1 <This is 509-cycle>
ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1 <This is 509-cycle>
ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1 <This is 509-cycle>
ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:8 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1 <This is 509-cycle>
ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:520 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1 <This is 509-cycle>
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL <This is 509-cycle>
label_LoopEndL:
s_wait_dscnt 5
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1]
s_wait_dscnt 3
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0]
s_wait_dscnt 2
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1]
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=4 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:2048 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:2560 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x2000
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_wait_alu depctr_va_vdst(13)
ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:2056 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
s_cselect_b32 s46, s[sgprWrapUA+0], s[sgprtdmABIncs] // select WrapU or normal inc (lo)
s_cselect_b32 s47, s[sgprWrapUA+1], 0              // select WrapU or normal inc (hi)
s_add_u64 s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[sgprtdmAGroup0+2:sgprtdmAGroup0+2+1], s[46:47] // TDM addr += inc (with wrap, 64-bit)
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1]
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1]
s_wait_alu depctr_va_vdst(14)
ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:2568 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=4 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS1
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1]
s_wait_alu depctr_vm_vsrc(1)
v_xor_b32 v[vgprLocalReadAddrB], 0x2000, v[vgprLocalReadAddrB] // swap Red Blk
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA], 0x2000, v[vgprLocalReadAddrA] // swap Red Blk
s_wait_dscnt 6
s_wait_tensorcnt 0
s_barrier_signal -1
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[106:107], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[118:119], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[106:107], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[118:119], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[106:107], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[118:119], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[106:107], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_wait_dscnt 5
v_mov_b32 v110, v[vgprValuA_X1_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+0+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[118:119], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+0+0+0+0]         // deint A re[0]
s_wait_dscnt 4
v_mov_b32 v112, v[vgprValuB_X1_I0+0+0+0+0]         // deint B re[0]
s_wait_dscnt 3
v_mov_b32 v113, v[vgprValuB_X1_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+0+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[106:107], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X1_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+0+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[118:119], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+0+0+0+0]         // deint A re[0]
s_wait_dscnt 2
v_mov_b32 v112, v[vgprValuB_X1_I0+4+0+0+0]         // deint B re[0]
s_wait_dscnt 1
v_mov_b32 v113, v[vgprValuB_X1_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+0+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[106:107], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_wait_dscnt 0
v_mov_b32 v110, v[vgprValuA_X1_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[118:119], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X1_I0+0+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X1_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[106:107], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X1_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[118:119], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X1_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X1_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[106:107], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[118:119], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi
s_barrier_wait -1                                  // auto token transition barrier, sync LDS0
s_wait_alu depctr_va_vdst(14)
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:8 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:520 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
label_toPGR1:
s_and_b32 s5, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s5, 1                                 // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1
s_mov_b32 s5, 0
s_or_b32 s70, s[sgprBeta+1], s[sgprBeta+0]         // tmp |= Beta[1] 
s_cmp_eq_u32 s70, s5                               // Beta == 0
s_cbranch_scc0 label_OptNLL_End                    // Branch if Beta is not zero
s_mov_b32 s70, 1.0                                 // Real part of 1.0
s_mov_b32 s71, 0.0                                 // Imaginary part of 1.0
s_cmp_eq_u64 s[sgprAlpha:sgprAlpha+1], s[70:71]    // Alpha == 1.0 ?
s_cbranch_scc0 label_OptNLL_End                    // branch if alpha != 1
s_and_b32 s70, 63, s[sgprSizeI]                    // s70 = s[sgprSizeI] % 64
s_mov_b32 s5, 0
s_add_u32 s71, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s71                // wg0 >= nwg0-1 ?
s_cselect_b32 s70, s70, 0                          // set rem
s_cmp_gt_u32 s70, s5                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s70, 63, s[sgprSizeJ]                    // s70 = s[sgprSizeJ] % 64
s_mov_b32 s5, 0
s_add_u32 s71, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s71                // wg1 >= nwg1-1
s_cselect_b32 s70, s70, 0                          // set rem
s_cmp_gt_u32 s70, s5                               // rem > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_and_b32 s71, 7, s[sgprSizesSum+0]                // s71 = s[sgprSizesSum+0] % 8
s_cmp_eq_u32 s71, 0                                // numIterL == 0
s_cbranch_scc0 label_OptNLL_End                    // skip if tail loop required
s_wait_dscnt 5
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1]
s_wait_dscnt 3
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0]
s_wait_dscnt 2
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1]
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=4 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:2048 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:2560 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0]
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br
s_wait_alu depctr_va_vdst(14)
ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:2056 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1]
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1]
s_wait_alu depctr_va_vdst(14)
ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:2568 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=4 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1]
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0]
s_wait_alu depctr_vm_vsrc(1)
v_xor_b32 v[vgprLocalReadAddrB], 0x2000, v[vgprLocalReadAddrB] // swap Red Blk
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA], 0x2000, v[vgprLocalReadAddrA] // swap Red Blk
v_add_f32 v107, -v111, 0                           // Ai=-Ai
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[106:107], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[118:119], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi
s_wait_dscnt 7
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0]
s_wait_dscnt 6
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[106:107], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[118:119], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[106:107], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[118:119], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[106:107], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_wait_dscnt 5
v_mov_b32 v110, v[vgprValuA_X1_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+0+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[118:119], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+0+0+0+0]         // deint A re[0]
s_wait_dscnt 4
v_mov_b32 v112, v[vgprValuB_X1_I0+0+0+0+0]         // deint B re[0]
s_wait_dscnt 3
v_mov_b32 v113, v[vgprValuB_X1_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+0+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[106:107], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X1_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+0+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[118:119], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+0+0+0+0]         // deint A re[0]
s_wait_dscnt 2
v_mov_b32 v112, v[vgprValuB_X1_I0+4+0+0+0]         // deint B re[0]
s_wait_dscnt 1
v_mov_b32 v113, v[vgprValuB_X1_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+0+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[106:107], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_wait_dscnt 0
v_mov_b32 v110, v[vgprValuA_X1_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[118:119], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X1_I0+0+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X1_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[106:107], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X1_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[118:119], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X1_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X1_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[106:107], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[118:119], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi
label_toPGR1end_OptNLL:
label_Summation_End_OptNLL:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v68, 5, v[vgprSerial]                // 68 = Serial / 32
v_and_b32 v64, 31, v[vgprSerial]                   // v64 = v[vgprSerial] % 32
s_mul_i32 s5, 64, s[sgprWorkGroup0]                // wgp0 * MT0
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_lshrrev_b32 v69, 1, v68                          // 69 = 68 / 2
v_lshrrev_b32 v64, 4, v64                          // 64 = 64 / 16
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_lshlrev_b32 v65, 4, v69                          // wave coordination offset 1
v_lshlrev_b32 v64, 3, v64                          // thread0 * continuous_output
v_and_b32 v69, 15, v[vgprSerial]                   // v69 = v[vgprSerial] % 16
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
v_add_lshl_u32 v65, v69, v65, 1                    // coordination 1 = vwB *(wave_id1 + tid1)
v_and_b32 v69, 1, v68                              // v69 = v68 % 2
v_mul_lo_u32 v66, v65, s[sgprStrideC1J]            //  offset 1
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
v_lshlrev_b32 v69, 4, v69                          // wave coordination offset 0
v_mul_lo_u32 v67, v65, s[sgprStrideD1J]            //  offset 1
v_add_lshl_u32 v64, v69, v64, 1                    // coordination 0 = vwA *(wave_id0 + tid0)
s_delay_alu instid0(VALU_DEP_1)
v_add_nc_u32 v64, s5, v64                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s5, 64, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v65, s5, v65                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
label_GW_B0_OptNLL_MB:
label_GW_B0_FD0_OptNLL_MB:
label_GW_B0_FD0_VW2_OptNLL_MB_Then:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
s_delay_alu instid0(VALU_DEP_2)
v_add_lshl_u32 v71, v67, v64, 3                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64 (multiple bpe)
v_mov_b32 v[vgprValuC+76], v[vgprValuC+0]          // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+77], v[vgprValuC+32]         // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+78], v[vgprValuC+8]          // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+79], v[vgprValuC+40]         // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+80], v[vgprValuC+1]          // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+81], v[vgprValuC+33]         // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+82], v[vgprValuC+9]          // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+83], v[vgprValuC+41]         // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+84], v[vgprValuC+2]          // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+85], v[vgprValuC+34]         // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+86], v[vgprValuC+10]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+87], v[vgprValuC+42]         // copy MI out reg to vreg[11]
v_mov_b32 v[vgprValuC+88], v[vgprValuC+3]          // copy MI out reg to vreg[12]
v_mov_b32 v[vgprValuC+89], v[vgprValuC+35]         // copy MI out reg to vreg[13]
v_mov_b32 v[vgprValuC+90], v[vgprValuC+11]         // copy MI out reg to vreg[14]
v_mov_b32 v[vgprValuC+91], v[vgprValuC+43]         // copy MI out reg to vreg[15]
v_mov_b32 v[vgprValuC+92], v[vgprValuC+4]          // copy MI out reg to vreg[16]
v_mov_b32 v[vgprValuC+93], v[vgprValuC+36]         // copy MI out reg to vreg[17]
v_mov_b32 v[vgprValuC+94], v[vgprValuC+12]         // copy MI out reg to vreg[18]
v_mov_b32 v[vgprValuC+95], v[vgprValuC+44]         // copy MI out reg to vreg[19]
v_mov_b32 v[vgprValuC+96], v[vgprValuC+5]          // copy MI out reg to vreg[20]
v_mov_b32 v[vgprValuC+97], v[vgprValuC+37]         // copy MI out reg to vreg[21]
v_mov_b32 v[vgprValuC+98], v[vgprValuC+13]         // copy MI out reg to vreg[22]
v_mov_b32 v[vgprValuC+99], v[vgprValuC+45]         // copy MI out reg to vreg[23]
v_mov_b32 v[vgprValuC+108], v[vgprValuC+6]         // copy MI out reg to vreg[24]
v_mov_b32 v[vgprValuC+109], v[vgprValuC+38]        // copy MI out reg to vreg[25]
v_mov_b32 v[vgprValuC+110], v[vgprValuC+14]        // copy MI out reg to vreg[26]
v_mov_b32 v[vgprValuC+111], v[vgprValuC+46]        // copy MI out reg to vreg[27]
v_mov_b32 v[vgprValuC+112], v[vgprValuC+7]         // copy MI out reg to vreg[28]
v_mov_b32 v[vgprValuC+113], v[vgprValuC+39]        // copy MI out reg to vreg[29]
v_mov_b32 v[vgprValuC+114], v[vgprValuC+15]        // copy MI out reg to vreg[30]
v_mov_b32 v[vgprValuC+115], v[vgprValuC+47]        // copy MI out reg to vreg[31]
v_mov_b32 v[vgprValuC+116], v[vgprValuC+16]        // copy MI out reg to vreg[32]
v_mov_b32 v[vgprValuC+117], v[vgprValuC+48]        // copy MI out reg to vreg[33]
v_mov_b32 v[vgprValuC+118], v[vgprValuC+24]        // copy MI out reg to vreg[34]
v_mov_b32 v[vgprValuC+119], v[vgprValuC+56]        // copy MI out reg to vreg[35]
v_mov_b32 v[vgprValuC+120], v[vgprValuC+17]        // copy MI out reg to vreg[36]
v_mov_b32 v[vgprValuC+121], v[vgprValuC+49]        // copy MI out reg to vreg[37]
v_mov_b32 v[vgprValuC+122], v[vgprValuC+25]        // copy MI out reg to vreg[38]
v_mov_b32 v[vgprValuC+123], v[vgprValuC+57]        // copy MI out reg to vreg[39]
v_mov_b32 v[vgprValuC+124], v[vgprValuC+18]        // copy MI out reg to vreg[40]
v_mov_b32 v[vgprValuC+125], v[vgprValuC+50]        // copy MI out reg to vreg[41]
v_mov_b32 v[vgprValuC+126], v[vgprValuC+26]        // copy MI out reg to vreg[42]
v_mov_b32 v[vgprValuC+127], v[vgprValuC+58]        // copy MI out reg to vreg[43]
v_mov_b32 v[vgprValuC+128], v[vgprValuC+19]        // copy MI out reg to vreg[44]
v_mov_b32 v[vgprValuC+129], v[vgprValuC+51]        // copy MI out reg to vreg[45]
v_mov_b32 v[vgprValuC+130], v[vgprValuC+27]        // copy MI out reg to vreg[46]
v_mov_b32 v[vgprValuC+131], v[vgprValuC+59]        // copy MI out reg to vreg[47]
v_mov_b32 v[vgprValuC+132], v[vgprValuC+20]        // copy MI out reg to vreg[48]
v_mov_b32 v[vgprValuC+133], v[vgprValuC+52]        // copy MI out reg to vreg[49]
v_mov_b32 v[vgprValuC+134], v[vgprValuC+28]        // copy MI out reg to vreg[50]
v_mov_b32 v[vgprValuC+135], v[vgprValuC+60]        // copy MI out reg to vreg[51]
v_mov_b32 v[vgprValuC+136], v[vgprValuC+21]        // copy MI out reg to vreg[52]
v_mov_b32 v[vgprValuC+137], v[vgprValuC+53]        // copy MI out reg to vreg[53]
v_mov_b32 v[vgprValuC+138], v[vgprValuC+29]        // copy MI out reg to vreg[54]
v_mov_b32 v[vgprValuC+139], v[vgprValuC+61]        // copy MI out reg to vreg[55]
v_mov_b32 v[vgprValuC+140], v[vgprValuC+22]        // copy MI out reg to vreg[56]
v_mov_b32 v[vgprValuC+141], v[vgprValuC+54]        // copy MI out reg to vreg[57]
v_mov_b32 v[vgprValuC+142], v[vgprValuC+30]        // copy MI out reg to vreg[58]
v_mov_b32 v[vgprValuC+143], v[vgprValuC+62]        // copy MI out reg to vreg[59]
v_mov_b32 v[vgprValuC+144], v[vgprValuC+23]        // copy MI out reg to vreg[60]
v_mov_b32 v[vgprValuC+145], v[vgprValuC+55]        // copy MI out reg to vreg[61]
v_mov_b32 v[vgprValuC+146], v[vgprValuC+31]        // copy MI out reg to vreg[62]
v_mov_b32 v[vgprValuC+147], v[vgprValuC+63]        // copy MI out reg to vreg[63]
s_wait_alu depctr_va_vdst(14)
buffer_store_b128 v[76:79], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[80:83], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[84:87], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:32 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[88:91], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:48 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[92:95], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:64 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[96:99], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:80 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[108:111], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:96 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[112:115], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:112 scope:SCOPE_CU th:TH_STORE_RT // store D
s_lshl_b32 s12, s[sgprStrideD1J], 3                // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b128 v[116:119], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[120:123], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[124:127], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:32 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[128:131], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:48 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(12)
buffer_store_b128 v[132:135], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:64 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(8)
buffer_store_b128 v[136:139], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:80 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(4)
buffer_store_b128 v[140:143], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:96 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[144:147], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:112 scope:SCOPE_CU th:TH_STORE_RT // store D
s_branch label_GW_End                              // jump to end
label_GW_End:
s_endpgm                                           // Kernel End
label_OptNLL_End:
label_GSU_3:
s_wait_dscnt 5
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1]
s_wait_dscnt 3
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0]
s_wait_dscnt 2
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1]
ds_load_b128 v[vgprValuA_X1_I0+0:vgprValuA_X1_I0+0+3], v[vgprLocalReadAddrA+0] offset:32 // L -> Reg lro=4 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X1_I0+0:vgprValuB_X1_I0+0+1], v[vgprLocalReadAddrB+0] offset:2048 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X1_I0+2:vgprValuB_X1_I0+2+1], v[vgprLocalReadAddrB+0] offset:2560 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0]
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br
s_wait_alu depctr_va_vdst(14)
ds_load_b64 v[vgprValuB_X1_I0+4:vgprValuB_X1_I0+4+1], v[vgprLocalReadAddrB+0] offset:2056 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1]
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1]
s_wait_alu depctr_va_vdst(14)
ds_load_b64 v[vgprValuB_X1_I0+6:vgprValuB_X1_I0+6+1], v[vgprLocalReadAddrB+0] offset:2568 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4:vgprValuA_X1_I0+4+3], v[vgprLocalReadAddrA+0] offset:96 // L -> Reg lro=4 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1]
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0]
s_wait_alu depctr_vm_vsrc(1)
v_xor_b32 v[vgprLocalReadAddrB], 0x2000, v[vgprLocalReadAddrB] // swap Red Blk
s_wait_alu depctr_vm_vsrc(0)
v_xor_b32 v[vgprLocalReadAddrA], 0x2000, v[vgprLocalReadAddrA] // swap Red Blk
v_add_f32 v107, -v111, 0                           // Ai=-Ai
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[106:107], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[118:119], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi
s_wait_dscnt 7
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0]
s_wait_dscnt 6
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[106:107], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[118:119], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[106:107], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[118:119], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[106:107], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_wait_dscnt 5
v_mov_b32 v110, v[vgprValuA_X1_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+0+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[118:119], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+0+0+0+0]         // deint A re[0]
s_wait_dscnt 4
v_mov_b32 v112, v[vgprValuB_X1_I0+0+0+0+0]         // deint B re[0]
s_wait_dscnt 3
v_mov_b32 v113, v[vgprValuB_X1_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+0+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[106:107], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X1_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+0+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[118:119], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+0+0+0+0]         // deint A re[0]
s_wait_dscnt 2
v_mov_b32 v112, v[vgprValuB_X1_I0+4+0+0+0]         // deint B re[0]
s_wait_dscnt 1
v_mov_b32 v113, v[vgprValuB_X1_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+0+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[106:107], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_wait_dscnt 0
v_mov_b32 v110, v[vgprValuA_X1_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[118:119], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X1_I0+0+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X1_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+0+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[106:107], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
v_mov_b32 v110, v[vgprValuA_X1_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v111, v[vgprValuA_X1_I0+4+0+0+3]         // deint A im[1]
s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[118:119], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X1_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v112, v[vgprValuB_X1_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v113, v[vgprValuB_X1_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v109, v[vgprValuA_X1_I0+4+0+0+2]         // deint A re[1]
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_mov_b32 v114, v[vgprValuB_X1_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v115, v[vgprValuB_X1_I0+4+0+0+3]         // deint B im[1]
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[106:107], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi
v_add_f32 v116, -v110, 0                           // Ai=-Ai
v_add_f32 v117, -v111, 0                           // Ai=-Ai
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br
v_add_f32 v118, -v108, 0                           // Ar=-Ar
v_add_f32 v119, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[118:119], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:
s_cmp_eq_u32 s[sgprOrigLoopCounter], 0             // skip if main loop was not executed
s_cbranch_scc1 label_SkipRemoveStagger             // skip removeStagger
s_sub_i32 s70, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s70, 0
s_cbranch_scc0 label_Negative_2
s_mul_hi_u32 s71, s70, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_branch label_MultiplyDone_3
label_Negative_2:
s_abs_i32 s70, s70
s_mul_hi_u32 s71, s70, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_xor_b32 s70, s70, 0xffffffff
s_xor_b32 s71, s71, 0xffffffff
s_add_u32 s70, s70, 0x1
s_addc_u32 s71, s71, 0
label_MultiplyDone_3:
s_sub_u64 s[70:71], s[70:71], s[sgprWrapUA:sgprWrapUA+1] // S - WrapU
s_bitcmp1_b32 s[sgprArgType], 8                    // check wave parity
s_cbranch_scc1 label_SkipRemoveStaggerA            // skip: odd waves do not carry A
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s70 // TDM addr += removeStagger offset (lo)
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], s71 // TDM addr += removeStagger offset (hi)
label_SkipRemoveStaggerA:
s_sub_i32 s70, 3, s[sgprStaggerUIter]
s_cmp_ge_i32 s70, 0
s_cbranch_scc0 label_Negative_4
s_mul_hi_u32 s71, s70, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_branch label_MultiplyDone_5
label_Negative_4:
s_abs_i32 s70, s70
s_mul_hi_u32 s71, s70, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_xor_b32 s70, s70, 0xffffffff
s_xor_b32 s71, s71, 0xffffffff
s_add_u32 s70, s70, 0x1
s_addc_u32 s71, s71, 0
label_MultiplyDone_5:
s_sub_u64 s[70:71], s[70:71], s[sgprWrapUB:sgprWrapUB+1] // S - WrapU
s_bitcmp1_b32 s[sgprArgType], 8                    // check wave parity
s_cbranch_scc0 label_SkipRemoveStaggerB            // skip: even waves do not carry B
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s70 // TDM addr += removeStagger offset (lo)
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], s71 // TDM addr += removeStagger offset (hi)
label_SkipRemoveStaggerB:
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
label_SkipRemoveStagger:
s_and_b32 s[sgprLoopCounterL], 7, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 8
s_and_b32 s70, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s71, s[sgprSizesSum], 3                 // s71 = s[sgprSizesSum] / 8
s_and_b32 s72, s[sgprGSU], 0xfff                   // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v68, s72                             // s70 = s71 / s72
s_delay_alu instid0(VALU_DEP_1)
v_rcp_iflag_f32 v68, v68                           // s70 = s71 / s72
v_cvt_f32_u32 v69, s71                             // s70 = s71 / s72
s_delay_alu instid0(TRANS32_DEP_1) | instskip(SAME) | instid1(VALU_DEP_1)
v_mul_f32 v68, v68, v69                            // s70 = s71 / s72
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_cvt_u32_f32 v68, v68                             // s70 = s71 / s72
v_mul_u32_u24 v69, v68, s72                        // s70 = s71 / s72
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_sub_nc_u32 v69, s71, v69                         // s70 = s71 / s72
v_cmp_eq_u32 vcc_lo, v69, s72                      // s70 = s71 / s72
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // s70 = s71 / s72
v_add_nc_u32 v68, 1, v68                           // s70 = s71 / s72
v_mov_b32 v69, 0                                   // s[sgprGSUSumIdx+1] = s71 % s72
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
v_cmp_gt_u32 vcc_lo, v69, s72                      // overflow happened in remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v68, v68, 1                           // quotient - 1
v_mul_u32_u24 v69, v68, s72                        // re-calculate remainder
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
v_sub_nc_u32 v69, s71, v69                         // re-calculate remainder
s_wait_alu depctr_va_vdst(0)
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s70, v68                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v69        // remainder
s_sub_u32 s71, s72, 1                              // GSU-1
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
s_cmp_eq_u32 s70, 0                                // quotient == 0
s_cselect_b32 s70, s[sgprGSUSumIdx+1], s71         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s70                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0                  // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0
s_barrier_signal -1
s_barrier_wait -1                                  // Barrier before tail TDM loads (WAR hazard with NLL LDS reads)
label_TDMResetTailA:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s46, v[vgprSerial]             // first tId
s_delay_alu instid0(VALU_DEP_1)
s_lshr_b32 s46, s46, 5                             // wId=fTid // wavelen
s_bitcmp1_b32 s46, 0                               // Check parity of wId
s_cbranch_scc1 label_TDMResetTailB                 // Jump to B on its wave parity
s_and_b32 s5, s[sgprSizeL], 7
s_and_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], 0xffff
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff0000
s_lshl_b32 s7, s5, 0x10
s_or_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], s7
s_lshr_b32 s7, s5, 0x10
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s7
s_branch label_TDMResetTailABEnd
label_TDMResetTailB:
s_lshr_b32 s47, s46, 1                             // wOffset=B compId
s_mul_i32 s47, s47, 4                              // wOffset = wOffset * du // numpComp
s_and_b32 s5, s[sgprSizeL], 7
s_sub_u32 s5, s5, s47                              // consider multiple waves
s_cmov_b32 s5, 0                                   // set to 0 for waves that no enough data to load
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff0000
s_lshl_b32 s7, s5, 0x10
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s7
s_lshr_b32 s7, s5, 0x10
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], s7
label_TDMResetTailABEnd:
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS0
s_wait_loadcnt 0
s_wait_tensorcnt 0                                 // 2wait for global read
s_barrier_signal -1
s_barrier_wait -1
s_wait_dscnt 0                                     // 5wait for local write
s_barrier_signal -1
s_barrier_wait -1                                  // Tail loop LW->LR, sync LDS0
.set vgprValuA_X0_I0_BASE, 68
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+8
.set vgprValuB_X0_I0_BASE, 84
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+8
.align 16
label_TailLoopBeginL:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
s_wait_alu depctr_va_vdst(1)
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB+0] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB+0] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB+0] offset:8 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b64 v[vgprValuB_X0_I0+6:vgprValuB_X0_I0+6+1], v[vgprLocalReadAddrB+0] offset:520 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_mov_b32 s5, 32                                   // inc
s_wait_alu depctr_vm_vsrc(4)
v_add_co_u32 v[vgprLocalReadAddrA+0], vcc_lo, s5, v[vgprLocalReadAddrA+0] // lrA += 32 (bpeDS)
s_mov_b32 s5, 2048                                 // inc
s_wait_alu depctr_vm_vsrc(0)
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, s5, v[vgprLocalReadAddrB+0] // lrB += 2048 ((MT+PAD)*bpeDS)
s_wait_dscnt 0                                     // 4wait for local read
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0]
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1]
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1]
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0]
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1]
v_add_f32 v100, -v110, 0                           // Ai=-Ai
v_add_f32 v101, -v111, 0                           // Ai=-Ai
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_add_f32 v116, -v108, 0                           // Ar=-Ar
v_add_f32 v117, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[108:109], v[112:113], v[vgprValuC+0:vgprValuC+0+7] // Cr += Ar*Br
v_wmma_f32_16x16x4_f32 v[vgprValuC+0:vgprValuC+0+7], v[100:101], v[114:115], v[vgprValuC+0:vgprValuC+0+7] // Cr += -Ai*Bi
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[106:107], v[112:113], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ai*Br
s_delay_alu instid0(VALU_DEP_4)
v_wmma_f32_16x16x4_f32 v[vgprValuC+32:vgprValuC+32+7], v[116:117], v[114:115], v[vgprValuC+32:vgprValuC+32+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+0+0+0+0]         // deint A re[0]
v_mov_b32 v110, v[vgprValuA_X0_I0+0+0+0+1]         // deint A im[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+0+0+0+2]         // deint A re[1]
v_mov_b32 v111, v[vgprValuA_X0_I0+0+0+0+3]         // deint A im[1]
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1]
v_add_f32 v100, -v110, 0                           // Ai=-Ai
v_add_f32 v101, -v111, 0                           // Ai=-Ai
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_add_f32 v116, -v108, 0                           // Ar=-Ar
v_add_f32 v117, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[108:109], v[112:113], v[vgprValuC+16:vgprValuC+16+7] // Cr += Ar*Br
v_wmma_f32_16x16x4_f32 v[vgprValuC+16:vgprValuC+16+7], v[100:101], v[114:115], v[vgprValuC+16:vgprValuC+16+7] // Cr += -Ai*Bi
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[106:107], v[112:113], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ai*Br
s_delay_alu instid0(VALU_DEP_4)
v_wmma_f32_16x16x4_f32 v[vgprValuC+48:vgprValuC+48+7], v[116:117], v[114:115], v[vgprValuC+48:vgprValuC+48+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1]
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1]
v_mov_b32 v112, v[vgprValuB_X0_I0+0+0+0+0]         // deint B re[0]
v_mov_b32 v114, v[vgprValuB_X0_I0+0+0+0+1]         // deint B im[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+0+0+0+2]         // deint B re[1]
v_mov_b32 v115, v[vgprValuB_X0_I0+0+0+0+3]         // deint B im[1]
v_add_f32 v100, -v110, 0                           // Ai=-Ai
v_add_f32 v101, -v111, 0                           // Ai=-Ai
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_add_f32 v116, -v108, 0                           // Ar=-Ar
v_add_f32 v117, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[108:109], v[112:113], v[vgprValuC+8:vgprValuC+8+7] // Cr += Ar*Br
v_wmma_f32_16x16x4_f32 v[vgprValuC+8:vgprValuC+8+7], v[100:101], v[114:115], v[vgprValuC+8:vgprValuC+8+7] // Cr += -Ai*Bi
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[106:107], v[112:113], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ai*Br
s_delay_alu instid0(VALU_DEP_4)
v_wmma_f32_16x16x4_f32 v[vgprValuC+40:vgprValuC+40+7], v[116:117], v[114:115], v[vgprValuC+40:vgprValuC+40+7] // Ci += -Ar*Bi
v_mov_b32 v108, v[vgprValuA_X0_I0+4+0+0+0]         // deint A re[0]
v_mov_b32 v110, v[vgprValuA_X0_I0+4+0+0+1]         // deint A im[0]
v_mov_b32 v109, v[vgprValuA_X0_I0+4+0+0+2]         // deint A re[1]
v_mov_b32 v111, v[vgprValuA_X0_I0+4+0+0+3]         // deint A im[1]
v_mov_b32 v112, v[vgprValuB_X0_I0+4+0+0+0]         // deint B re[0]
v_mov_b32 v114, v[vgprValuB_X0_I0+4+0+0+1]         // deint B im[0]
v_mov_b32 v113, v[vgprValuB_X0_I0+4+0+0+2]         // deint B re[1]
v_mov_b32 v115, v[vgprValuB_X0_I0+4+0+0+3]         // deint B im[1]
v_add_f32 v100, -v110, 0                           // Ai=-Ai
v_add_f32 v101, -v111, 0                           // Ai=-Ai
v_add_f32 v106, -v110, 0                           // Ai=-Ai
v_add_f32 v107, -v111, 0                           // Ai=-Ai
v_add_f32 v116, -v108, 0                           // Ar=-Ar
v_add_f32 v117, -v109, 0                           // Ar=-Ar
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[108:109], v[112:113], v[vgprValuC+24:vgprValuC+24+7] // Cr += Ar*Br
v_wmma_f32_16x16x4_f32 v[vgprValuC+24:vgprValuC+24+7], v[100:101], v[114:115], v[vgprValuC+24:vgprValuC+24+7] // Cr += -Ai*Bi
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[106:107], v[112:113], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ai*Br
s_delay_alu instid0(VALU_DEP_4)
v_wmma_f32_16x16x4_f32 v[vgprValuC+56:vgprValuC+56+7], v[116:117], v[114:115], v[vgprValuC+56:vgprValuC+56+7] // Ci += -Ar*Bi
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x4 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x4 // inc counterL
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
label_Summation_End_6:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprtdmABIncs, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprWrapUA, UNDEF
.set sgprtdmAGroup0, UNDEF
.set sgprtdmAGroup1, UNDEF
.set sgprWrapUB, UNDEF
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v68, 5, v[vgprSerial]                // 68 = Serial / 32
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_lshrrev_b32 v69, 1, v68                          // 69 = 68 / 2
v_lshlrev_b32 v65, 4, v69                          // wave coordination offset 1
v_and_b32 v69, 15, v[vgprSerial]                   // v69 = v[vgprSerial] % 16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_add_lshl_u32 v65, v69, v65, 1                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v66, v65, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v67, v65, s[sgprStrideD1J]            //  offset 1
v_and_b32 v69, 1, v68                              // v69 = v68 % 2
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_lshlrev_b32 v69, 4, v69                          // wave coordination offset 0
v_and_b32 v64, 31, v[vgprSerial]                   // v64 = v[vgprSerial] % 32
v_lshrrev_b32 v64, 4, v64                          // 64 = 64 / 16
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_lshlrev_b32 v64, 3, v64                          // thread0 * continuous_output
v_add_lshl_u32 v64, v69, v64, 1                    // coordination 0 = vwA *(wave_id0 + tid0)
s_mul_i32 s5, 64, s[sgprWorkGroup0]                // wgp0 * MT0
s_delay_alu instid0(VALU_DEP_1)
v_add_nc_u32 v64, s5, v64                          // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s5, 64, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_nc_u32 v65, s5, v65                          // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
s_and_b32 s5, s[sgprGSU], 0xfff                    // Restore GSU
s_cmp_eq_u32 s5, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_4                         // branch if GSU == 1
label_GW_B0_MB:
label_GW_B0_FD0_MB:
s_and_b32 s32, 63, s[sgprSizeI]                    // s32 = s[sgprSizeI] % 64
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rem
s_mov_b32 s5, 0
s_cmp_gt_u32 s32, s5                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW2_MB_Else         // jump if edges required
s_and_b32 s32, 63, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 64
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rem
s_mov_b32 s5, 0
s_cmp_gt_u32 s32, s5                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW2_MB_Then         // jump if edges required
label_GW_B0_FD0_VW2_MB_NonEdge:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_add_lshl_u32 v71, v67, v64, 3                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64 (multiple bpe)
v_mov_b32 v[vgprValuC+76], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76+1], v[vgprValuC+32]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78+1], v[vgprValuC+40]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80+1], v[vgprValuC+33]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82+1], v[vgprValuC+41]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84+1], v[vgprValuC+34]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86+1], v[vgprValuC+42]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88+1], v[vgprValuC+35]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90], v[vgprValuC+11]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90+1], v[vgprValuC+43]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92+1], v[vgprValuC+36]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+12]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94+1], v[vgprValuC+44]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96+1], v[vgprValuC+37]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98], v[vgprValuC+13]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98+1], v[vgprValuC+45]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+6]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108+1], v[vgprValuC+38]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+14]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110+1], v[vgprValuC+46]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112+1], v[vgprValuC+39]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114+1], v[vgprValuC+47]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116+1], v[vgprValuC+48]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118+1], v[vgprValuC+56]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120+1], v[vgprValuC+49]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122+1], v[vgprValuC+57]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124+1], v[vgprValuC+50]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126+1], v[vgprValuC+58]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128+1], v[vgprValuC+51]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130+1], v[vgprValuC+59]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132+1], v[vgprValuC+52]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134+1], v[vgprValuC+60]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136+1], v[vgprValuC+53]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138+1], v[vgprValuC+61]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140+1], v[vgprValuC+54]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142+1], v[vgprValuC+62]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+144+1], v[vgprValuC+55]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+146+1], v[vgprValuC+63]      // Rearrange MI out reg
s_wait_alu depctr_va_vdst(14)
buffer_store_b128 v[76:79], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[80:83], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[84:87], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:32 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[88:91], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:48 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[92:95], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:64 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[96:99], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:80 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[108:111], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:96 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[112:115], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:112 scope:SCOPE_CU th:TH_STORE_RT // store D
s_lshl_b32 s12, s[sgprStrideD1J], 3                // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b128 v[116:119], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[120:123], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[124:127], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:32 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[128:131], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:48 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(12)
buffer_store_b128 v[132:135], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:64 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(8)
buffer_store_b128 v[136:139], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:80 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(4)
buffer_store_b128 v[140:143], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:96 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[144:147], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:112 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW2_MB_NonEdgeEnd:
label_GW_B0_FD0_VW2_MB_Then:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v70, BufferOOB
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v71, v67, v64, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v70, v71, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v100, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v100, v70, v100, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v101, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v70, v101, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v105, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v105, v70, v105, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v106, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v70, v106, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v107, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v107, v70, v107, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v144, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v70, v144, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v145, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
v_cndmask_b32 v145, v70, v145, s14                 // LDD clip if OOB. offset
v_add_co_u32 v65, vcc_lo, v65, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_add_nc_u32 v66, v66, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v67, v67, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v146, v67, v64, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v146, v70, v146, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v147, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v70, v147, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v148, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v148, v70, v148, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v149, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v70, v149, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v150, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v150, v70, v150, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v151, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v70, v151, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v152, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v152, v70, v152, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v153, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v70, v153, s14                 // LDD clip if OOB. offset
v_mov_b32 v[vgprValuC+72], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+72+1], v[vgprValuC+32]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+74], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+74+1], v[vgprValuC+40]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76+1], v[vgprValuC+33]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78+1], v[vgprValuC+41]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80+1], v[vgprValuC+34]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82+1], v[vgprValuC+42]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84+1], v[vgprValuC+35]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86], v[vgprValuC+11]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86+1], v[vgprValuC+43]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88+1], v[vgprValuC+36]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90], v[vgprValuC+12]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90+1], v[vgprValuC+44]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92+1], v[vgprValuC+37]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+13]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94+1], v[vgprValuC+45]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96+1], v[vgprValuC+38]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98], v[vgprValuC+14]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98+1], v[vgprValuC+46]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108+1], v[vgprValuC+39]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110+1], v[vgprValuC+47]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112+1], v[vgprValuC+48]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114+1], v[vgprValuC+56]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116+1], v[vgprValuC+49]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118+1], v[vgprValuC+57]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120+1], v[vgprValuC+50]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122+1], v[vgprValuC+58]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124+1], v[vgprValuC+51]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126+1], v[vgprValuC+59]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128+1], v[vgprValuC+52]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130+1], v[vgprValuC+60]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132+1], v[vgprValuC+53]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134+1], v[vgprValuC+61]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136+1], v[vgprValuC+54]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138+1], v[vgprValuC+62]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+140+1], v[vgprValuC+55]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+142+1], v[vgprValuC+63]      // Rearrange MI out reg
s_wait_alu depctr_va_vdst(14)
buffer_store_b128 v[72:75], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[76:79], v100, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[80:83], v101, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[84:87], v105, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[88:91], v106, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[92:95], v107, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[96:99], v144, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[108:111], v145, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[112:115], v146, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[116:119], v147, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[120:123], v148, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[124:127], v149, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(12)
buffer_store_b128 v[128:131], v150, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(8)
buffer_store_b128 v[132:135], v151, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(4)
buffer_store_b128 v[136:139], v152, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[140:143], v153, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_FD0_VW2_MB_Else:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v70, BufferOOB
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v71, v67, v64, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v70, v71, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v105, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v105, v70, v105, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v140, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v70, v140, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 3                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v141, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v141, v70, v141, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v142, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v70, v142, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v143, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v143, v70, v143, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v144, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v70, v144, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 7                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v145, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v145, v70, v145, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v146, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v70, v146, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v147, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v147, v70, v147, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v148, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v70, v148, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 11                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v149, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v149, v70, v149, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v150, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v70, v150, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v151, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v151, v70, v151, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v152, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v70, v152, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 15                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v153, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
v_cndmask_b32 v153, v70, v153, s14                 // LDD clip if OOB. offset
v_add_co_u32 v65, vcc_lo, v65, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_add_nc_u32 v66, v66, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v67, v67, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v154, v67, v64, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v154, v70, v154, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v155, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v70, v155, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v156, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v156, v70, v156, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 3                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v157, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v70, v157, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v158, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v158, v70, v158, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v159, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v70, v159, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v160, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v160, v70, v160, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 7                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v161, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v70, v161, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v162, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v162, v70, v162, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v163, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v70, v163, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v164, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v164, v70, v164, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 11                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v165, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v70, v165, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v166, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v166, v70, v166, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v167, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v70, v167, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v168, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v168, v70, v168, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 15                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v169, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v70, v169, s14                 // LDD clip if OOB. offset
v_mov_b32 v[vgprValuC+72], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+72+1], v[vgprValuC+32]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+74], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+74+1], v[vgprValuC+40]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+76+1], v[vgprValuC+33]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+78+1], v[vgprValuC+41]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+80+1], v[vgprValuC+34]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+82+1], v[vgprValuC+42]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+84+1], v[vgprValuC+35]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86], v[vgprValuC+11]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+86+1], v[vgprValuC+43]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+88+1], v[vgprValuC+36]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90], v[vgprValuC+12]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+90+1], v[vgprValuC+44]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+92+1], v[vgprValuC+37]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94], v[vgprValuC+13]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+94+1], v[vgprValuC+45]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+96+1], v[vgprValuC+38]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98], v[vgprValuC+14]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+98+1], v[vgprValuC+46]       // Rearrange MI out reg
v_mov_b32 v[vgprValuC+100], v[vgprValuC+7]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+100+1], v[vgprValuC+39]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+106], v[vgprValuC+15]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+106+1], v[vgprValuC+47]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108], v[vgprValuC+16]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+108+1], v[vgprValuC+48]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110], v[vgprValuC+24]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+110+1], v[vgprValuC+56]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112], v[vgprValuC+17]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+112+1], v[vgprValuC+49]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114], v[vgprValuC+25]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+114+1], v[vgprValuC+57]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116], v[vgprValuC+18]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+116+1], v[vgprValuC+50]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118], v[vgprValuC+26]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+118+1], v[vgprValuC+58]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120], v[vgprValuC+19]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+120+1], v[vgprValuC+51]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122], v[vgprValuC+27]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+122+1], v[vgprValuC+59]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124], v[vgprValuC+20]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+124+1], v[vgprValuC+52]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126], v[vgprValuC+28]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+126+1], v[vgprValuC+60]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128], v[vgprValuC+21]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+128+1], v[vgprValuC+53]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130], v[vgprValuC+29]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+130+1], v[vgprValuC+61]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132], v[vgprValuC+22]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+132+1], v[vgprValuC+54]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134], v[vgprValuC+30]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+134+1], v[vgprValuC+62]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136], v[vgprValuC+23]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+136+1], v[vgprValuC+55]      // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138], v[vgprValuC+31]        // Rearrange MI out reg
v_mov_b32 v[vgprValuC+138+1], v[vgprValuC+63]      // Rearrange MI out reg
s_wait_alu depctr_va_vdst(14)
buffer_store_b64 v[72:73], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[74:75], v105, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[76:77], v140, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[78:79], v141, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[80:81], v142, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[82:83], v143, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[84:85], v144, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[86:87], v145, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[88:89], v146, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[90:91], v147, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[92:93], v148, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[94:95], v149, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[96:97], v150, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[98:99], v151, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[100:101], v152, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[106:107], v153, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[108:109], v154, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[110:111], v155, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[112:113], v156, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[114:115], v157, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[116:117], v158, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[118:119], v159, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[120:121], v160, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[122:123], v161, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[124:125], v162, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(12)
buffer_store_b64 v[126:127], v163, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[128:129], v164, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(8)
buffer_store_b64 v[130:131], v165, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[132:133], v166, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(4)
buffer_store_b64 v[134:135], v167, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[136:137], v168, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b64 v[138:139], v169, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s14, label_KernelEnd, 4                  // target branch offset
s_add_u32 s12, s12, s14                            // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_setpc_b64 s[12:13]                               // branch to label_KernelEnd
label_GSU_4:
s_or_b32 s5, s[sgprBeta+1], s[sgprBeta+0]          // tmp |= Beta[1] 
s_mov_b32 s7, 0
s_cmp_eq_u32 s5, s7                                // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero
label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:
s_and_b32 s32, 63, s[sgprSizeI]                    // s32 = s[sgprSizeI] % 64
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rem
s_mov_b32 s5, 0
s_cmp_gt_u32 s32, s5                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW2_GSU1_Else       // jump if edges required
s_and_b32 s32, 63, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 64
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rem
s_mov_b32 s5, 0
s_cmp_gt_u32 s32, s5                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW2_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW2_GSU1_NonEdge:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_add_lshl_u32 v71, v67, v64, 3                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64 (multiple bpe)
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+0]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+0]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+32], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+32], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+8]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+8]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+40], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+40], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+1]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+1]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+33], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+33], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+9]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+9]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+41], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+41], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+2]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+2]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+84], s[sgprAlpha+1], -v[vgprValuC+34], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+84+1], s[sgprAlpha+0], v[vgprValuC+34], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+10]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+10]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+86], s[sgprAlpha+1], -v[vgprValuC+42], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+86+1], s[sgprAlpha+0], v[vgprValuC+42], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+3]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+3]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+88], s[sgprAlpha+1], -v[vgprValuC+35], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+88+1], s[sgprAlpha+0], v[vgprValuC+35], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+11]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+11]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+90], s[sgprAlpha+1], -v[vgprValuC+43], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+90+1], s[sgprAlpha+0], v[vgprValuC+43], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+4]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+4]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+92], s[sgprAlpha+1], -v[vgprValuC+36], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+92+1], s[sgprAlpha+0], v[vgprValuC+36], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+12]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+12]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+94], s[sgprAlpha+1], -v[vgprValuC+44], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+94+1], s[sgprAlpha+0], v[vgprValuC+44], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+5]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+5]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+96], s[sgprAlpha+1], -v[vgprValuC+37], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+96+1], s[sgprAlpha+0], v[vgprValuC+37], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+13]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+13]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+98], s[sgprAlpha+1], -v[vgprValuC+45], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+98+1], s[sgprAlpha+0], v[vgprValuC+45], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+6]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+6]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+108], s[sgprAlpha+1], -v[vgprValuC+38], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+108+1], s[sgprAlpha+0], v[vgprValuC+38], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+14]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+14]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+110], s[sgprAlpha+1], -v[vgprValuC+46], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+110+1], s[sgprAlpha+0], v[vgprValuC+46], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+7]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+7]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+112], s[sgprAlpha+1], -v[vgprValuC+39], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+112+1], s[sgprAlpha+0], v[vgprValuC+39], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+15]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+15]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+114], s[sgprAlpha+1], -v[vgprValuC+47], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+114+1], s[sgprAlpha+0], v[vgprValuC+47], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+16]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+16]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+116], s[sgprAlpha+1], -v[vgprValuC+48], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+116+1], s[sgprAlpha+0], v[vgprValuC+48], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+24]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+24]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+118], s[sgprAlpha+1], -v[vgprValuC+56], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+118+1], s[sgprAlpha+0], v[vgprValuC+56], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+17]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+17]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+120], s[sgprAlpha+1], -v[vgprValuC+49], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+120+1], s[sgprAlpha+0], v[vgprValuC+49], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+25]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+25]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+122], s[sgprAlpha+1], -v[vgprValuC+57], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+122+1], s[sgprAlpha+0], v[vgprValuC+57], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+18]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+18]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+124], s[sgprAlpha+1], -v[vgprValuC+50], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+124+1], s[sgprAlpha+0], v[vgprValuC+50], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+26]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+26]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+126], s[sgprAlpha+1], -v[vgprValuC+58], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+126+1], s[sgprAlpha+0], v[vgprValuC+58], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+19]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+19]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+128], s[sgprAlpha+1], -v[vgprValuC+51], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+128+1], s[sgprAlpha+0], v[vgprValuC+51], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+27]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+27]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+130], s[sgprAlpha+1], -v[vgprValuC+59], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+130+1], s[sgprAlpha+0], v[vgprValuC+59], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+20]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+20]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+132], s[sgprAlpha+1], -v[vgprValuC+52], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+132+1], s[sgprAlpha+0], v[vgprValuC+52], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+28]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+28]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+134], s[sgprAlpha+1], -v[vgprValuC+60], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+134+1], s[sgprAlpha+0], v[vgprValuC+60], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+21]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+21]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+136], s[sgprAlpha+1], -v[vgprValuC+53], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+136+1], s[sgprAlpha+0], v[vgprValuC+53], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+29]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+29]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+138], s[sgprAlpha+1], -v[vgprValuC+61], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+138+1], s[sgprAlpha+0], v[vgprValuC+61], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+22]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+22]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+140], s[sgprAlpha+1], -v[vgprValuC+54], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+140+1], s[sgprAlpha+0], v[vgprValuC+54], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+30]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+30]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+142], s[sgprAlpha+1], -v[vgprValuC+62], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+142+1], s[sgprAlpha+0], v[vgprValuC+62], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+23]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+23]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+144], s[sgprAlpha+1], -v[vgprValuC+55], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+144+1], s[sgprAlpha+0], v[vgprValuC+55], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+31]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+31]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+146], s[sgprAlpha+1], -v[vgprValuC+63], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+146+1], s[sgprAlpha+0], v[vgprValuC+63], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_alu depctr_va_vdst(14)
buffer_store_b128 v[76:79], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[80:83], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[84:87], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:32 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[88:91], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:48 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[92:95], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:64 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[96:99], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:80 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[108:111], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:96 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[112:115], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:112 scope:SCOPE_CU th:TH_STORE_RT // store D
s_lshl_b32 s12, s[sgprStrideD1J], 3                // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b128 v[116:119], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[120:123], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[124:127], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:32 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[128:131], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:48 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[132:135], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:64 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[136:139], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:80 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(8)
buffer_store_b128 v[140:143], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:96 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[144:147], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:112 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW2_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW2_GSU1_Then:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v70, BufferOOB
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v71, v67, v64, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v70, v71, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v100, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v100, v70, v100, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v101, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v70, v101, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v105, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v105, v70, v105, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v106, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v70, v106, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v107, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v107, v70, v107, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v144, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v70, v144, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v145, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
v_cndmask_b32 v145, v70, v145, s14                 // LDD clip if OOB. offset
v_add_co_u32 v65, vcc_lo, v65, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_add_nc_u32 v66, v66, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v67, v67, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v146, v67, v64, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v146, v70, v146, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v147, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v147, v70, v147, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v148, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v148, v70, v148, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v149, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v70, v149, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v150, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v150, v70, v150, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v151, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v151, v70, v151, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v152, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v152, v70, v152, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v153, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v153, v70, v153, s14                 // LDD clip if OOB. offset
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+0]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+0]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+72], s[sgprAlpha+1], -v[vgprValuC+32], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+72+1], s[sgprAlpha+0], v[vgprValuC+32], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+8]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+8]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+74], s[sgprAlpha+1], -v[vgprValuC+40], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+74+1], s[sgprAlpha+0], v[vgprValuC+40], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+1]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+1]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+33], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+33], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+9]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+9]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+41], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+41], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+2]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+2]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+34], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+34], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+10]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+10]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+42], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+42], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+3]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+3]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+84], s[sgprAlpha+1], -v[vgprValuC+35], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+84+1], s[sgprAlpha+0], v[vgprValuC+35], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+11]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+11]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+86], s[sgprAlpha+1], -v[vgprValuC+43], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+86+1], s[sgprAlpha+0], v[vgprValuC+43], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+4]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+4]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+88], s[sgprAlpha+1], -v[vgprValuC+36], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+88+1], s[sgprAlpha+0], v[vgprValuC+36], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+12]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+12]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+90], s[sgprAlpha+1], -v[vgprValuC+44], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+90+1], s[sgprAlpha+0], v[vgprValuC+44], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+5]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+5]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+92], s[sgprAlpha+1], -v[vgprValuC+37], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+92+1], s[sgprAlpha+0], v[vgprValuC+37], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+13]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+13]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+94], s[sgprAlpha+1], -v[vgprValuC+45], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+94+1], s[sgprAlpha+0], v[vgprValuC+45], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+6]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+6]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+96], s[sgprAlpha+1], -v[vgprValuC+38], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+96+1], s[sgprAlpha+0], v[vgprValuC+38], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+14]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+14]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+98], s[sgprAlpha+1], -v[vgprValuC+46], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+98+1], s[sgprAlpha+0], v[vgprValuC+46], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+7]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+7]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+108], s[sgprAlpha+1], -v[vgprValuC+39], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+108+1], s[sgprAlpha+0], v[vgprValuC+39], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+15]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+15]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+110], s[sgprAlpha+1], -v[vgprValuC+47], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+110+1], s[sgprAlpha+0], v[vgprValuC+47], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+16]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+16]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+112], s[sgprAlpha+1], -v[vgprValuC+48], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+112+1], s[sgprAlpha+0], v[vgprValuC+48], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+24]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+24]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+114], s[sgprAlpha+1], -v[vgprValuC+56], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+114+1], s[sgprAlpha+0], v[vgprValuC+56], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+17]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+17]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+116], s[sgprAlpha+1], -v[vgprValuC+49], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+116+1], s[sgprAlpha+0], v[vgprValuC+49], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+25]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+25]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+118], s[sgprAlpha+1], -v[vgprValuC+57], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+118+1], s[sgprAlpha+0], v[vgprValuC+57], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+18]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+18]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+120], s[sgprAlpha+1], -v[vgprValuC+50], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+120+1], s[sgprAlpha+0], v[vgprValuC+50], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+26]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+26]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+122], s[sgprAlpha+1], -v[vgprValuC+58], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+122+1], s[sgprAlpha+0], v[vgprValuC+58], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+19]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+19]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+124], s[sgprAlpha+1], -v[vgprValuC+51], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+124+1], s[sgprAlpha+0], v[vgprValuC+51], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+27]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+27]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+126], s[sgprAlpha+1], -v[vgprValuC+59], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+126+1], s[sgprAlpha+0], v[vgprValuC+59], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+20]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+20]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+128], s[sgprAlpha+1], -v[vgprValuC+52], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+128+1], s[sgprAlpha+0], v[vgprValuC+52], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+28]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+28]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+130], s[sgprAlpha+1], -v[vgprValuC+60], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+130+1], s[sgprAlpha+0], v[vgprValuC+60], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+21]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+21]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+132], s[sgprAlpha+1], -v[vgprValuC+53], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+132+1], s[sgprAlpha+0], v[vgprValuC+53], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+29]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+29]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+134], s[sgprAlpha+1], -v[vgprValuC+61], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+134+1], s[sgprAlpha+0], v[vgprValuC+61], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+22]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+22]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+136], s[sgprAlpha+1], -v[vgprValuC+54], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+136+1], s[sgprAlpha+0], v[vgprValuC+54], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+30]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+30]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+138], s[sgprAlpha+1], -v[vgprValuC+62], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+138+1], s[sgprAlpha+0], v[vgprValuC+62], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+23]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+23]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+140], s[sgprAlpha+1], -v[vgprValuC+55], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+140+1], s[sgprAlpha+0], v[vgprValuC+55], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+31]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+31]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+142], s[sgprAlpha+1], -v[vgprValuC+63], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+142+1], s[sgprAlpha+0], v[vgprValuC+63], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_alu depctr_va_vdst(14)
buffer_store_b128 v[72:75], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[76:79], v100, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[80:83], v101, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[84:87], v105, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[88:91], v106, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[92:95], v107, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[96:99], v144, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[108:111], v145, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[112:115], v146, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[116:119], v147, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[120:123], v148, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[124:127], v149, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[128:131], v150, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b128 v[132:135], v151, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(8)
buffer_store_b128 v[136:139], v152, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[140:143], v153, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_FD0_VW2_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v70, BufferOOB
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v71, v67, v64, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v70, v71, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v105, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v105, v70, v105, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v140, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v140, v70, v140, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 3                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v141, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v141, v70, v141, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v142, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v70, v142, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v143, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v143, v70, v143, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v144, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v144, v70, v144, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 7                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v145, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v145, v70, v145, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v146, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v146, v70, v146, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v147, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v147, v70, v147, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v148, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v70, v148, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 11                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v149, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v149, v70, v149, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v150, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v150, v70, v150, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v151, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v151, v70, v151, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v152, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v152, v70, v152, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 15                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v153, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
v_cndmask_b32 v153, v70, v153, s14                 // LDD clip if OOB. offset
v_add_co_u32 v65, vcc_lo, v65, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_add_nc_u32 v66, v66, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v67, v67, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v154, v67, v64, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v154, v70, v154, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v155, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v70, v155, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v156, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v156, v70, v156, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 3                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v157, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v157, v70, v157, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v158, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v158, v70, v158, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v159, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v159, v70, v159, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v160, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v160, v70, v160, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 7                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v161, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v70, v161, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v162, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v162, v70, v162, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v163, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v70, v163, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v164, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v164, v70, v164, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 11                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v165, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v165, v70, v165, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v166, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v166, v70, v166, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v167, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v70, v167, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v168, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v168, v70, v168, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 15                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v169, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v169, v70, v169, s14                 // LDD clip if OOB. offset
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+0]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+0]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+72], s[sgprAlpha+1], -v[vgprValuC+32], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+72+1], s[sgprAlpha+0], v[vgprValuC+32], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+8]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+8]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+74], s[sgprAlpha+1], -v[vgprValuC+40], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+74+1], s[sgprAlpha+0], v[vgprValuC+40], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+1]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+1]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+33], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+33], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+9]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+9]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+41], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+41], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+2]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+2]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+34], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+34], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+10]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+10]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+42], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+42], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+3]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+3]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+84], s[sgprAlpha+1], -v[vgprValuC+35], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+84+1], s[sgprAlpha+0], v[vgprValuC+35], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+11]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+11]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+86], s[sgprAlpha+1], -v[vgprValuC+43], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+86+1], s[sgprAlpha+0], v[vgprValuC+43], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+4]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+4]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+88], s[sgprAlpha+1], -v[vgprValuC+36], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+88+1], s[sgprAlpha+0], v[vgprValuC+36], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+12]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+12]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+90], s[sgprAlpha+1], -v[vgprValuC+44], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+90+1], s[sgprAlpha+0], v[vgprValuC+44], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+5]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+5]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+92], s[sgprAlpha+1], -v[vgprValuC+37], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+92+1], s[sgprAlpha+0], v[vgprValuC+37], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+13]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+13]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+94], s[sgprAlpha+1], -v[vgprValuC+45], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+94+1], s[sgprAlpha+0], v[vgprValuC+45], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+6]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+6]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+96], s[sgprAlpha+1], -v[vgprValuC+38], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+96+1], s[sgprAlpha+0], v[vgprValuC+38], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+14]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+14]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+98], s[sgprAlpha+1], -v[vgprValuC+46], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+98+1], s[sgprAlpha+0], v[vgprValuC+46], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+7]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+7]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+100], s[sgprAlpha+1], -v[vgprValuC+39], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+100+1], s[sgprAlpha+0], v[vgprValuC+39], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+15]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+15]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+106], s[sgprAlpha+1], -v[vgprValuC+47], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+106+1], s[sgprAlpha+0], v[vgprValuC+47], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+16]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+16]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+108], s[sgprAlpha+1], -v[vgprValuC+48], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+108+1], s[sgprAlpha+0], v[vgprValuC+48], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+24]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+24]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+110], s[sgprAlpha+1], -v[vgprValuC+56], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+110+1], s[sgprAlpha+0], v[vgprValuC+56], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+17]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+17]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+112], s[sgprAlpha+1], -v[vgprValuC+49], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+112+1], s[sgprAlpha+0], v[vgprValuC+49], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+25]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+25]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+114], s[sgprAlpha+1], -v[vgprValuC+57], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+114+1], s[sgprAlpha+0], v[vgprValuC+57], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+18]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+18]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+116], s[sgprAlpha+1], -v[vgprValuC+50], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+116+1], s[sgprAlpha+0], v[vgprValuC+50], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+26]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+26]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+118], s[sgprAlpha+1], -v[vgprValuC+58], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+118+1], s[sgprAlpha+0], v[vgprValuC+58], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+19]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+19]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+120], s[sgprAlpha+1], -v[vgprValuC+51], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+120+1], s[sgprAlpha+0], v[vgprValuC+51], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+27]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+27]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+122], s[sgprAlpha+1], -v[vgprValuC+59], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+122+1], s[sgprAlpha+0], v[vgprValuC+59], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+20]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+20]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+124], s[sgprAlpha+1], -v[vgprValuC+52], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+124+1], s[sgprAlpha+0], v[vgprValuC+52], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+28]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+28]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+126], s[sgprAlpha+1], -v[vgprValuC+60], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+126+1], s[sgprAlpha+0], v[vgprValuC+60], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+21]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+21]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+128], s[sgprAlpha+1], -v[vgprValuC+53], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+128+1], s[sgprAlpha+0], v[vgprValuC+53], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+29]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+29]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+130], s[sgprAlpha+1], -v[vgprValuC+61], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+130+1], s[sgprAlpha+0], v[vgprValuC+61], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+22]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+22]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+132], s[sgprAlpha+1], -v[vgprValuC+54], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+132+1], s[sgprAlpha+0], v[vgprValuC+54], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+30]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+30]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+134], s[sgprAlpha+1], -v[vgprValuC+62], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+134+1], s[sgprAlpha+0], v[vgprValuC+62], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+23]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+23]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+136], s[sgprAlpha+1], -v[vgprValuC+55], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+136+1], s[sgprAlpha+0], v[vgprValuC+55], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+31]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+31]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+138], s[sgprAlpha+1], -v[vgprValuC+63], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+138+1], s[sgprAlpha+0], v[vgprValuC+63], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_alu depctr_va_vdst(14)
buffer_store_b64 v[72:73], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[74:75], v105, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[76:77], v140, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[78:79], v141, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[80:81], v142, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[82:83], v143, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[84:85], v144, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[86:87], v145, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[88:89], v146, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[90:91], v147, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[92:93], v148, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[94:95], v149, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[96:97], v150, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[98:99], v151, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[100:101], v152, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[106:107], v153, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[108:109], v154, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[110:111], v155, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[112:113], v156, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[114:115], v157, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[116:117], v158, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[118:119], v159, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[120:121], v160, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[122:123], v161, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[124:125], v162, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[126:127], v163, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[128:129], v164, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
buffer_store_b64 v[130:131], v165, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(12)
buffer_store_b64 v[132:133], v166, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(8)
buffer_store_b64 v[134:135], v167, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(4)
buffer_store_b64 v[136:137], v168, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b64 v[138:139], v169, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:
s_and_b32 s32, 63, s[sgprSizeI]                    // s32 = s[sgprSizeI] % 64
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rem
s_mov_b32 s5, 0
s_cmp_gt_u32 s32, s5                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW2_GSU1_Else       // jump if edges required
s_and_b32 s32, 63, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 64
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rem
s_mov_b32 s5, 0
s_cmp_gt_u32 s32, s5                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW2_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW2_GSU1_NonEdge:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_add_lshl_u32 v72, v66, v64, 3                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64 (multiple bpe)
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[140:143], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[144:147], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:16 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[148:151], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:32 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[152:155], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:48 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[156:159], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:64 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[160:163], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:80 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[164:167], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:96 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[168:171], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:112 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_lshl_b32 s12, s[sgprStrideC1J], 3                // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s12        // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_b128 v[172:175], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[176:179], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:16 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[180:183], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:32 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[184:187], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:48 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[188:191], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:64 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[192:195], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:80 scope:SCOPE_CU th:TH_LOAD_RT // load C
v_add_lshl_u32 v71, v67, v64, 3                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=64, coord0Vgpr=64 (multiple bpe)
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+0]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+0]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+32], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+32], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+8]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+8]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+40], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+40], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+1]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+1]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+33], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+33], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+9]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+9]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+41], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+41], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+2]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+2]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+84], s[sgprAlpha+1], -v[vgprValuC+34], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+84+1], s[sgprAlpha+0], v[vgprValuC+34], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+10]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+10]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+86], s[sgprAlpha+1], -v[vgprValuC+42], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+86+1], s[sgprAlpha+0], v[vgprValuC+42], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+3]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+3]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+88], s[sgprAlpha+1], -v[vgprValuC+35], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+88+1], s[sgprAlpha+0], v[vgprValuC+35], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+11]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+11]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+90], s[sgprAlpha+1], -v[vgprValuC+43], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+90+1], s[sgprAlpha+0], v[vgprValuC+43], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+4]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+4]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+92], s[sgprAlpha+1], -v[vgprValuC+36], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+92+1], s[sgprAlpha+0], v[vgprValuC+36], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+12]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+12]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+94], s[sgprAlpha+1], -v[vgprValuC+44], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+94+1], s[sgprAlpha+0], v[vgprValuC+44], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+5]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+5]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+96], s[sgprAlpha+1], -v[vgprValuC+37], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+96+1], s[sgprAlpha+0], v[vgprValuC+37], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+13]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+13]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+98], s[sgprAlpha+1], -v[vgprValuC+45], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+98+1], s[sgprAlpha+0], v[vgprValuC+45], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+6]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+6]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+108], s[sgprAlpha+1], -v[vgprValuC+38], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+108+1], s[sgprAlpha+0], v[vgprValuC+38], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+14]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+14]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+110], s[sgprAlpha+1], -v[vgprValuC+46], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+110+1], s[sgprAlpha+0], v[vgprValuC+46], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+7]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+7]     // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+112], s[sgprAlpha+1], -v[vgprValuC+39], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+112+1], s[sgprAlpha+0], v[vgprValuC+39], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+15]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+15]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+114], s[sgprAlpha+1], -v[vgprValuC+47], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+114+1], s[sgprAlpha+0], v[vgprValuC+47], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+16]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+16]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+116], s[sgprAlpha+1], -v[vgprValuC+48], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+116+1], s[sgprAlpha+0], v[vgprValuC+48], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+24]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+24]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+118], s[sgprAlpha+1], -v[vgprValuC+56], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+118+1], s[sgprAlpha+0], v[vgprValuC+56], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+17]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+17]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+120], s[sgprAlpha+1], -v[vgprValuC+49], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+120+1], s[sgprAlpha+0], v[vgprValuC+49], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+25]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+25]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+122], s[sgprAlpha+1], -v[vgprValuC+57], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+122+1], s[sgprAlpha+0], v[vgprValuC+57], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+18]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+18]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+124], s[sgprAlpha+1], -v[vgprValuC+50], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+124+1], s[sgprAlpha+0], v[vgprValuC+50], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+26]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+26]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+126], s[sgprAlpha+1], -v[vgprValuC+58], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+126+1], s[sgprAlpha+0], v[vgprValuC+58], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+19]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+19]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+128], s[sgprAlpha+1], -v[vgprValuC+51], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+128+1], s[sgprAlpha+0], v[vgprValuC+51], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+27]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+27]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+130], s[sgprAlpha+1], -v[vgprValuC+59], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+130+1], s[sgprAlpha+0], v[vgprValuC+59], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+20]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+20]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+132], s[sgprAlpha+1], -v[vgprValuC+52], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+132+1], s[sgprAlpha+0], v[vgprValuC+52], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+28]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+28]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+134], s[sgprAlpha+1], -v[vgprValuC+60], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+134+1], s[sgprAlpha+0], v[vgprValuC+60], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+21]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+21]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+136], s[sgprAlpha+1], -v[vgprValuC+53], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+136+1], s[sgprAlpha+0], v[vgprValuC+53], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+29]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+29]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+138], s[sgprAlpha+1], -v[vgprValuC+61], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+138+1], s[sgprAlpha+0], v[vgprValuC+61], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_loadcnt 13                                  // vlcnt(13) = 14 - 1 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+76], v140, s[sgprBeta]      // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+76], v141, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+77], v141, s[sgprBeta]      // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+77], v140, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+78], v142, s[sgprBeta]      // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+78], v143, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+79], v143, s[sgprBeta]      // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+79], v142, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
s_wait_loadcnt 12                                  // vlcnt(12) = 14 - 2 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+80], v144, s[sgprBeta]      // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+80], v145, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+81], v145, s[sgprBeta]      // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+81], v144, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+82], v146, s[sgprBeta]      // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+82], v147, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+83], v147, s[sgprBeta]      // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+83], v146, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
s_wait_loadcnt 11                                  // vlcnt(11) = 14 - 3 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+84], v148, s[sgprBeta]      // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+84], v149, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[76:79], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[80:83], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+85], v149, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+85], v148, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+86], v150, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+86], v151, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+87], v151, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+87], v150, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
s_wait_loadcnt 10                                  // vlcnt(10) = 14 - 4 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+88], v152, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+88], v153, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+89], v153, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+89], v152, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+90], v154, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+90], v155, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+91], v155, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+91], v154, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
s_wait_loadcnt 9                                   // vlcnt(9) = 14 - 5 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+92], v156, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+92], v157, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[84:87], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:32 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[88:91], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:48 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+93], v157, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+93], v156, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+94], v158, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+94], v159, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+95], v159, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+95], v158, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
s_wait_loadcnt 8                                   // vlcnt(8) = 14 - 6 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+96], v160, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+96], v161, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+97], v161, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+97], v160, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+98], v162, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+98], v163, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+99], v163, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+99], v162, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
s_wait_loadcnt 7                                   // vlcnt(7) = 14 - 7 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+108], v164, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+108], v165, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[92:95], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:64 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[96:99], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:80 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+109], v165, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+109], v164, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+110], v166, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+110], v167, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+111], v167, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+111], v166, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_loadcnt 6                                   // vlcnt(6) = 14 - 8 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+112], v168, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+112], v169, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+113], v169, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+113], v168, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+114], v170, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+114], v171, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+115], v171, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+115], v170, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_loadcnt 5                                   // vlcnt(5) = 14 - 9 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+116], v172, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+116], v173, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[108:111], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:96 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[112:115], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:112 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+117], v173, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+117], v172, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+118], v174, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+118], v175, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+119], v175, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+119], v174, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_lshl_b32 s12, s[sgprStrideD1J], 3                // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
s_wait_loadcnt 4                                   // vlcnt(4) = 14 - 10 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+120], v176, s[sgprBeta]     // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+120], v177, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+121], v177, s[sgprBeta]     // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+121], v176, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+122], v178, s[sgprBeta]     // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+122], v179, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+123], v179, s[sgprBeta]     // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+123], v178, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_loadcnt 3                                   // vlcnt(3) = 14 - 11 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+124], v180, s[sgprBeta]     // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+124], v181, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[116:119], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[120:123], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:16 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+125], v181, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+125], v180, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+126], v182, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+126], v183, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+127], v183, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+127], v182, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_loadcnt 2                                   // vlcnt(2) = 14 - 12 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+128], v184, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+128], v185, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+129], v185, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+129], v184, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+130], v186, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+130], v187, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+131], v187, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+131], v186, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_loadcnt 1                                   // vlcnt(1) = 14 - 13 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+132], v188, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+132], v189, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[124:127], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:32 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[128:131], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:48 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+133], v189, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+133], v188, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+134], v190, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+134], v191, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+135], v191, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+135], v190, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_loadcnt 0                                   // vlcnt(0) = 14 - 14 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+136], v192, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+136], v193, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+137], v193, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+137], v192, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+138], v194, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+138], v195, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+139], v195, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+139], v194, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_alu depctr_va_vdst(8)
buffer_store_b128 v[132:135], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:64 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[136:139], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:80 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
buffer_load_b128 v[84:87], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:96 scope:SCOPE_CU th:TH_LOAD_RT // load C
buffer_load_b128 v[88:91], v72, s[sgprSrdC:sgprSrdC+3], null offen offset:112 scope:SCOPE_CU th:TH_LOAD_RT // load C
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+22]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+22]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+54], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+54], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+30]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+30]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+62], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+62], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+23]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+23]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+55], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+55], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+31]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+31]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+63], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+63], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_loadcnt 1                                   // vlcnt(1) = 2 - 1 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+76], v84, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+76], v85, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+77], v85, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+77], v84, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+78], v86, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+78], v87, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+79], v87, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+79], v86, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
s_wait_loadcnt 0                                   // vlcnt(0) = 2 - 2 (beta) (interleaved)
v_fmac_f32 v[vgprValuC+80], v88, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+80], v89, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+81], v89, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+81], v88, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+82], v90, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+82], v91, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+83], v91, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+83], v90, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
s_wait_alu depctr_va_vdst(8)
buffer_store_b128 v[76:79], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:96 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[80:83], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:112 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW2_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW2_GSU1_Then:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v70, BufferOOB
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v71, v66, v64, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v70, v71, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[132:135], v71, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v71, v67, v64, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v71, v70, v71, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v100, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v100, v70, v100, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[136:139], v100, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v100, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v100, v70, v100, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v101, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v101, v70, v101, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[140:143], v101, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v101, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v101, v70, v101, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v105, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v70, v105, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[144:147], v105, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v105, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v105, v70, v105, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v106, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v106, v70, v106, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[148:151], v106, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v106, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v106, v70, v106, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v107, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v107, v70, v107, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[152:155], v107, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v107, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v107, v70, v107, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v160, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v70, v160, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[156:159], v160, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v160, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v160, v70, v160, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v161, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v70, v161, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[164:167], v161, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v161, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
v_cndmask_b32 v161, v70, v161, s14                 // LDD clip if OOB. offset
v_add_co_u32 v65, vcc_lo, v65, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_add_nc_u32 v66, v66, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v67, v67, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v162, v66, v64, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v162, v70, v162, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[168:171], v162, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v162, v67, v64, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v162, v70, v162, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v163, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v163, v70, v163, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[172:175], v163, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v163, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v163, v70, v163, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v180, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v180, v70, v180, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[176:179], v180, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v180, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v180, v70, v180, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v181, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v181, v70, v181, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[184:187], v181, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v181, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v181, v70, v181, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v182, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v182, v70, v182, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[188:191], v182, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v182, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_cndmask_b32 v182, v70, v182, s14                 // LDD clip if OOB. offset
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+0]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+0]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+72], s[sgprAlpha+1], -v[vgprValuC+32], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+72+1], s[sgprAlpha+0], v[vgprValuC+32], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+8]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+8]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+74], s[sgprAlpha+1], -v[vgprValuC+40], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+74+1], s[sgprAlpha+0], v[vgprValuC+40], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+1]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+1]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+33], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+33], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+9]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+9]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+41], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+41], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+2]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+2]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+34], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+34], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+10]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+10]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+42], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+42], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+3]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+3]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+84], s[sgprAlpha+1], -v[vgprValuC+35], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+84+1], s[sgprAlpha+0], v[vgprValuC+35], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+11]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+11]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+86], s[sgprAlpha+1], -v[vgprValuC+43], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+86+1], s[sgprAlpha+0], v[vgprValuC+43], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+4]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+4]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+88], s[sgprAlpha+1], -v[vgprValuC+36], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+88+1], s[sgprAlpha+0], v[vgprValuC+36], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+12]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+12]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+90], s[sgprAlpha+1], -v[vgprValuC+44], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+90+1], s[sgprAlpha+0], v[vgprValuC+44], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+5]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+5]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+92], s[sgprAlpha+1], -v[vgprValuC+37], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+92+1], s[sgprAlpha+0], v[vgprValuC+37], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+13]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+13]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+94], s[sgprAlpha+1], -v[vgprValuC+45], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+94+1], s[sgprAlpha+0], v[vgprValuC+45], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+6]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+6]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+96], s[sgprAlpha+1], -v[vgprValuC+38], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+96+1], s[sgprAlpha+0], v[vgprValuC+38], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+14]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+14]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+98], s[sgprAlpha+1], -v[vgprValuC+46], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+98+1], s[sgprAlpha+0], v[vgprValuC+46], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+7]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+7]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+108], s[sgprAlpha+1], -v[vgprValuC+39], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+108+1], s[sgprAlpha+0], v[vgprValuC+39], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+15]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+15]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+110], s[sgprAlpha+1], -v[vgprValuC+47], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+110+1], s[sgprAlpha+0], v[vgprValuC+47], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+16]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+16]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+112], s[sgprAlpha+1], -v[vgprValuC+48], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+112+1], s[sgprAlpha+0], v[vgprValuC+48], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+24]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+24]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+114], s[sgprAlpha+1], -v[vgprValuC+56], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+114+1], s[sgprAlpha+0], v[vgprValuC+56], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+17]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+17]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+116], s[sgprAlpha+1], -v[vgprValuC+49], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+116+1], s[sgprAlpha+0], v[vgprValuC+49], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+25]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+25]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+118], s[sgprAlpha+1], -v[vgprValuC+57], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+118+1], s[sgprAlpha+0], v[vgprValuC+57], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+18]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+18]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+120], s[sgprAlpha+1], -v[vgprValuC+50], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+120+1], s[sgprAlpha+0], v[vgprValuC+50], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+26]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+26]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+122], s[sgprAlpha+1], -v[vgprValuC+58], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+122+1], s[sgprAlpha+0], v[vgprValuC+58], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+19]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+19]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+124], s[sgprAlpha+1], -v[vgprValuC+51], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+124+1], s[sgprAlpha+0], v[vgprValuC+51], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+27]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+27]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+126], s[sgprAlpha+1], -v[vgprValuC+59], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+126+1], s[sgprAlpha+0], v[vgprValuC+59], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+20]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+20]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+128], s[sgprAlpha+1], -v[vgprValuC+52], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+128+1], s[sgprAlpha+0], v[vgprValuC+52], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+28]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+28]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+130], s[sgprAlpha+1], -v[vgprValuC+60], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fma_f32 v[vgprValuC+130+1], s[sgprAlpha+0], v[vgprValuC+60], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_loadcnt 0                                   // wait for Beta
v_fmac_f32 v[vgprValuC+72], v132, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+72], v133, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+73], v133, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+73], v132, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+74], v134, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+74], v135, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+75], v135, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+75], v134, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+76], v136, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+76], v137, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+77], v137, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+77], v136, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+78], v138, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+78], v139, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+79], v139, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+79], v138, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+80], v140, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+80], v141, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[72:75], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[76:79], v100, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+81], v141, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+81], v140, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+82], v142, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+82], v143, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+83], v143, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+83], v142, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+84], v144, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+84], v145, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+85], v145, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+85], v144, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+86], v146, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+86], v147, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+87], v147, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+87], v146, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+88], v148, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+88], v149, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[80:83], v101, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[84:87], v105, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+89], v149, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+89], v148, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+90], v150, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+90], v151, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+91], v151, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+91], v150, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+92], v152, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+92], v153, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+93], v153, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+93], v152, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+94], v154, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+94], v155, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+95], v155, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+95], v154, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+96], v156, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+96], v157, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[88:91], v106, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[92:95], v107, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+97], v157, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+97], v156, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+98], v158, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+98], v159, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+99], v159, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+99], v158, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+108], v164, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+108], v165, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+109], v165, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+109], v164, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+110], v166, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+110], v167, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+111], v167, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+111], v166, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+112], v168, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+112], v169, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[96:99], v160, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[108:111], v161, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+113], v169, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+113], v168, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+114], v170, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+114], v171, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+115], v171, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+115], v170, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+116], v172, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+116], v173, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+117], v173, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+117], v172, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+118], v174, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+118], v175, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+119], v175, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+119], v174, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+120], v176, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+120], v177, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[112:115], v162, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[116:119], v163, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+121], v177, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+121], v176, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+122], v178, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+122], v179, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+123], v179, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+123], v178, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+124], v184, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+124], v185, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+125], v185, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+125], v184, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+126], v186, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+126], v187, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+127], v187, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+127], v186, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+128], v188, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+128], v189, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[120:123], v180, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[124:127], v181, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+129], v189, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+129], v188, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+130], v190, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+130], v191, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+131], v191, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+131], v190, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[128:131], v182, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
v_mov_b32 v70, BufferOOB
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v71, v66, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v71, v70, v71, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(6)
buffer_load_b128 v[84:87], v71, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v71, v67, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v70, v71, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v92, v66, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v92, v70, v92, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[88:91], v92, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v92, v67, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v70, v92, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v93, v66, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v93, v70, v93, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b128 v[96:99], v93, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v93, v67, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v70, v93, s14                   // LDD clip if OOB. offset
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+21]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+21]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+72], s[sgprAlpha+1], -v[vgprValuC+53], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+72+1], s[sgprAlpha+0], v[vgprValuC+53], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+29]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+29]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+74], s[sgprAlpha+1], -v[vgprValuC+61], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+74+1], s[sgprAlpha+0], v[vgprValuC+61], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+22]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+22]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+54], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+54], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+30]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+30]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+62], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+62], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+23]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+23]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+55], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+55], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+31]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+31]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+63], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+63], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_loadcnt 0                                   // wait for Beta
v_fmac_f32 v[vgprValuC+72], v84, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+72], v85, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+73], v85, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+73], v84, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+74], v86, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+74], v87, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+75], v87, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+75], v86, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+76], v88, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+76], v89, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+77], v89, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+77], v88, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+78], v90, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+78], v91, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+79], v91, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+79], v90, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+80], v96, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+80], v97, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b128 v[72:75], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b128 v[76:79], v92, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+81], v97, s[sgprBeta]       // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+81], v96, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+82], v98, s[sgprBeta]       // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+82], v99, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+83], v99, s[sgprBeta]       // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+83], v98, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
s_wait_alu depctr_va_vdst(0)
buffer_store_b128 v[80:83], v93, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_FD0_VW2_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_mov_b32 v70, BufferOOB
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v71, v66, v64, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v70, v71, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[124:125], v71, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v71, v67, v64, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v71, v70, v71, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v105, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v70, v105, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[126:127], v105, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v105, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v105, v70, v105, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v130, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v130, v70, v130, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[128:129], v130, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v130, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v130, v70, v130, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 3                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v131, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v131, v70, v131, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[132:133], v131, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v131, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v131, v70, v131, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v136, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v136, v70, v136, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[134:135], v136, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v136, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v136, v70, v136, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v137, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v137, v70, v137, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[138:139], v137, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v137, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v137, v70, v137, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v142, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v142, v70, v142, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[140:141], v142, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v142, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v142, v70, v142, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 7                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v143, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v143, v70, v143, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[144:145], v143, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v143, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v143, v70, v143, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v148, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v148, v70, v148, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[146:147], v148, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v148, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v148, v70, v148, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v149, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v149, v70, v149, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[150:151], v149, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v149, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v149, v70, v149, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v154, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v154, v70, v154, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[152:153], v154, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v154, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v154, v70, v154, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 11                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v155, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v155, v70, v155, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[156:157], v155, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v155, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v155, v70, v155, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v160, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v160, v70, v160, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[158:159], v160, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v160, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v160, v70, v160, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v161, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v161, v70, v161, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[162:163], v161, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v161, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v161, v70, v161, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v166, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v166, v70, v166, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[164:165], v166, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v166, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v166, v70, v166, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 15                  // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v167, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v167, v70, v167, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[168:169], v167, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v167, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
v_cndmask_b32 v167, v70, v167, s14                 // LDD clip if OOB. offset
v_add_co_u32 v65, vcc_lo, v65, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1
v_add_nc_u32 v66, v66, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v67, v67, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s12, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v172, v66, v64, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v172, v70, v172, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[170:171], v172, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v172, v67, v64, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v172, v70, v172, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v173, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v173, v70, v173, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[174:175], v173, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v173, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v173, v70, v173, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 2                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v178, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v178, v70, v178, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[176:177], v178, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v178, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v178, v70, v178, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 3                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v179, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v179, v70, v179, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[180:181], v179, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v179, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v179, v70, v179, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 4                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v184, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v184, v70, v184, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[182:183], v184, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v184, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v184, v70, v184, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 5                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v185, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v185, v70, v185, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[186:187], v185, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v185, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v185, v70, v185, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 6                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v190, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v190, v70, v190, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[188:189], v190, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v190, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cndmask_b32 v190, v70, v190, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 7                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v191, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v191, v70, v191, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[192:193], v191, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v191, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_cndmask_b32 v191, v70, v191, s14                 // LDD clip if OOB. offset
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+0]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+0]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+72], s[sgprAlpha+1], -v[vgprValuC+32], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+72+1], s[sgprAlpha+0], v[vgprValuC+32], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+8]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+8]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+74], s[sgprAlpha+1], -v[vgprValuC+40], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+74+1], s[sgprAlpha+0], v[vgprValuC+40], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+1]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+1]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+33], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+33], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+9]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+9]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+41], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+41], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+2]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+2]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+34], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+34], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+10]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+10]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+42], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+42], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+3]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+3]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+84], s[sgprAlpha+1], -v[vgprValuC+35], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+84+1], s[sgprAlpha+0], v[vgprValuC+35], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+11]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+11]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+86], s[sgprAlpha+1], -v[vgprValuC+43], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+86+1], s[sgprAlpha+0], v[vgprValuC+43], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+4]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+4]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+88], s[sgprAlpha+1], -v[vgprValuC+36], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+88+1], s[sgprAlpha+0], v[vgprValuC+36], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+12]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+12]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+90], s[sgprAlpha+1], -v[vgprValuC+44], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+90+1], s[sgprAlpha+0], v[vgprValuC+44], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+5]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+5]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+92], s[sgprAlpha+1], -v[vgprValuC+37], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+92+1], s[sgprAlpha+0], v[vgprValuC+37], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+13]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+13]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+94], s[sgprAlpha+1], -v[vgprValuC+45], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+94+1], s[sgprAlpha+0], v[vgprValuC+45], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+6]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+6]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+96], s[sgprAlpha+1], -v[vgprValuC+38], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+96+1], s[sgprAlpha+0], v[vgprValuC+38], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+14]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+14]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+98], s[sgprAlpha+1], -v[vgprValuC+46], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+98+1], s[sgprAlpha+0], v[vgprValuC+46], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+7]     // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+7]     // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+100], s[sgprAlpha+1], -v[vgprValuC+39], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+100+1], s[sgprAlpha+0], v[vgprValuC+39], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+15]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+15]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+106], s[sgprAlpha+1], -v[vgprValuC+47], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+106+1], s[sgprAlpha+0], v[vgprValuC+47], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+16]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+16]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+108], s[sgprAlpha+1], -v[vgprValuC+48], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+108+1], s[sgprAlpha+0], v[vgprValuC+48], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+24]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+24]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+110], s[sgprAlpha+1], -v[vgprValuC+56], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+110+1], s[sgprAlpha+0], v[vgprValuC+56], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+17]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+17]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+112], s[sgprAlpha+1], -v[vgprValuC+49], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+112+1], s[sgprAlpha+0], v[vgprValuC+49], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+25]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+25]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+114], s[sgprAlpha+1], -v[vgprValuC+57], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+114+1], s[sgprAlpha+0], v[vgprValuC+57], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+18]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+18]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+116], s[sgprAlpha+1], -v[vgprValuC+50], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+116+1], s[sgprAlpha+0], v[vgprValuC+50], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+26]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+26]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+118], s[sgprAlpha+1], -v[vgprValuC+58], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+118+1], s[sgprAlpha+0], v[vgprValuC+58], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+19]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+19]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+120], s[sgprAlpha+1], -v[vgprValuC+51], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+120+1], s[sgprAlpha+0], v[vgprValuC+51], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+27]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+27]    // tmp2 = a.imag * b.real
v_fma_f32 v[vgprValuC+122], s[sgprAlpha+1], -v[vgprValuC+59], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
v_fma_f32 v[vgprValuC+122+1], s[sgprAlpha+0], v[vgprValuC+59], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_loadcnt 0                                   // wait for Beta
v_fmac_f32 v[vgprValuC+72], v124, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+72], v125, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+73], v125, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+73], v124, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+74], v126, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+74], v127, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+75], v127, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+75], v126, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+76], v128, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+76], v129, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+77], v129, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+77], v128, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+78], v132, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+78], v133, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[72:73], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[74:75], v105, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[76:77], v130, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+79], v133, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+79], v132, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+80], v134, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+80], v135, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+81], v135, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+81], v134, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+82], v138, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+82], v139, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+83], v139, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+83], v138, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+84], v140, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+84], v141, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[78:79], v131, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[80:81], v136, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[82:83], v137, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+85], v141, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_wait_xcnt 0
s_prefetch_inst_pc_rel 0, null, 31
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+85], v140, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+86], v144, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+86], v145, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+87], v145, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+87], v144, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+88], v146, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+88], v147, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+89], v147, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+89], v146, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+90], v150, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+90], v151, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[84:85], v142, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[86:87], v143, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[88:89], v148, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+91], v151, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+91], v150, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+92], v152, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+92], v153, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+93], v153, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+93], v152, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+94], v156, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+94], v157, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+95], v157, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+95], v156, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+96], v158, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+96], v159, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[90:91], v149, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[92:93], v154, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[94:95], v155, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+97], v159, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+97], v158, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+98], v162, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+98], v163, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+99], v163, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+99], v162, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+100], v164, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+100], v165, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+101], v165, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+101], v164, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+106], v168, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+106], v169, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[96:97], v160, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[98:99], v161, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[100:101], v166, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+107], v169, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+107], v168, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+108], v170, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+108], v171, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+109], v171, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+109], v170, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+110], v174, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+110], v175, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+111], v175, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+111], v174, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+112], v176, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+112], v177, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[106:107], v167, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[108:109], v172, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[110:111], v173, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+113], v177, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+113], v176, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+114], v180, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+114], v181, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+115], v181, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+115], v180, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+116], v182, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+116], v183, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+117], v183, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+117], v182, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+118], v186, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+118], v187, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[112:113], v178, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[114:115], v179, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[116:117], v184, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+119], v187, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+119], v186, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+120], v188, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+120], v189, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+121], v189, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+121], v188, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+122], v192, s[sgprBeta]     // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+122], v193, -s[sgprBeta+1]  // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+123], v193, s[sgprBeta]     // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+123], v192, s[sgprBeta+1]   // finalSum Ci += old Cr * Bi
s_wait_alu depctr_va_vdst(8)
buffer_store_b64 v[118:119], v185, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(4)
buffer_store_b64 v[120:121], v190, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b64 v[122:123], v191, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
v_mov_b32 v70, BufferOOB
v_add_co_u32 v68, vcc_lo, v64, 8                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v71, v66, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v71, v70, v71, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[88:89], v71, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v71, v67, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v70, v71, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 9                   // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v92, v66, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v92, v70, v92, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[90:91], v92, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v92, v67, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v92, v70, v92, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 10                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v93, v66, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v93, v70, v93, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[94:95], v93, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v93, v67, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v93, v70, v93, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 11                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v98, v66, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v98, v70, v98, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[96:97], v98, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v98, v67, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v98, v70, v98, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 12                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v99, v66, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v99, v70, v99, s14                   // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[100:101], v99, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v99, v67, v68, 3                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v99, v70, v99, s14                   // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 13                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v105, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v105, v70, v105, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[106:107], v105, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v105, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v105, v70, v105, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 14                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v110, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v110, v70, v110, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[108:109], v110, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v110, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v110, v70, v110, s14                 // LDD clip if OOB. offset
v_add_co_u32 v68, vcc_lo, v64, 15                  // coord0.1: coord0 += d0*sg0*VW + vc0
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_cmp_lt_u32 s12, v68, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s14, v65, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s14, s12, s14                            // in0 && in1
v_add_lshl_u32 v111, v66, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
v_cndmask_b32 v111, v70, v111, s14                 // LDC clip if OOB. offset
s_wait_alu depctr_va_vdst(0)
buffer_load_b64 v[112:113], v111, s[sgprSrdC:sgprSrdC+3], null offen offset:0 scope:SCOPE_CU th:TH_LOAD_RT // load C
s_wait_alu depctr_vm_vsrc(0)
v_add_lshl_u32 v111, v67, v68, 3                   // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v111, v70, v111, s14                 // LDD clip if OOB. offset
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+20]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+20]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+72], s[sgprAlpha+1], -v[vgprValuC+52], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+72+1], s[sgprAlpha+0], v[vgprValuC+52], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+28]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+28]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+74], s[sgprAlpha+1], -v[vgprValuC+60], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+74+1], s[sgprAlpha+0], v[vgprValuC+60], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+21]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+21]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+76], s[sgprAlpha+1], -v[vgprValuC+53], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+76+1], s[sgprAlpha+0], v[vgprValuC+53], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+29]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+29]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+78], s[sgprAlpha+1], -v[vgprValuC+61], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+78+1], s[sgprAlpha+0], v[vgprValuC+61], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+22]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+22]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+80], s[sgprAlpha+1], -v[vgprValuC+54], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+80+1], s[sgprAlpha+0], v[vgprValuC+54], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+30]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+30]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+82], s[sgprAlpha+1], -v[vgprValuC+62], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+82+1], s[sgprAlpha+0], v[vgprValuC+62], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+23]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+23]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+84], s[sgprAlpha+1], -v[vgprValuC+55], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+84+1], s[sgprAlpha+0], v[vgprValuC+55], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
v_mul_f32 v102, s[sgprAlpha+0], v[vgprValuC+31]    // tmp1 = a.real * b.real
v_mul_f32 v103, s[sgprAlpha+1], v[vgprValuC+31]    // tmp2 = a.imag * b.real
s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
v_fma_f32 v[vgprValuC+86], s[sgprAlpha+1], -v[vgprValuC+63], v102 // c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
v_fma_f32 v[vgprValuC+86+1], s[sgprAlpha+0], v[vgprValuC+63], v103 // c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
s_wait_loadcnt 0                                   // wait for Beta
v_fmac_f32 v[vgprValuC+72], v88, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+72], v89, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+73], v89, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+73], v88, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+74], v90, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+74], v91, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+75], v91, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+75], v90, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+76], v94, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+76], v95, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+77], v95, s[sgprBeta]       // finalSum Ci += old Ci * Br
v_fmac_f32 v[vgprValuC+77], v94, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+78], v96, s[sgprBeta]       // finalSum Cr += old Cr * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+78], v97, -s[sgprBeta+1]    // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[72:73], v71, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[74:75], v92, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[76:77], v93, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+79], v97, s[sgprBeta]       // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+79], v96, s[sgprBeta+1]     // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+80], v100, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+80], v101, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+81], v101, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+81], v100, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+82], v106, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+82], v107, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+83], v107, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+83], v106, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+84], v108, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+84], v109, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
s_wait_alu depctr_va_vdst(10)
buffer_store_b64 v[78:79], v98, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(6)
buffer_store_b64 v[80:81], v99, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(2)
buffer_store_b64 v[82:83], v105, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
v_fmac_f32 v[vgprValuC+85], v109, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+85], v108, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
v_fmac_f32 v[vgprValuC+86], v112, s[sgprBeta]      // finalSum Cr += old Cr * Br
v_fmac_f32 v[vgprValuC+86], v113, -s[sgprBeta+1]   // finalSum Cr += old Ci * -Bi
v_fmac_f32 v[vgprValuC+87], v113, s[sgprBeta]      // finalSum Ci += old Ci * Br
s_delay_alu instid0(VALU_DEP_1)
v_fmac_f32 v[vgprValuC+87], v112, s[sgprBeta+1]    // finalSum Ci += old Cr * Bi
s_wait_alu depctr_va_vdst(4)
buffer_store_b64 v[84:85], v110, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_wait_alu depctr_va_vdst(0)
buffer_store_b64 v[86:87], v111, s[sgprSrdD:sgprSrdD+3], null offen offset:0 scope:SCOPE_CU th:TH_STORE_RT // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_End_2:
label_KernelEnd:
s_wait_xcnt 0
s_prefetch_inst_pc_rel 0, null, 31
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
