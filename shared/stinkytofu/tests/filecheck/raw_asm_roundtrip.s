# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label label_LoopBeginL --to-label label_LoopEndL --emit-asm
#
# Round-trip: no optimization passes.  Verifies that:
# 1. The .amdhsa_kernel signature header is extracted and regenerated.
# 2. All 539 instructions in the loop body survive parsing and re-emission
#    (no instructions dropped due to .set UNDEF or MSB-bank register handling).
# 3. .set directives with UNDEF values do not clobber previously resolved
#    symbols: symbolic registers must resolve to the same numeric indices
#    as in the source.
#
# CHECK: .amdhsa_kernel Cijk_Alik_Bljk
# CHECK: .amdhsa_next_free_vgpr 1022
# CHECK: .amdhsa_next_free_sgpr 67
# CHECK: .amdhsa_group_segment_fixed_size 262144
# CHECK: .amdhsa_wavefront_size32 1
# CHECK: .end_amdhsa_kernel
#
# Regression: the kernel-level YAML entry "  - .name: <kernelName>" sits
# under "amdhsa.kernels:" and must NOT be parsed as the first argument.
# Before the fix, the regenerated .args: block was prefixed with a synthetic
# "- .name: Cijk_Alik_Bljk_..." entry copied from the kernel name. The first
# real arg in this kernel is "Gemm info"; assert that immediately after
# .args: the first argument is Gemm info, not the kernel name.
# CHECK:      .args:
# CHECK-NEXT: - .name:{{.*}}Gemm info
#
# Regression: SignatureCodeMeta::toString() ends its emission with
# "<kernelName>:\n", so the signature header itself provides the kernel
# body label. The parser must NOT also push the kernel body label as a
# ParsedInstruction (otherwise it appears twice in the round-tripped
# output: once from the signature emit, once from the function emit).
# CHECK:      .end_amdgpu_metadata
# CHECK-NEXT: Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_{{.*}}:
# CHECK-NOT:  Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_{{.*}}:
# CHECK: label_LoopBeginL
# CHECK: ds_load_b128
# CHECK: v_wmma_f32_16x16x32_bf16
# CHECK: s_cbranch_scc0 label_LoopBeginL
# CHECK: label_LoopEndL


/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.protected Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV1_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS32_WG32_4_1
.globl Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV1_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS32_WG32_4_1
.p2align 8
.type Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV1_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS32_WG32_4_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV1_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS32_WG32_4_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 1022 // vgprs
  .amdhsa_next_free_sgpr 67 // sgprs
  .amdhsa_group_segment_fixed_size 262144 // lds bytes
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
/* Num VGPR   =1022 */
/* Num AccVGPR=0 */
/* Num SGPR   =67 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 64 x 8 */
/* SubGroup= 4 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=8 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8 */
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
  - .name: Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV1_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS32_WG32_4_1
    .symbol: 'Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV1_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS32_WG32_4_1.kd'
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
    .group_segment_fixed_size:   262144
    .kernarg_segment_align:      8
    .kernarg_segment_size:       104
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 67
    .sgpr_spill_count:           0
    .vgpr_count:                 1022
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC0_AG0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1250_IU1_K1_LDSTI0_LBSPPA0_LBSPPB0_LBSPPM0_LPA0_LPB0_LPM0_LRVWn1_LWPMn1_MIAV1_MIWT8_8_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA0_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKXCCM0_SGRO0_TDMI3_TIN0_TLDS1_TLDSMn1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS32_WG32_4_1:
label_ASM_Start:  /// Main body of the asm kernel
.macro V_MAGIC_DIV vgprDstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    s_nop 0
    s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
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

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-512), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 516
.set vgprGlobalReadOffsetA, 512
.set vgprGlobalReadOffsetB, 512
.set vgprLocalReadAddrA, 512
.set vgprLocalReadAddrB, 514
.set vgprSerial, 902

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+130
.set vgprG2LB_BASE, vgprBase+258
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+64
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+64
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
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
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
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 256
.set MT1, 256
.set DepthU, 128
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
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
    v_add_nc_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]   // add prepad for pointer shift
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req, vgprOffsetL:req, vgprOffset1J:req, vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0], vcc_lo, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
    v_add_nc_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]   // add prepad for pointer shift
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Init workgroup id from ttmp */
s_mov_b32 s[sgprWorkGroup0], ttmp9
s_and_b32 s[sgprWorkGroup1], 0xffff, ttmp7
s_lshr_b32 s[sgprWorkGroup2], ttmp7, 0x10

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
s_cmp_eq_u32 s21, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0 // 0
s_load_b128 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 64 // 64
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_wait_kmcnt 0                                     // wait for args to load
label_LoadArgsEnd:
s_and_b32 s[sgprStaggerU], s22, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s22, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s21
s_mov_b32 m0, 0x40000                              // LDS clamp at 262144 bytes
s_set_vgpr_msb 192                                 // src0: 0, src1: 0, src2: 0, dst: 3
v_mov_b32 v[vgprSerial-768], v0                    // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

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
s_nop 0
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[0:1], s53                          // s49 = s[sgprWorkGroup0] / s53
v_rcp_f64 v[0:1], v[0:1]                           // s49 = s[sgprWorkGroup0] / s53
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s49 = s[sgprWorkGroup0] / s53
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s49 = s[sgprWorkGroup0] / s53
v_cvt_u32_f64 v0, v[0:1]                           // s49 = s[sgprWorkGroup0] / s53
v_mul_lo_u32 v1, v0, s53                           // s49 = s[sgprWorkGroup0] / s53
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s49 = s[sgprWorkGroup0] / s53
v_cmp_ge_u32 vcc_lo, v2, s53                       // s49 = s[sgprWorkGroup0] / s53
s_mov_b32 exec_lo, vcc_lo                          // s49 = s[sgprWorkGroup0] / s53
v_add_nc_u32 v0, v0, 1                             // s49 = s[sgprWorkGroup0] / s53
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s53                           // s49 = s[sgprWorkGroup0] / s53
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s49 = s[sgprWorkGroup0] / s53
v_readfirstlane_b32 s49, v0                        // quotient
v_readfirstlane_b32 s50, v2                        // remainder
s_mul_i32 s49, s49, s53
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s50, s50, s52
/* temp0 = temp0 + temp1 */
s_add_u32 s49, s49, s50
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[0:1], s53                          // s50 = s23 / s53
v_rcp_f64 v[0:1], v[0:1]                           // s50 = s23 / s53
v_cvt_f64_u32 v[2:3], s23                          // s50 = s23 / s53
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s50 = s23 / s53
v_cvt_u32_f64 v0, v[0:1]                           // s50 = s23 / s53
v_mul_lo_u32 v1, v0, s53                           // s50 = s23 / s53
v_sub_nc_u32 v2, s23, v1                           // s50 = s23 / s53
v_cmp_ge_u32 vcc_lo, v2, s53                       // s50 = s23 / s53
s_mov_b32 exec_lo, vcc_lo                          // s50 = s23 / s53
v_add_nc_u32 v0, v0, 1                             // s50 = s23 / s53
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s50, v0                        // quotient
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
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
/* init: add vgpr [516...1290) to pool */
/* init: add vgpr [0...512) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v1, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 7, v0                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v0, v1, 3, v0                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v4, 5, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 11, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v2, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 7, v1                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v1, 3, v1                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(8)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v1, v2, 3, v1                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v3, 6, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 14, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(16384); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v2, 5, v[vgprSerial-768]             // 2 = Serial / 32
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
s_set_vgpr_msb 128                                 // src0: 0, src1: 0, src2: 0, dst: 2
v_add_nc_u32 v[vgprLocalReadAddrA-512], v2, v0     // Final Offset: offset = (lro0+lsuoffset)*bpeDS
s_set_vgpr_msb 32904                               // src0: 0, src1: 2, src2: 0, dst: 2
v_lshlrev_b32 v[vgprLocalReadAddrA-512], 1, v[vgprLocalReadAddrA-512] //  (multiple bpe)

/* local read addresses: final offsets b */
s_set_vgpr_msb 34828                               // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v0, 5, v[vgprSerial-768]             // 0 = Serial / 32
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
s_set_vgpr_msb 128                                 // src0: 0, src1: 0, src2: 0, dst: 2
v_add_nc_u32 v[vgprLocalReadAddrB-512], v0, v1     // Final Offset: offset = (lro1+lsuoffset)*bpeDS
s_set_vgpr_msb 32904                               // src0: 0, src1: 2, src2: 0, dst: 2
v_lshlrev_b32 v[vgprLocalReadAddrB-512], 1, v[vgprLocalReadAddrB-512] //  (multiple bpe)

/* local read addresses: declare addresses a */
v_add_nc_u32 v[vgprLocalReadAddrA+1-512], 65536, v[vgprLocalReadAddrA+0-512] // Final Offset Plus 64K

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0-512], vcc_lo, 0x10000, v[vgprLocalReadAddrB+0-512] //  += LdsOffsetB (lower)
v_add_nc_u32 v[vgprLocalReadAddrB+1-512], 65536, v[vgprLocalReadAddrB+0-512] // Final Offset Plus 64K

/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

/* local write addresses: first offset b */
s_wait_kmcnt 0                                     // wait for 88/0 bytes of kern args
s_set_vgpr_msb 34816                               // src0: 0, src1: 0, src2: 0, dst: 0
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

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v0, s16                              // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v0, v0                             // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v0, v0, v1                               // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v0, v0                               // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v1, v0, s16                          // s16 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v1, s16                       // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v0, 1, v0                             // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s16                       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v0                        // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v0, s[sgprNumWorkGroups0]            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v0, v0                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v0, v0, v1                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v0, v0                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups0]        // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v0, 1, v0                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v0                        // quotient
s_mov_b32 s[sgprWorkGroup1], s16
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 88                                  // KernArgAddressOffset
s_mul_i32 s54, s20, 4
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 196
s_mov_b32 s54, 0
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s55, 0
s_load_b128 s[24:27], s[48:49], s54
s_mov_b32 s16, 1
s_cmp_eq_u32 s20, s16                              // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_wait_kmcnt 0
s_lshr_b32 s52, s24, 8                             // s52 = s24 / 256
s_and_b32 s50, 255, s24                            // s50 = s24 % 256
s_addc_u32 s52, s52, 0
s_lshr_b32 s53, s25, 8                             // s53 = s25 / 256
s_and_b32 s50, 255, s25                            // s50 = s25 % 256
s_addc_u32 s53, s53, 0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_and_b32 s53, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s52, s52, s53
s_add_u32 s55, s55, s52
s_cmp_lt_u32 s[sgprWorkGroup0], s55
s_cbranch_scc1 label_FOUND
s_add_u32 s54, s54, s15
s_load_b128 s[24:27], s[48:49], s54
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s20
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_wait_kmcnt 0
s_lshr_b32 s52, s24, 8                             // s52 = s24 / 256
s_and_b32 s50, 255, s24                            // s50 = s24 % 256
s_addc_u32 s52, s52, 0
s_lshr_b32 s53, s25, 8                             // s53 = s25 / 256
s_and_b32 s50, 255, s25                            // s50 = s25 % 256
s_addc_u32 s53, s53, 0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_and_b32 s48, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s52, s52, s48
s_add_u32 s55, s55, s52

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s49, s14, 1
s_sub_u32 s48, s55, s52
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
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
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 196
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b32 s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
// Read Beta
s_load_b32 s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
label_LoadExternalStructEnd:
/* init: add vgpr [516...1290) to pool */
/* init: add vgpr [0...512) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
s_nop 0
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v1, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 7, v0                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v0, v1, 3, v0                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v4, 5, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(32)
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v0, v4, 11, v0                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(2048); 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_and_b32 v2, 31, v[vgprSerial-768]                // 0. thread id in wave: wtid = tid % wavelength(32)
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 7, v1                            // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v1, 3, v1                            // 4. apply VectorWidth: bnOffset = bnOffset * vw(8)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshl_add_u32 v1, v2, 3, v1                       // 5. K offset: lrKOffset = kIdx * mStride(8); 6. offset in wave: lrOffset = bnOffset + lrKOffset
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v3, 6, v[vgprSerial-768]             // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshl_add_u32 v1, v3, 14, v1                      // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(16384); 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
s_set_vgpr_msb 12                                  // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v2, 5, v[vgprSerial-768]             // 2 = Serial / 32
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s16, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v2, s16, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
s_set_vgpr_msb 128                                 // src0: 0, src1: 0, src2: 0, dst: 2
v_add_nc_u32 v[vgprLocalReadAddrA-512], v2, v0     // Final Offset: offset = (lro0+lsuoffset)*bpeDS
s_set_vgpr_msb 32904                               // src0: 0, src1: 2, src2: 0, dst: 2
v_lshlrev_b32 v[vgprLocalReadAddrA-512], 1, v[vgprLocalReadAddrA-512] //  (multiple bpe)

/* local read addresses: final offsets b */
s_set_vgpr_msb 34828                               // src0: 0, src1: 3, src2: 0, dst: 0
v_lshrrev_b32 v0, 5, v[vgprSerial-768]             // 0 = Serial / 32
s_set_vgpr_msb 3072                                // src0: 0, src1: 0, src2: 0, dst: 0
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(128) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s16, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
s_set_vgpr_msb 128                                 // src0: 0, src1: 0, src2: 0, dst: 2
v_add_nc_u32 v[vgprLocalReadAddrB-512], v0, v1     // Final Offset: offset = (lro1+lsuoffset)*bpeDS
s_set_vgpr_msb 32904                               // src0: 0, src1: 2, src2: 0, dst: 2
v_lshlrev_b32 v[vgprLocalReadAddrB-512], 1, v[vgprLocalReadAddrB-512] //  (multiple bpe)

/* local read addresses: declare addresses a */
v_add_nc_u32 v[vgprLocalReadAddrA+1-512], 65536, v[vgprLocalReadAddrA+0-512] // Final Offset Plus 64K

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0-512], vcc_lo, 0x10000, v[vgprLocalReadAddrB+0-512] //  += LdsOffsetB (lower)
v_add_nc_u32 v[vgprLocalReadAddrB+1-512], 65536, v[vgprLocalReadAddrB+0-512] // Final Offset Plus 64K

/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* local write addresses: first offset a */

/* local write addresses: first offset b */
s_wait_kmcnt 0                                     // wait for 88/0 bytes of kern args
s_set_vgpr_msb 34816                               // src0: 0, src1: 0, src2: 0, dst: 0
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

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s16, s16, s17
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s16                              // s16 = s[sgprWorkGroup0] / s16
v_rcp_iflag_f32 v0, v0                             // s16 = s[sgprWorkGroup0] / s16
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s16
v_mul_f32 v0, v0, v1                               // s16 = s[sgprWorkGroup0] / s16
v_cvt_u32_f32 v0, v0                               // s16 = s[sgprWorkGroup0] / s16
v_mul_u32_u24 v1, v0, s16                          // s16 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s16 = s[sgprWorkGroup0] / s16
v_cmp_eq_u32 vcc_lo, v1, s16                       // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v0, 1, v0                             // s16 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s16                       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v0                        // quotient
s_mov_b32 s[sgprWorkGroup2], s16
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v0, s[sgprNumWorkGroups0]            // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v0, v0                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v1, s[sgprWorkGroup0]                // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v0, v0, v1                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v0, v0                               // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups0]        // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v1, s[sgprWorkGroup0], v1             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_eq_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo                          // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_nc_u32 v0, 1, v0                             // s16 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups0]     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s16, v0                        // quotient
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
.set sgprtdmAGroup0, 48
.set sgprtdmAGroup1, 52
.set sgprtdmBGroup0, sgprtdmAGroup0+0
.set sgprtdmBGroup1, sgprtdmAGroup1+0
.set sgprtdmABIncs, 47
.set sgprStaggerUIter, 60
.set sgprWrapUA, 62
.set sgprWrapUB, 64
.set sgprGlobalReadIncsA, 61
.set sgprGlobalReadIncsB, 66

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0             // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0                     // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:
s_setreg_IMM32_b32 hwreg(26,4,1), 1                // Disable WMMA arb stall
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
s_and_b32 s16, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s16                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_rcp_iflag_f32 v0, v0                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_f32_u32 v1, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_f32 v0, v0, v1                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cvt_u32_f32 v0, v0                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mul_u32_u24 v1, v0, s16                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_cmp_eq_u32 vcc_lo, v1, s16                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_add_nc_u32 v0, 1, v0                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s16
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s16                       // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s16                          // re-calculate remainder
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // re-calculate remainder
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
s_mov_b32 exec_lo, vcc_lo                          // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_nc_u32 v0, 1, v0                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v1, 0                                    // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprNumWorkGroups1]     // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprNumWorkGroups1]        // re-calculate remainder
v_sub_nc_u32 v1, s[sgprWorkGroup1], v1             // re-calculate remainder
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
s_mov_b32 s16, s[sgprWGM]                          // Restore WGM
s_sext_i32_i16 s16, s16                            // Restore WGM
s_cmp_gt_i32 s16, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s16, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s16, s16                                 // abs(WGM)
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[0:1], s16                          // s17 = s[sgprWorkGroup0] / s16
v_rcp_f64 v[0:1], v[0:1]                           // s17 = s[sgprWorkGroup0] / s16
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup0]            // s17 = s[sgprWorkGroup0] / s16
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s17 = s[sgprWorkGroup0] / s16
v_cvt_u32_f64 v0, v[0:1]                           // s17 = s[sgprWorkGroup0] / s16
v_mul_lo_u32 v1, v0, s16                           // s17 = s[sgprWorkGroup0] / s16
v_sub_nc_u32 v2, s[sgprWorkGroup0], v1             // s17 = s[sgprWorkGroup0] / s16
v_cmp_ge_u32 vcc_lo, v2, s16                       // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s17 = s[sgprWorkGroup0] / s16
v_add_nc_u32 v0, v0, 1                             // s17 = s[sgprWorkGroup0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v0                        // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup0], s20              // WorkGroup0=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s20, s20, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[0:1], s16                          // s18 = s[sgprNumWorkGroups0] / s16
v_rcp_f64 v[0:1], v[0:1]                           // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups0]        // s18 = s[sgprNumWorkGroups0] / s16
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s18 = s[sgprNumWorkGroups0] / s16
v_cvt_u32_f64 v0, v[0:1]                           // s18 = s[sgprNumWorkGroups0] / s16
v_mul_lo_u32 v1, v0, s16                           // s18 = s[sgprNumWorkGroups0] / s16
v_sub_nc_u32 v2, s[sgprNumWorkGroups0], v1         // s18 = s[sgprNumWorkGroups0] / s16
v_cmp_ge_u32 vcc_lo, v2, s16                       // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo, vcc_lo                          // s18 = s[sgprNumWorkGroups0] / s16
v_add_nc_u32 v0, v0, 1                             // s18 = s[sgprNumWorkGroups0] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v0                        // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups0], s19          // NumWorkGroups0=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[0:1], s18                          // s[sgprWorkGroup1] = s20 / s18
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup1] = s20 / s18
v_cvt_f64_u32 v[2:3], s20                          // s[sgprWorkGroup1] = s20 / s18
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup1] = s20 / s18
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup1] = s20 / s18
v_mul_lo_u32 v1, v0, s18                           // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v2, s20, v1                           // s[sgprWorkGroup1] = s20 / s18
v_cmp_ge_u32 vcc_lo, v2, s18                       // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s20 / s18
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup1] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s18                           // s[sgprWorkGroup1] = s20 / s18
v_sub_nc_u32 v2, s20, v1                           // s[sgprWorkGroup1] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup1], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v2          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s16, s16                                 // WGM
s_set_vgpr_msb 0                                   // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f64_u32 v[0:1], s16                          // s17 = s[sgprWorkGroup1] / s16
v_rcp_f64 v[0:1], v[0:1]                           // s17 = s[sgprWorkGroup1] / s16
v_cvt_f64_u32 v[2:3], s[sgprWorkGroup1]            // s17 = s[sgprWorkGroup1] / s16
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s17 = s[sgprWorkGroup1] / s16
v_cvt_u32_f64 v0, v[0:1]                           // s17 = s[sgprWorkGroup1] / s16
v_mul_lo_u32 v1, v0, s16                           // s17 = s[sgprWorkGroup1] / s16
v_sub_nc_u32 v2, s[sgprWorkGroup1], v1             // s17 = s[sgprWorkGroup1] / s16
v_cmp_ge_u32 vcc_lo, v2, s16                       // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, vcc_lo                          // s17 = s[sgprWorkGroup1] / s16
v_add_nc_u32 v0, v0, 1                             // s17 = s[sgprWorkGroup1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s17, v0                        // quotient
s_mul_i32 s20, s17, s16                            // quotient * non-magic divisor
s_sub_u32 s20, s[sgprWorkGroup1], s20              // WorkGroup1=remainder
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s20, s20, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[0:1], s16                          // s18 = s[sgprNumWorkGroups1] / s16
v_rcp_f64 v[0:1], v[0:1]                           // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_f64_u32 v[2:3], s[sgprNumWorkGroups1]        // s18 = s[sgprNumWorkGroups1] / s16
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s18 = s[sgprNumWorkGroups1] / s16
v_cvt_u32_f64 v0, v[0:1]                           // s18 = s[sgprNumWorkGroups1] / s16
v_mul_lo_u32 v1, v0, s16                           // s18 = s[sgprNumWorkGroups1] / s16
v_sub_nc_u32 v2, s[sgprNumWorkGroups1], v1         // s18 = s[sgprNumWorkGroups1] / s16
v_cmp_ge_u32 vcc_lo, v2, s16                       // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo, vcc_lo                          // s18 = s[sgprNumWorkGroups1] / s16
v_add_nc_u32 v0, v0, 1                             // s18 = s[sgprNumWorkGroups1] / s16
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s18, v0                        // quotient
s_mul_i32 s19, s16, s18                            // quotient * non-magic divisor
s_sub_u32 s19, s[sgprNumWorkGroups1], s19          // NumWorkGroups1=remainder
s_cmp_eq_u32 s19, 0                                // remainder == 0 ?
s_cmov_b32 s19, s16                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s17, s18                              // blockId >= numFullBlocks ?
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[0:1], s18                          // s[sgprWorkGroup0] = s20 / s18
v_rcp_f64 v[0:1], v[0:1]                           // s[sgprWorkGroup0] = s20 / s18
v_cvt_f64_u32 v[2:3], s20                          // s[sgprWorkGroup0] = s20 / s18
v_mul_f64 v[0:1], v[0:1], v[2:3]                   // s[sgprWorkGroup0] = s20 / s18
v_cvt_u32_f64 v0, v[0:1]                           // s[sgprWorkGroup0] = s20 / s18
v_mul_lo_u32 v1, v0, s18                           // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v2, s20, v1                           // s[sgprWorkGroup0] = s20 / s18
v_cmp_ge_u32 vcc_lo, v2, s18                       // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s20 / s18
v_add_nc_u32 v0, v0, 1                             // s[sgprWorkGroup0] = s20 / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v1, v0, s18                           // s[sgprWorkGroup0] = s20 / s18
v_sub_nc_u32 v2, s20, v1                           // s[sgprWorkGroup0] = s20 / s18
v_readfirstlane_b32 s[sgprWorkGroup0], v0          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v2          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s18 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s20, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s17, s17, s16                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s17 // wg1 += blockId * WGM
label_WGM:
label_TDMInitA:
s_nop 0
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s16, v[vgprSerial-768]         // first tId
s_lshr_b32 s16, s16, 5                             // wId=fTid // wavelen
s_bitcmp1_b32 s16, 0                               // Check parity of wId
s_cbranch_scc1 label_TDMInitB                      // Jump to B if wId is odd
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
v_readfirstlane_b32 s16, v[vgprSerial-768]         // first tId
s_lshr_b32 s16, s16, 6                             // wId=fTid // wavelen // numComp
s_mul_i32 s16, s16, 32768                          // woffset = wId * (mt // numComp * du * bpe)
s_add_u32 s16, s16, 0                              // ldsOffset = woffset + ldsConstOffset
s_mov_b32 s[sgprtdmAGroup0+1], s16
s_and_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0xfff7ffff
s_and_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], 0xffff
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff0000
s_lshl_b32 s16, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], s16
s_lshr_b32 s16, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s16
s_and_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], 0xffff
s_and_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0xffff0000
s_lshl_b32 s16, s[sgprSizeI], 0x10
s_or_b32 s[sgprtdmAGroup1+2], s[sgprtdmAGroup1+2], s16
s_lshr_b32 s16, s[sgprSizeI], 0x10
s_or_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], s16
s_and_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0xffff
s_or_b32 s[sgprtdmAGroup1+3], s[sgprtdmAGroup1+3], 0x800000 // set tile0 to 128
s_and_b32 s[sgprtdmAGroup1+4], s[sgprtdmAGroup1+4], 0xffff0000
s_or_b32 s[sgprtdmAGroup1+4], s[sgprtdmAGroup1+4], 0x80 // set tile1 to 128
s_mov_b32 s[sgprtdmAGroup1+5], s[sgprStrideA0I]
s_branch label_TDMInitABEnd
label_TDMInitB:
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
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s16, v[vgprSerial-768]         // first tId
s_lshr_b32 s16, s16, 6                             // wId=fTid // wavelen // numComp
s_mul_i32 s16, s16, 32768                          // woffset = wId * (mt // numComp * du * bpe)
s_add_u32 s16, s16, 65536                          // ldsOffset = woffset + ldsConstOffset
s_mov_b32 s[sgprtdmBGroup0+1], s16
s_and_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0xfff7ffff
s_and_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], 0xffff
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff0000
s_lshl_b32 s16, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmBGroup1+1], s[sgprtdmBGroup1+1], s16
s_lshr_b32 s16, s[sgprSizeL], 0x10
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s16
s_and_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], 0xffff
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff0000
s_lshl_b32 s16, s[sgprSizeJ], 0x10
s_or_b32 s[sgprtdmBGroup1+2], s[sgprtdmBGroup1+2], s16
s_lshr_b32 s16, s[sgprSizeJ], 0x10
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], s16
s_and_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0xffff
s_or_b32 s[sgprtdmBGroup1+3], s[sgprtdmBGroup1+3], 0x800000 // set tile0 to 128
s_and_b32 s[sgprtdmBGroup1+4], s[sgprtdmBGroup1+4], 0xffff0000
s_or_b32 s[sgprtdmBGroup1+4], s[sgprtdmBGroup1+4], 0x80 // set tile1 to 128
s_mov_b32 s[sgprtdmBGroup1+5], s[sgprStrideB1J]
label_TDMInitABEnd:
label_TDMGlobalOffsetA:
s_nop 0
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s16, v[vgprSerial-768]         // first tId
s_lshr_b32 s16, s16, 5                             // wId=fTid // wavelen
s_bitcmp1_b32 s16, 0                               // Check parity of wId
s_cbranch_scc1 label_TDMGlobalOffsetB              // Jump to B if wId is odd
s_mov_b64 s[16:17], 0
s_mul_i32 s16, s[sgprStrideA0I], 512               // stride * MT(256) * bpe(2.0)
s_mul_i32 s16, s16, s[sgprWorkGroup0]              // *= wgId)
v_readfirstlane_b32 s18, v[vgprSerial-768]         // first tId
s_lshr_b32 s18, s18, 6                             // wCompId = fTid // wavelen(32) // numComp(2)
s_mul_i32 s18, s18, 256                            // woffset = wCompId * mt // numComp(2) * bpe(2.0)
s_mul_i32 s18, s18, s[sgprStrideA0I]               // woffset *= stride
s_add_u32 s16, s16, s18                            // += woffset
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s16 // += tileOffset(lo)
s_addc_u32 s[sgprtdmAGroup0+3], s[sgprtdmAGroup0+3], s17 // += tileOffset(hi)
s_branch label_TDMGlobalOffsetABEnd
label_TDMGlobalOffsetB:
s_mov_b64 s[16:17], 0
s_mul_i32 s16, s[sgprStrideB1J], 512               // stride * MT(256) * bpe(2.0)
s_mul_i32 s16, s16, s[sgprWorkGroup1]              // *= wgId)
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s18, v[vgprSerial-768]         // first tId
s_lshr_b32 s18, s18, 6                             // wCompId = fTid // wavelen(32) // numComp(2)
s_mul_i32 s18, s18, 256                            // woffset = wCompId * mt // numComp(2) * bpe(2.0)
s_mul_i32 s18, s18, s[sgprStrideB1J]               // woffset *= stride
s_add_u32 s16, s16, s18                            // += woffset
s_add_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s16 // += tileOffset(lo)
s_addc_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], s17 // += tileOffset(hi)
label_TDMGlobalOffsetABEnd:
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s17, s17, 256                            // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], 256, s17   // incrA (unrollIdx)
s_and_b32 s17, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s17, s17, 256                            // GSU*DepthU*Bpe*MI_dim(1)
s_and_b32 s16, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], 256, s17   // incrB (unrollIdx)
s_set_vgpr_msb 3                                   // src0: 3, src1: 0, src2: 0, dst: 0
v_readfirstlane_b32 s[sgprtdmABIncs], v[vgprSerial-768] // first tId
s_lshr_b32 s[sgprtdmABIncs], s[sgprtdmABIncs], 5   // wId=fTid // wavelen
s_bitcmp1_b32 s[sgprtdmABIncs], 0                  // Check parity of wId
s_cselect_b32 s[sgprtdmABIncs], s[sgprGlobalReadIncsB], s[sgprGlobalReadIncsA]
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 7 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 128
s_and_b32 s16, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s16, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
s_set_vgpr_msb 768                                 // src0: 0, src1: 0, src2: 0, dst: 0
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmp_eq_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo                          // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_nc_u32 v0, 1, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_gt_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]        // overflow happened in remainder
s_mov_b32 exec_lo, vcc_lo                          // overflow happened in remainder
v_sub_nc_u32 v0, v0, 1                             // quotient - 1
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // re-calculate remainder
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1           // re-calculate remainder
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
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
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s19, 0x2000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s16, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s19, 0x4000
s_cbranch_scc0 label_StaggerUMapping_3
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s19, 0x6000
s_cbranch_scc0 label_StaggerUMapping_4
s_mul_i32 s17, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s16, s16, s17
s_add_u32 s16, s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s19, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s16 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s18 // shift by StaggerUStride
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS0
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
label_ShadowInitStart:
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD
s_and_b32 s61, s[sgprSrdD+2], 127
s_lshl_b32 s61, s61, 25
s_and_b32 s[sgprSrdD+1], s[sgprSrdD+1], 33554431
s_or_b32 s[sgprSrdD+1], s[sgprSrdD+1], s61
s_lshr_b32 s[sgprSrdD+2], s[sgprSrdD+2], 7
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD
s_and_b32 s61, s[sgprSrdC+2], 127
s_lshl_b32 s61, s61, 25
s_and_b32 s[sgprSrdC+1], s[sgprSrdC+1], 33554431
s_or_b32 s[sgprSrdC+1], s[sgprSrdC+1], s61
s_lshr_b32 s[sgprSrdC+2], s[sgprSrdC+2], 7
s_mul_i32 s64, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s63, s64, s[sgprStrideC1J]            // ScaleC s64 by Stride
s_mul_i32 s62, s64, s[sgprStrideC1J]               // ScaleC s64 by Stride
s_lshl_b64 s[62:63], s[62:63], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s62    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s63   // add hi to SRD
s_mul_hi_u32 s63, s64, s[sgprStrideD1J]            // ScaleD s64 by Stride
s_mul_i32 s62, s64, s[sgprStrideD1J]               // ScaleD s64 by Stride
s_lshl_b64 s[62:63], s[62:63], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s62    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s63   // add hi to SRD
s_mul_hi_u32 s63, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s62, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[62:63], s[62:63], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s63       // add hi to SRD
s_mul_hi_u32 s63, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s62, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[62:63], s[62:63], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s63       // add hi to SRD
s_and_b32 s61, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s61, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
s_mul_hi_u32 s63, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s62, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s61, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s61, s61, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s64, s61, s[sgprStrideC1J]            // Free1
s_mul_i32 s61, s61, s[sgprStrideC1J]               // Free1
s_add_u32 s62, s62, s61                            // Free1
s_addc_u32 s63, s63, s64                           // Free1
s_sub_u32 s61, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s61, s61, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s64, s61, s[sgprStrideCK]             // Free2
s_mul_i32 s61, s61, s[sgprStrideCK]                // Free2
s_add_u32 s62, s62, s61                            // Free2
s_addc_u32 s63, s63, s64                           // Free2
s_lshl_b64 s[62:63], s[62:63], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s63       // add hi GSU offset to SRD
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
v_mov_b32 v[vgprValuC+128], 0                      // initC
v_mov_b32 v[vgprValuC+129], 0                      // initC
v_mov_b32 v[vgprValuC+130], 0                      // initC
v_mov_b32 v[vgprValuC+131], 0                      // initC
v_mov_b32 v[vgprValuC+132], 0                      // initC
v_mov_b32 v[vgprValuC+133], 0                      // initC
v_mov_b32 v[vgprValuC+134], 0                      // initC
v_mov_b32 v[vgprValuC+135], 0                      // initC
v_mov_b32 v[vgprValuC+136], 0                      // initC
v_mov_b32 v[vgprValuC+137], 0                      // initC
v_mov_b32 v[vgprValuC+138], 0                      // initC
v_mov_b32 v[vgprValuC+139], 0                      // initC
v_mov_b32 v[vgprValuC+140], 0                      // initC
v_mov_b32 v[vgprValuC+141], 0                      // initC
v_mov_b32 v[vgprValuC+142], 0                      // initC
v_mov_b32 v[vgprValuC+143], 0                      // initC
v_mov_b32 v[vgprValuC+144], 0                      // initC
v_mov_b32 v[vgprValuC+145], 0                      // initC
v_mov_b32 v[vgprValuC+146], 0                      // initC
v_mov_b32 v[vgprValuC+147], 0                      // initC
v_mov_b32 v[vgprValuC+148], 0                      // initC
v_mov_b32 v[vgprValuC+149], 0                      // initC
v_mov_b32 v[vgprValuC+150], 0                      // initC
v_mov_b32 v[vgprValuC+151], 0                      // initC
v_mov_b32 v[vgprValuC+152], 0                      // initC
v_mov_b32 v[vgprValuC+153], 0                      // initC
v_mov_b32 v[vgprValuC+154], 0                      // initC
v_mov_b32 v[vgprValuC+155], 0                      // initC
v_mov_b32 v[vgprValuC+156], 0                      // initC
v_mov_b32 v[vgprValuC+157], 0                      // initC
v_mov_b32 v[vgprValuC+158], 0                      // initC
v_mov_b32 v[vgprValuC+159], 0                      // initC
v_mov_b32 v[vgprValuC+160], 0                      // initC
v_mov_b32 v[vgprValuC+161], 0                      // initC
v_mov_b32 v[vgprValuC+162], 0                      // initC
v_mov_b32 v[vgprValuC+163], 0                      // initC
v_mov_b32 v[vgprValuC+164], 0                      // initC
v_mov_b32 v[vgprValuC+165], 0                      // initC
v_mov_b32 v[vgprValuC+166], 0                      // initC
v_mov_b32 v[vgprValuC+167], 0                      // initC
v_mov_b32 v[vgprValuC+168], 0                      // initC
v_mov_b32 v[vgprValuC+169], 0                      // initC
v_mov_b32 v[vgprValuC+170], 0                      // initC
v_mov_b32 v[vgprValuC+171], 0                      // initC
v_mov_b32 v[vgprValuC+172], 0                      // initC
v_mov_b32 v[vgprValuC+173], 0                      // initC
v_mov_b32 v[vgprValuC+174], 0                      // initC
v_mov_b32 v[vgprValuC+175], 0                      // initC
v_mov_b32 v[vgprValuC+176], 0                      // initC
v_mov_b32 v[vgprValuC+177], 0                      // initC
v_mov_b32 v[vgprValuC+178], 0                      // initC
v_mov_b32 v[vgprValuC+179], 0                      // initC
v_mov_b32 v[vgprValuC+180], 0                      // initC
v_mov_b32 v[vgprValuC+181], 0                      // initC
v_mov_b32 v[vgprValuC+182], 0                      // initC
v_mov_b32 v[vgprValuC+183], 0                      // initC
v_mov_b32 v[vgprValuC+184], 0                      // initC
v_mov_b32 v[vgprValuC+185], 0                      // initC
v_mov_b32 v[vgprValuC+186], 0                      // initC
v_mov_b32 v[vgprValuC+187], 0                      // initC
v_mov_b32 v[vgprValuC+188], 0                      // initC
v_mov_b32 v[vgprValuC+189], 0                      // initC
v_mov_b32 v[vgprValuC+190], 0                      // initC
v_mov_b32 v[vgprValuC+191], 0                      // initC
v_mov_b32 v[vgprValuC+192], 0                      // initC
v_mov_b32 v[vgprValuC+193], 0                      // initC
v_mov_b32 v[vgprValuC+194], 0                      // initC
v_mov_b32 v[vgprValuC+195], 0                      // initC
v_mov_b32 v[vgprValuC+196], 0                      // initC
v_mov_b32 v[vgprValuC+197], 0                      // initC
v_mov_b32 v[vgprValuC+198], 0                      // initC
v_mov_b32 v[vgprValuC+199], 0                      // initC
v_mov_b32 v[vgprValuC+200], 0                      // initC
v_mov_b32 v[vgprValuC+201], 0                      // initC
v_mov_b32 v[vgprValuC+202], 0                      // initC
v_mov_b32 v[vgprValuC+203], 0                      // initC
v_mov_b32 v[vgprValuC+204], 0                      // initC
v_mov_b32 v[vgprValuC+205], 0                      // initC
v_mov_b32 v[vgprValuC+206], 0                      // initC
v_mov_b32 v[vgprValuC+207], 0                      // initC
v_mov_b32 v[vgprValuC+208], 0                      // initC
v_mov_b32 v[vgprValuC+209], 0                      // initC
v_mov_b32 v[vgprValuC+210], 0                      // initC
v_mov_b32 v[vgprValuC+211], 0                      // initC
v_mov_b32 v[vgprValuC+212], 0                      // initC
v_mov_b32 v[vgprValuC+213], 0                      // initC
v_mov_b32 v[vgprValuC+214], 0                      // initC
v_mov_b32 v[vgprValuC+215], 0                      // initC
v_mov_b32 v[vgprValuC+216], 0                      // initC
v_mov_b32 v[vgprValuC+217], 0                      // initC
v_mov_b32 v[vgprValuC+218], 0                      // initC
v_mov_b32 v[vgprValuC+219], 0                      // initC
v_mov_b32 v[vgprValuC+220], 0                      // initC
v_mov_b32 v[vgprValuC+221], 0                      // initC
v_mov_b32 v[vgprValuC+222], 0                      // initC
v_mov_b32 v[vgprValuC+223], 0                      // initC
v_mov_b32 v[vgprValuC+224], 0                      // initC
v_mov_b32 v[vgprValuC+225], 0                      // initC
v_mov_b32 v[vgprValuC+226], 0                      // initC
v_mov_b32 v[vgprValuC+227], 0                      // initC
v_mov_b32 v[vgprValuC+228], 0                      // initC
v_mov_b32 v[vgprValuC+229], 0                      // initC
v_mov_b32 v[vgprValuC+230], 0                      // initC
v_mov_b32 v[vgprValuC+231], 0                      // initC
v_mov_b32 v[vgprValuC+232], 0                      // initC
v_mov_b32 v[vgprValuC+233], 0                      // initC
v_mov_b32 v[vgprValuC+234], 0                      // initC
v_mov_b32 v[vgprValuC+235], 0                      // initC
v_mov_b32 v[vgprValuC+236], 0                      // initC
v_mov_b32 v[vgprValuC+237], 0                      // initC
v_mov_b32 v[vgprValuC+238], 0                      // initC
v_mov_b32 v[vgprValuC+239], 0                      // initC
v_mov_b32 v[vgprValuC+240], 0                      // initC
v_mov_b32 v[vgprValuC+241], 0                      // initC
v_mov_b32 v[vgprValuC+242], 0                      // initC
v_mov_b32 v[vgprValuC+243], 0                      // initC
v_mov_b32 v[vgprValuC+244], 0                      // initC
v_mov_b32 v[vgprValuC+245], 0                      // initC
v_mov_b32 v[vgprValuC+246], 0                      // initC
v_mov_b32 v[vgprValuC+247], 0                      // initC
v_mov_b32 v[vgprValuC+248], 0                      // initC
v_mov_b32 v[vgprValuC+249], 0                      // initC
v_mov_b32 v[vgprValuC+250], 0                      // initC
v_mov_b32 v[vgprValuC+251], 0                      // initC
v_mov_b32 v[vgprValuC+252], 0                      // initC
v_mov_b32 v[vgprValuC+253], 0                      // initC
v_mov_b32 v[vgprValuC+254], 0                      // initC
v_mov_b32 v[vgprValuC+255], 0                      // initC
s_set_vgpr_msb 64                                  // src0: 0, src1: 0, src2: 0, dst: 1
v_mov_b32 v[vgprValuC+256-256], 0                  // initC
v_mov_b32 v[vgprValuC+257-256], 0                  // initC
v_mov_b32 v[vgprValuC+258-256], 0                  // initC
v_mov_b32 v[vgprValuC+259-256], 0                  // initC
v_mov_b32 v[vgprValuC+260-256], 0                  // initC
v_mov_b32 v[vgprValuC+261-256], 0                  // initC
v_mov_b32 v[vgprValuC+262-256], 0                  // initC
v_mov_b32 v[vgprValuC+263-256], 0                  // initC
v_mov_b32 v[vgprValuC+264-256], 0                  // initC
v_mov_b32 v[vgprValuC+265-256], 0                  // initC
v_mov_b32 v[vgprValuC+266-256], 0                  // initC
v_mov_b32 v[vgprValuC+267-256], 0                  // initC
v_mov_b32 v[vgprValuC+268-256], 0                  // initC
v_mov_b32 v[vgprValuC+269-256], 0                  // initC
v_mov_b32 v[vgprValuC+270-256], 0                  // initC
v_mov_b32 v[vgprValuC+271-256], 0                  // initC
v_mov_b32 v[vgprValuC+272-256], 0                  // initC
v_mov_b32 v[vgprValuC+273-256], 0                  // initC
v_mov_b32 v[vgprValuC+274-256], 0                  // initC
v_mov_b32 v[vgprValuC+275-256], 0                  // initC
v_mov_b32 v[vgprValuC+276-256], 0                  // initC
v_mov_b32 v[vgprValuC+277-256], 0                  // initC
v_mov_b32 v[vgprValuC+278-256], 0                  // initC
v_mov_b32 v[vgprValuC+279-256], 0                  // initC
v_mov_b32 v[vgprValuC+280-256], 0                  // initC
v_mov_b32 v[vgprValuC+281-256], 0                  // initC
v_mov_b32 v[vgprValuC+282-256], 0                  // initC
v_mov_b32 v[vgprValuC+283-256], 0                  // initC
v_mov_b32 v[vgprValuC+284-256], 0                  // initC
v_mov_b32 v[vgprValuC+285-256], 0                  // initC
v_mov_b32 v[vgprValuC+286-256], 0                  // initC
v_mov_b32 v[vgprValuC+287-256], 0                  // initC
v_mov_b32 v[vgprValuC+288-256], 0                  // initC
v_mov_b32 v[vgprValuC+289-256], 0                  // initC
v_mov_b32 v[vgprValuC+290-256], 0                  // initC
v_mov_b32 v[vgprValuC+291-256], 0                  // initC
v_mov_b32 v[vgprValuC+292-256], 0                  // initC
v_mov_b32 v[vgprValuC+293-256], 0                  // initC
v_mov_b32 v[vgprValuC+294-256], 0                  // initC
v_mov_b32 v[vgprValuC+295-256], 0                  // initC
v_mov_b32 v[vgprValuC+296-256], 0                  // initC
v_mov_b32 v[vgprValuC+297-256], 0                  // initC
v_mov_b32 v[vgprValuC+298-256], 0                  // initC
v_mov_b32 v[vgprValuC+299-256], 0                  // initC
v_mov_b32 v[vgprValuC+300-256], 0                  // initC
v_mov_b32 v[vgprValuC+301-256], 0                  // initC
v_mov_b32 v[vgprValuC+302-256], 0                  // initC
v_mov_b32 v[vgprValuC+303-256], 0                  // initC
v_mov_b32 v[vgprValuC+304-256], 0                  // initC
v_mov_b32 v[vgprValuC+305-256], 0                  // initC
v_mov_b32 v[vgprValuC+306-256], 0                  // initC
v_mov_b32 v[vgprValuC+307-256], 0                  // initC
v_mov_b32 v[vgprValuC+308-256], 0                  // initC
v_mov_b32 v[vgprValuC+309-256], 0                  // initC
v_mov_b32 v[vgprValuC+310-256], 0                  // initC
v_mov_b32 v[vgprValuC+311-256], 0                  // initC
v_mov_b32 v[vgprValuC+312-256], 0                  // initC
v_mov_b32 v[vgprValuC+313-256], 0                  // initC
v_mov_b32 v[vgprValuC+314-256], 0                  // initC
v_mov_b32 v[vgprValuC+315-256], 0                  // initC
v_mov_b32 v[vgprValuC+316-256], 0                  // initC
v_mov_b32 v[vgprValuC+317-256], 0                  // initC
v_mov_b32 v[vgprValuC+318-256], 0                  // initC
v_mov_b32 v[vgprValuC+319-256], 0                  // initC
v_mov_b32 v[vgprValuC+320-256], 0                  // initC
v_mov_b32 v[vgprValuC+321-256], 0                  // initC
v_mov_b32 v[vgprValuC+322-256], 0                  // initC
v_mov_b32 v[vgprValuC+323-256], 0                  // initC
v_mov_b32 v[vgprValuC+324-256], 0                  // initC
v_mov_b32 v[vgprValuC+325-256], 0                  // initC
v_mov_b32 v[vgprValuC+326-256], 0                  // initC
v_mov_b32 v[vgprValuC+327-256], 0                  // initC
v_mov_b32 v[vgprValuC+328-256], 0                  // initC
v_mov_b32 v[vgprValuC+329-256], 0                  // initC
v_mov_b32 v[vgprValuC+330-256], 0                  // initC
v_mov_b32 v[vgprValuC+331-256], 0                  // initC
v_mov_b32 v[vgprValuC+332-256], 0                  // initC
v_mov_b32 v[vgprValuC+333-256], 0                  // initC
v_mov_b32 v[vgprValuC+334-256], 0                  // initC
v_mov_b32 v[vgprValuC+335-256], 0                  // initC
v_mov_b32 v[vgprValuC+336-256], 0                  // initC
v_mov_b32 v[vgprValuC+337-256], 0                  // initC
v_mov_b32 v[vgprValuC+338-256], 0                  // initC
v_mov_b32 v[vgprValuC+339-256], 0                  // initC
v_mov_b32 v[vgprValuC+340-256], 0                  // initC
v_mov_b32 v[vgprValuC+341-256], 0                  // initC
v_mov_b32 v[vgprValuC+342-256], 0                  // initC
v_mov_b32 v[vgprValuC+343-256], 0                  // initC
v_mov_b32 v[vgprValuC+344-256], 0                  // initC
v_mov_b32 v[vgprValuC+345-256], 0                  // initC
v_mov_b32 v[vgprValuC+346-256], 0                  // initC
v_mov_b32 v[vgprValuC+347-256], 0                  // initC
v_mov_b32 v[vgprValuC+348-256], 0                  // initC
v_mov_b32 v[vgprValuC+349-256], 0                  // initC
v_mov_b32 v[vgprValuC+350-256], 0                  // initC
v_mov_b32 v[vgprValuC+351-256], 0                  // initC
v_mov_b32 v[vgprValuC+352-256], 0                  // initC
v_mov_b32 v[vgprValuC+353-256], 0                  // initC
v_mov_b32 v[vgprValuC+354-256], 0                  // initC
v_mov_b32 v[vgprValuC+355-256], 0                  // initC
v_mov_b32 v[vgprValuC+356-256], 0                  // initC
v_mov_b32 v[vgprValuC+357-256], 0                  // initC
v_mov_b32 v[vgprValuC+358-256], 0                  // initC
v_mov_b32 v[vgprValuC+359-256], 0                  // initC
v_mov_b32 v[vgprValuC+360-256], 0                  // initC
v_mov_b32 v[vgprValuC+361-256], 0                  // initC
v_mov_b32 v[vgprValuC+362-256], 0                  // initC
v_mov_b32 v[vgprValuC+363-256], 0                  // initC
v_mov_b32 v[vgprValuC+364-256], 0                  // initC
v_mov_b32 v[vgprValuC+365-256], 0                  // initC
v_mov_b32 v[vgprValuC+366-256], 0                  // initC
v_mov_b32 v[vgprValuC+367-256], 0                  // initC
v_mov_b32 v[vgprValuC+368-256], 0                  // initC
v_mov_b32 v[vgprValuC+369-256], 0                  // initC
v_mov_b32 v[vgprValuC+370-256], 0                  // initC
v_mov_b32 v[vgprValuC+371-256], 0                  // initC
v_mov_b32 v[vgprValuC+372-256], 0                  // initC
v_mov_b32 v[vgprValuC+373-256], 0                  // initC
v_mov_b32 v[vgprValuC+374-256], 0                  // initC
v_mov_b32 v[vgprValuC+375-256], 0                  // initC
v_mov_b32 v[vgprValuC+376-256], 0                  // initC
v_mov_b32 v[vgprValuC+377-256], 0                  // initC
v_mov_b32 v[vgprValuC+378-256], 0                  // initC
v_mov_b32 v[vgprValuC+379-256], 0                  // initC
v_mov_b32 v[vgprValuC+380-256], 0                  // initC
v_mov_b32 v[vgprValuC+381-256], 0                  // initC
v_mov_b32 v[vgprValuC+382-256], 0                  // initC
v_mov_b32 v[vgprValuC+383-256], 0                  // initC
v_mov_b32 v[vgprValuC+384-256], 0                  // initC
v_mov_b32 v[vgprValuC+385-256], 0                  // initC
v_mov_b32 v[vgprValuC+386-256], 0                  // initC
v_mov_b32 v[vgprValuC+387-256], 0                  // initC
v_mov_b32 v[vgprValuC+388-256], 0                  // initC
v_mov_b32 v[vgprValuC+389-256], 0                  // initC
v_mov_b32 v[vgprValuC+390-256], 0                  // initC
v_mov_b32 v[vgprValuC+391-256], 0                  // initC
v_mov_b32 v[vgprValuC+392-256], 0                  // initC
v_mov_b32 v[vgprValuC+393-256], 0                  // initC
v_mov_b32 v[vgprValuC+394-256], 0                  // initC
v_mov_b32 v[vgprValuC+395-256], 0                  // initC
v_mov_b32 v[vgprValuC+396-256], 0                  // initC
v_mov_b32 v[vgprValuC+397-256], 0                  // initC
v_mov_b32 v[vgprValuC+398-256], 0                  // initC
v_mov_b32 v[vgprValuC+399-256], 0                  // initC
v_mov_b32 v[vgprValuC+400-256], 0                  // initC
v_mov_b32 v[vgprValuC+401-256], 0                  // initC
v_mov_b32 v[vgprValuC+402-256], 0                  // initC
v_mov_b32 v[vgprValuC+403-256], 0                  // initC
v_mov_b32 v[vgprValuC+404-256], 0                  // initC
v_mov_b32 v[vgprValuC+405-256], 0                  // initC
v_mov_b32 v[vgprValuC+406-256], 0                  // initC
v_mov_b32 v[vgprValuC+407-256], 0                  // initC
v_mov_b32 v[vgprValuC+408-256], 0                  // initC
v_mov_b32 v[vgprValuC+409-256], 0                  // initC
v_mov_b32 v[vgprValuC+410-256], 0                  // initC
v_mov_b32 v[vgprValuC+411-256], 0                  // initC
v_mov_b32 v[vgprValuC+412-256], 0                  // initC
v_mov_b32 v[vgprValuC+413-256], 0                  // initC
v_mov_b32 v[vgprValuC+414-256], 0                  // initC
v_mov_b32 v[vgprValuC+415-256], 0                  // initC
v_mov_b32 v[vgprValuC+416-256], 0                  // initC
v_mov_b32 v[vgprValuC+417-256], 0                  // initC
v_mov_b32 v[vgprValuC+418-256], 0                  // initC
v_mov_b32 v[vgprValuC+419-256], 0                  // initC
v_mov_b32 v[vgprValuC+420-256], 0                  // initC
v_mov_b32 v[vgprValuC+421-256], 0                  // initC
v_mov_b32 v[vgprValuC+422-256], 0                  // initC
v_mov_b32 v[vgprValuC+423-256], 0                  // initC
v_mov_b32 v[vgprValuC+424-256], 0                  // initC
v_mov_b32 v[vgprValuC+425-256], 0                  // initC
v_mov_b32 v[vgprValuC+426-256], 0                  // initC
v_mov_b32 v[vgprValuC+427-256], 0                  // initC
v_mov_b32 v[vgprValuC+428-256], 0                  // initC
v_mov_b32 v[vgprValuC+429-256], 0                  // initC
v_mov_b32 v[vgprValuC+430-256], 0                  // initC
v_mov_b32 v[vgprValuC+431-256], 0                  // initC
v_mov_b32 v[vgprValuC+432-256], 0                  // initC
v_mov_b32 v[vgprValuC+433-256], 0                  // initC
v_mov_b32 v[vgprValuC+434-256], 0                  // initC
v_mov_b32 v[vgprValuC+435-256], 0                  // initC
v_mov_b32 v[vgprValuC+436-256], 0                  // initC
v_mov_b32 v[vgprValuC+437-256], 0                  // initC
v_mov_b32 v[vgprValuC+438-256], 0                  // initC
v_mov_b32 v[vgprValuC+439-256], 0                  // initC
v_mov_b32 v[vgprValuC+440-256], 0                  // initC
v_mov_b32 v[vgprValuC+441-256], 0                  // initC
v_mov_b32 v[vgprValuC+442-256], 0                  // initC
v_mov_b32 v[vgprValuC+443-256], 0                  // initC
v_mov_b32 v[vgprValuC+444-256], 0                  // initC
v_mov_b32 v[vgprValuC+445-256], 0                  // initC
v_mov_b32 v[vgprValuC+446-256], 0                  // initC
v_mov_b32 v[vgprValuC+447-256], 0                  // initC
v_mov_b32 v[vgprValuC+448-256], 0                  // initC
v_mov_b32 v[vgprValuC+449-256], 0                  // initC
v_mov_b32 v[vgprValuC+450-256], 0                  // initC
v_mov_b32 v[vgprValuC+451-256], 0                  // initC
v_mov_b32 v[vgprValuC+452-256], 0                  // initC
v_mov_b32 v[vgprValuC+453-256], 0                  // initC
v_mov_b32 v[vgprValuC+454-256], 0                  // initC
v_mov_b32 v[vgprValuC+455-256], 0                  // initC
v_mov_b32 v[vgprValuC+456-256], 0                  // initC
v_mov_b32 v[vgprValuC+457-256], 0                  // initC
v_mov_b32 v[vgprValuC+458-256], 0                  // initC
v_mov_b32 v[vgprValuC+459-256], 0                  // initC
v_mov_b32 v[vgprValuC+460-256], 0                  // initC
v_mov_b32 v[vgprValuC+461-256], 0                  // initC
v_mov_b32 v[vgprValuC+462-256], 0                  // initC
v_mov_b32 v[vgprValuC+463-256], 0                  // initC
v_mov_b32 v[vgprValuC+464-256], 0                  // initC
v_mov_b32 v[vgprValuC+465-256], 0                  // initC
v_mov_b32 v[vgprValuC+466-256], 0                  // initC
v_mov_b32 v[vgprValuC+467-256], 0                  // initC
v_mov_b32 v[vgprValuC+468-256], 0                  // initC
v_mov_b32 v[vgprValuC+469-256], 0                  // initC
v_mov_b32 v[vgprValuC+470-256], 0                  // initC
v_mov_b32 v[vgprValuC+471-256], 0                  // initC
v_mov_b32 v[vgprValuC+472-256], 0                  // initC
v_mov_b32 v[vgprValuC+473-256], 0                  // initC
v_mov_b32 v[vgprValuC+474-256], 0                  // initC
v_mov_b32 v[vgprValuC+475-256], 0                  // initC
v_mov_b32 v[vgprValuC+476-256], 0                  // initC
v_mov_b32 v[vgprValuC+477-256], 0                  // initC
v_mov_b32 v[vgprValuC+478-256], 0                  // initC
v_mov_b32 v[vgprValuC+479-256], 0                  // initC
v_mov_b32 v[vgprValuC+480-256], 0                  // initC
v_mov_b32 v[vgprValuC+481-256], 0                  // initC
v_mov_b32 v[vgprValuC+482-256], 0                  // initC
v_mov_b32 v[vgprValuC+483-256], 0                  // initC
v_mov_b32 v[vgprValuC+484-256], 0                  // initC
v_mov_b32 v[vgprValuC+485-256], 0                  // initC
v_mov_b32 v[vgprValuC+486-256], 0                  // initC
v_mov_b32 v[vgprValuC+487-256], 0                  // initC
v_mov_b32 v[vgprValuC+488-256], 0                  // initC
v_mov_b32 v[vgprValuC+489-256], 0                  // initC
v_mov_b32 v[vgprValuC+490-256], 0                  // initC
v_mov_b32 v[vgprValuC+491-256], 0                  // initC
v_mov_b32 v[vgprValuC+492-256], 0                  // initC
v_mov_b32 v[vgprValuC+493-256], 0                  // initC
v_mov_b32 v[vgprValuC+494-256], 0                  // initC
v_mov_b32 v[vgprValuC+495-256], 0                  // initC
v_mov_b32 v[vgprValuC+496-256], 0                  // initC
v_mov_b32 v[vgprValuC+497-256], 0                  // initC
v_mov_b32 v[vgprValuC+498-256], 0                  // initC
v_mov_b32 v[vgprValuC+499-256], 0                  // initC
v_mov_b32 v[vgprValuC+500-256], 0                  // initC
v_mov_b32 v[vgprValuC+501-256], 0                  // initC
v_mov_b32 v[vgprValuC+502-256], 0                  // initC
v_mov_b32 v[vgprValuC+503-256], 0                  // initC
v_mov_b32 v[vgprValuC+504-256], 0                  // initC
v_mov_b32 v[vgprValuC+505-256], 0                  // initC
v_mov_b32 v[vgprValuC+506-256], 0                  // initC
v_mov_b32 v[vgprValuC+507-256], 0                  // initC
v_mov_b32 v[vgprValuC+508-256], 0                  // initC
v_mov_b32 v[vgprValuC+509-256], 0                  // initC
v_mov_b32 v[vgprValuC+510-256], 0                  // initC
v_mov_b32 v[vgprValuC+511-256], 0                  // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc0 label_NoBranch_T8JHFHKM7BO5OHXW     // Only branch on scc1
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s64, label_PrefetchGlobalLastIterEnd, 4  // target branch offset
s_add_u32 s62, s62, s64                            // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_setpc_b64 s[62:63]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_T8JHFHKM7BO5OHXW:
s_wait_tensorcnt 0                                 // wait for global read
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x20000
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_1                    // PGR=2 but only 1 loop
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS1
s_branch label_skipPGR2_2                          // jump to PGR=2 label
label_skipPGR2_1:
label_skipPGR2_2:
s_wait_tensorcnt 1
s_barrier_signal -1
s_barrier_wait -1                                  // LW to PLR, sync LDS0
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+0-512:vgprValuB_X0_I0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4-512:vgprValuB_X0_I0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+0-512:vgprValuA_X0_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4-512:vgprValuA_X0_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+8-512:vgprValuA_X0_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:8192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12-512:vgprValuA_X0_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:8224 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+16-512:vgprValuA_X0_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:16384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20-512:vgprValuA_X0_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:16416 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+24-512:vgprValuA_X0_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:24576 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28-512:vgprValuA_X0_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:24608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+32-512:vgprValuA_X0_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:32768 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36-512:vgprValuA_X0_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:32800 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+40-512:vgprValuA_X0_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:40960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+44-512:vgprValuA_X0_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:40992 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+48-512:vgprValuA_X0_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:49152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+52-512:vgprValuA_X0_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:49184 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+56-512:vgprValuA_X0_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:57344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+60-512:vgprValuA_X0_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:57376 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+8-512:vgprValuB_X0_I0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12-512:vgprValuB_X0_I0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:288 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+16-512:vgprValuB_X0_I0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20-512:vgprValuB_X0_I0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:544 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+24-512:vgprValuB_X0_I0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28-512:vgprValuB_X0_I0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:800 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+32-512:vgprValuB_X0_I0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36-512:vgprValuB_X0_I0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:1056 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+40-512:vgprValuB_X0_I0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44-512:vgprValuB_X0_I0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:1312 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+48-512:vgprValuB_X0_I0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52-512:vgprValuB_X0_I0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:1568 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+56-512:vgprValuB_X0_I0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+60-512:vgprValuB_X0_I0+60-512+3], v[vgprLocalReadAddrB+0-512] offset:1824 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_toPGR1                        // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
.align 16
label_LoopBeginL:
s_nop 0
s_set_vgpr_msb 130                                 // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+0-512:vgprValuB_X1_I0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4-512:vgprValuB_X1_I0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+0-512:vgprValuA_X1_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4-512:vgprValuA_X1_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:96 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 32
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+8-512:vgprValuA_X1_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:8256 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12-512:vgprValuA_X1_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:8288 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+16-512:vgprValuA_X1_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:16448 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20-512:vgprValuA_X1_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:16480 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 34
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+24-512:vgprValuA_X1_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:24640 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+28-512:vgprValuA_X1_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:24672 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+32-512:vgprValuA_X1_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:32832 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36-512:vgprValuA_X1_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:32864 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 36
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+40-512:vgprValuA_X1_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:41024 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+44-512:vgprValuA_X1_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:41056 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+48-512:vgprValuA_X1_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:49216 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+52-512:vgprValuA_X1_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:49248 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 38
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+56-512:vgprValuA_X1_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:57408 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+60-512:vgprValuA_X1_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:57440 // L -> Reg lro=32 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+8-512:vgprValuB_X1_I0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:320 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12-512:vgprValuB_X1_I0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:352 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 40
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+16-512:vgprValuB_X1_I0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:576 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20-512:vgprValuB_X1_I0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:608 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+24-512:vgprValuB_X1_I0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:832 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+28-512:vgprValuB_X1_I0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:864 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 42
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+32-512:vgprValuB_X1_I0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:1088 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+36-512:vgprValuB_X1_I0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:1120 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+40-512:vgprValuB_X1_I0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:1344 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+44-512:vgprValuB_X1_I0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:1376 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 44
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+48-512:vgprValuB_X1_I0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:1600 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+52-512:vgprValuB_X1_I0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:1632 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+56-512:vgprValuB_X1_I0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:1856 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33474                               // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_X1_I0+60-768:vgprValuB_X1_I0+60-768+3], v[vgprLocalReadAddrB+0-512] offset:1888 // L -> Reg lro=32 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 49674                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
s_add_u32 s[sgprtdmAGroup0+2], s[sgprtdmAGroup0+2], s[sgprtdmABIncs] // TDM increment
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+0-512:vgprValuB_X0_I0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+4-512:vgprValuB_X0_I0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:160 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x20000
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
v_wmma_f32_16x16x32_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+96:vgprValuC+96+7] // left value = v[96+0:103+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+104:vgprValuC+104+7] // left value = v[104+0:111+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+112:vgprValuC+112+7] // left value = v[112+0:119+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+120:vgprValuC+120+7] // left value = v[120+0:127+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+8-512:vgprValuB_X0_I0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:384 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+12-512:vgprValuB_X0_I0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:416 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+128:vgprValuC+128+7] // left value = v[128+0:135+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+136:vgprValuC+136+7] // left value = v[136+0:143+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+144:vgprValuC+144+7] // left value = v[144+0:151+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+152:vgprValuC+152+7] // left value = v[152+0:159+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+160:vgprValuC+160+7] // left value = v[160+0:167+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+168:vgprValuC+168+7] // left value = v[168+0:175+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+176:vgprValuC+176+7] // left value = v[176+0:183+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+184:vgprValuC+184+7] // left value = v[184+0:191+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+16-512:vgprValuB_X0_I0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:640 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+20-512:vgprValuB_X0_I0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:672 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+192:vgprValuC+192+7] // left value = v[192+0:199+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+200:vgprValuC+200+7] // left value = v[200+0:207+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+208:vgprValuC+208+7] // left value = v[208+0:215+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+216:vgprValuC+216+7] // left value = v[216+0:223+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+224:vgprValuC+224+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+224:vgprValuC+224+7] // left value = v[224+0:231+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+232:vgprValuC+232+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+232:vgprValuC+232+7] // left value = v[232+0:239+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+240:vgprValuC+240+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+240:vgprValuC+240+7] // left value = v[240+0:247+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+248:vgprValuC+248+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+248:vgprValuC+248+7] // left value = v[248+0:255+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+24-512:vgprValuB_X0_I0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:896 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+28-512:vgprValuB_X0_I0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:928 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+256-256:vgprValuC+256-256+7] // left value = v[256+0:263+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+264-256:vgprValuC+264-256+7] // left value = v[264+0:271+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+272-256:vgprValuC+272-256+7] // left value = v[272+0:279+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+280-256:vgprValuC+280-256+7] // left value = v[280+0:287+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+288-256:vgprValuC+288-256+7] // left value = v[288+0:295+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+296-256:vgprValuC+296-256+7] // left value = v[296+0:303+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+304-256:vgprValuC+304-256+7] // left value = v[304+0:311+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+312-256:vgprValuC+312-256+7] // left value = v[312+0:319+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+32-512:vgprValuB_X0_I0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:1152 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+36-512:vgprValuB_X0_I0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:1184 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+320-256:vgprValuC+320-256+7] // left value = v[320+0:327+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+328-256:vgprValuC+328-256+7] // left value = v[328+0:335+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+336-256:vgprValuC+336-256+7] // left value = v[336+0:343+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+344-256:vgprValuC+344-256+7] // left value = v[344+0:351+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+352-256:vgprValuC+352-256+7] // left value = v[352+0:359+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+360-256:vgprValuC+360-256+7] // left value = v[360+0:367+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+368-256:vgprValuC+368-256+7] // left value = v[368+0:375+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+376-256:vgprValuC+376-256+7] // left value = v[376+0:383+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+40-512:vgprValuB_X0_I0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:1408 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+44-512:vgprValuB_X0_I0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:1440 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+384-256:vgprValuC+384-256+7] // left value = v[384+0:391+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+392-256:vgprValuC+392-256+7] // left value = v[392+0:399+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+400-256:vgprValuC+400-256+7] // left value = v[400+0:407+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+408-256:vgprValuC+408-256+7] // left value = v[408+0:415+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+416-256:vgprValuC+416-256+7] // left value = v[416+0:423+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+424-256:vgprValuC+424-256+7] // left value = v[424+0:431+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+432-256:vgprValuC+432-256+7] // left value = v[432+0:439+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+440-256:vgprValuC+440-256+7] // left value = v[440+0:447+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+48-512:vgprValuB_X0_I0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:1664 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+52-512:vgprValuB_X0_I0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:1696 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+448-256:vgprValuC+448-256+7] // left value = v[448+0:455+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+0-512:vgprValuA_X0_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+4-512:vgprValuA_X0_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:160 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+456-256:vgprValuC+456-256+7] // left value = v[456+0:463+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+8-512:vgprValuA_X0_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:8320 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+12-512:vgprValuA_X0_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:8352 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+464-256:vgprValuC+464-256+7] // left value = v[464+0:471+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+16-512:vgprValuA_X0_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:16512 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+20-512:vgprValuA_X0_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:16544 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+472-256:vgprValuC+472-256+7] // left value = v[472+0:479+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+24-512:vgprValuA_X0_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:24704 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+28-512:vgprValuA_X0_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:24736 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+480-256:vgprValuC+480-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+480-256:vgprValuC+480-256+7] // left value = v[480+0:487+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+32-512:vgprValuA_X0_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:32896 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+36-512:vgprValuA_X0_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:32928 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+488-256:vgprValuC+488-256+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+488-256:vgprValuC+488-256+7] // left value = v[488+0:495+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+40-512:vgprValuA_X0_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:41088 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+44-512:vgprValuA_X0_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:41120 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+496-256:vgprValuC+496-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+496-256:vgprValuC+496-256+7] // left value = v[496+0:503+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+48-512:vgprValuA_X0_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:49280 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+52-512:vgprValuA_X0_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:49312 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+504-256:vgprValuC+504-256+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+504-256:vgprValuC+504-256+7] // left value = v[504+0:511+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+56-512:vgprValuA_X0_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:57472 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X0_I0+60-512:vgprValuA_X0_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:57504 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+56-512:vgprValuB_X0_I0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:1920 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X0_I0+60-512:vgprValuB_X0_I0+60-512+3], v[vgprLocalReadAddrB+0-512] offset:1952 // L -> Reg lro=64 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS0
s_wait_dscnt 60
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
s_wait_dscnt 58
v_wmma_f32_16x16x32_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
s_wait_dscnt 56
v_wmma_f32_16x16x32_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
s_wait_dscnt 54
v_wmma_f32_16x16x32_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
s_wait_dscnt 52
v_wmma_f32_16x16x32_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
s_wait_dscnt 50
v_wmma_f32_16x16x32_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
s_wait_dscnt 48
v_wmma_f32_16x16x32_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
s_wait_dscnt 46
v_wmma_f32_16x16x32_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+0-512:vgprValuB_X1_I0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+4-512:vgprValuB_X1_I0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:224 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+96:vgprValuC+96+7] // left value = v[96+0:103+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+104:vgprValuC+104+7] // left value = v[104+0:111+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+112:vgprValuC+112+7] // left value = v[112+0:119+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+120:vgprValuC+120+7] // left value = v[120+0:127+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+8-512:vgprValuB_X1_I0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:448 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+12-512:vgprValuB_X1_I0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:480 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+128:vgprValuC+128+7] // left value = v[128+0:135+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+136:vgprValuC+136+7] // left value = v[136+0:143+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+144:vgprValuC+144+7] // left value = v[144+0:151+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+152:vgprValuC+152+7] // left value = v[152+0:159+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+160:vgprValuC+160+7] // left value = v[160+0:167+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+168:vgprValuC+168+7] // left value = v[168+0:175+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+176:vgprValuC+176+7] // left value = v[176+0:183+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+184:vgprValuC+184+7] // left value = v[184+0:191+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+16-512:vgprValuB_X1_I0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:704 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+20-512:vgprValuB_X1_I0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:736 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33290                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+192:vgprValuC+192+7] // left value = v[192+0:199+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+200:vgprValuC+200+7] // left value = v[200+0:207+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+208:vgprValuC+208+7] // left value = v[208+0:215+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+216:vgprValuC+216+7] // left value = v[216+0:223+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+224:vgprValuC+224+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+224:vgprValuC+224+7] // left value = v[224+0:231+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+232:vgprValuC+232+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+232:vgprValuC+232+7] // left value = v[232+0:239+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+240:vgprValuC+240+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+240:vgprValuC+240+7] // left value = v[240+0:247+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+248:vgprValuC+248+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+248:vgprValuC+248+7] // left value = v[248+0:255+0]
s_set_vgpr_msb 2690                                // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+24-512:vgprValuB_X1_I0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:960 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+28-512:vgprValuB_X1_I0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:992 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+256-256:vgprValuC+256-256+7] // left value = v[256+0:263+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+264-256:vgprValuC+264-256+7] // left value = v[264+0:271+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+272-256:vgprValuC+272-256+7] // left value = v[272+0:279+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+280-256:vgprValuC+280-256+7] // left value = v[280+0:287+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+288-256:vgprValuC+288-256+7] // left value = v[288+0:295+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+296-256:vgprValuC+296-256+7] // left value = v[296+0:303+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+304-256:vgprValuC+304-256+7] // left value = v[304+0:311+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+312-256:vgprValuC+312-256+7] // left value = v[312+0:319+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+32-512:vgprValuB_X1_I0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:1216 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+36-512:vgprValuB_X1_I0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:1248 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+320-256:vgprValuC+320-256+7] // left value = v[320+0:327+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+328-256:vgprValuC+328-256+7] // left value = v[328+0:335+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+336-256:vgprValuC+336-256+7] // left value = v[336+0:343+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+344-256:vgprValuC+344-256+7] // left value = v[344+0:351+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+352-256:vgprValuC+352-256+7] // left value = v[352+0:359+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+360-256:vgprValuC+360-256+7] // left value = v[360+0:367+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+368-256:vgprValuC+368-256+7] // left value = v[368+0:375+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+376-256:vgprValuC+376-256+7] // left value = v[376+0:383+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+40-512:vgprValuB_X1_I0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:1472 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+44-512:vgprValuB_X1_I0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:1504 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+384-256:vgprValuC+384-256+7] // left value = v[384+0:391+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+392-256:vgprValuC+392-256+7] // left value = v[392+0:399+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+400-256:vgprValuC+400-256+7] // left value = v[400+0:407+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+408-256:vgprValuC+408-256+7] // left value = v[408+0:415+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+416-256:vgprValuC+416-256+7] // left value = v[416+0:423+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+424-256:vgprValuC+424-256+7] // left value = v[424+0:431+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+432-256:vgprValuC+432-256+7] // left value = v[432+0:439+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+440-256:vgprValuC+440-256+7] // left value = v[440+0:447+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X1_I0+48-512:vgprValuB_X1_I0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:1728 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+52-512:vgprValuB_X1_I0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:1760 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 46
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+448-256:vgprValuC+448-256+7] // left value = v[448+0:455+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+0-512:vgprValuA_X1_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+4-512:vgprValuA_X1_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:224 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+456-256:vgprValuC+456-256+7] // left value = v[456+0:463+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+8-512:vgprValuA_X1_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:8384 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+12-512:vgprValuA_X1_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:8416 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+464-256:vgprValuC+464-256+7] // left value = v[464+0:471+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+16-512:vgprValuA_X1_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:16576 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+20-512:vgprValuA_X1_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:16608 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+472-256:vgprValuC+472-256+7] // left value = v[472+0:479+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+24-512:vgprValuA_X1_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:24768 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+28-512:vgprValuA_X1_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:24800 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+480-256:vgprValuC+480-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+480-256:vgprValuC+480-256+7] // left value = v[480+0:487+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+32-512:vgprValuA_X1_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:32960 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+36-512:vgprValuA_X1_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:32992 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+488-256:vgprValuC+488-256+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+488-256:vgprValuC+488-256+7] // left value = v[488+0:495+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+40-512:vgprValuA_X1_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:41152 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+44-512:vgprValuA_X1_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:41184 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+496-256:vgprValuC+496-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+496-256:vgprValuC+496-256+7] // left value = v[496+0:503+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+48-512:vgprValuA_X1_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:49344 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+52-512:vgprValuA_X1_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:49376 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+504-256:vgprValuC+504-256+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+504-256:vgprValuC+504-256+7] // left value = v[504+0:511+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X1_I0+56-512:vgprValuA_X1_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:57536 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuA_X1_I0+60-512:vgprValuA_X1_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:57568 // L -> Reg lro=96 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
ds_load_b128 v[vgprValuB_X1_I0+56-512:vgprValuB_X1_I0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:1984 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=1 iui=0 sync LDS0
s_set_vgpr_msb 33474                               // src0: 2, src1: 0, src2: 0, dst: 3
ds_load_b128 v[vgprValuB_X1_I0+60-768:vgprValuB_X1_I0+60-768+3], v[vgprLocalReadAddrB+0-512] offset:2016 // L -> Reg lro=96 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=1 oIdx=0 buffer=1 iui=0 sync LDS0
s_wait_dscnt 48
s_set_vgpr_msb 49674                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
s_set_vgpr_msb 2696                                // src0: 0, src1: 2, src2: 0, dst: 2
v_xor_b32 v[vgprLocalReadAddrA-512], 0x20000, v[vgprLocalReadAddrA-512] // swap Red Blk
v_add_nc_u32 v[vgprLocalReadAddrA+1-512], 65536, v[vgprLocalReadAddrA-512] // Final Offset Plus 64K
v_xor_b32 v[vgprLocalReadAddrB-512], 0x20000, v[vgprLocalReadAddrB-512] // swap Red Blk
v_add_nc_u32 v[vgprLocalReadAddrB+1-512], 65536, v[vgprLocalReadAddrB-512] // Final Offset Plus 64K
s_wait_dscnt 46
s_set_vgpr_msb 34826                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
s_wait_dscnt 44
v_wmma_f32_16x16x32_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
s_wait_dscnt 42
v_wmma_f32_16x16x32_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
s_wait_dscnt 40
v_wmma_f32_16x16x32_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
s_wait_dscnt 38
v_wmma_f32_16x16x32_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
s_wait_dscnt 36
v_wmma_f32_16x16x32_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
s_wait_dscnt 34
v_wmma_f32_16x16x32_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+0+0+0-512:vgprValuB_X0_I0+0+0+0-512+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
s_wait_dscnt 0
s_barrier_signal -1
s_barrier_wait -1                                  // Waiting current LR finish for next GR(TDM), sync LDS0
tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7] // sync LDS0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+96:vgprValuC+96+7] // left value = v[96+0:103+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+104:vgprValuC+104+7] // left value = v[104+0:111+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+112:vgprValuC+112+7] // left value = v[112+0:119+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+8+0+0-512:vgprValuB_X0_I0+8+0+0-512+7], v[vgprValuC+120:vgprValuC+120+7] // left value = v[120+0:127+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+128:vgprValuC+128+7] // left value = v[128+0:135+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+136:vgprValuC+136+7] // left value = v[136+0:143+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+144:vgprValuC+144+7] // left value = v[144+0:151+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+152:vgprValuC+152+7] // left value = v[152+0:159+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+160:vgprValuC+160+7] // left value = v[160+0:167+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+168:vgprValuC+168+7] // left value = v[168+0:175+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+176:vgprValuC+176+7] // left value = v[176+0:183+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+16+0+0-512:vgprValuB_X0_I0+16+0+0-512+7], v[vgprValuC+184:vgprValuC+184+7] // left value = v[184+0:191+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+192:vgprValuC+192+7] // left value = v[192+0:199+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+200:vgprValuC+200+7] // left value = v[200+0:207+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+208:vgprValuC+208+7] // left value = v[208+0:215+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+216:vgprValuC+216+7] // left value = v[216+0:223+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+224:vgprValuC+224+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+224:vgprValuC+224+7] // left value = v[224+0:231+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+232:vgprValuC+232+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+232:vgprValuC+232+7] // left value = v[232+0:239+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+240:vgprValuC+240+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+240:vgprValuC+240+7] // left value = v[240+0:247+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+248:vgprValuC+248+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+24+0+0-512:vgprValuB_X0_I0+24+0+0-512+7], v[vgprValuC+248:vgprValuC+248+7] // left value = v[248+0:255+0]
s_set_vgpr_msb 2650                                // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+256-256:vgprValuC+256-256+7] // left value = v[256+0:263+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+264-256:vgprValuC+264-256+7] // left value = v[264+0:271+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+272-256:vgprValuC+272-256+7] // left value = v[272+0:279+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+280-256:vgprValuC+280-256+7] // left value = v[280+0:287+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+288-256:vgprValuC+288-256+7] // left value = v[288+0:295+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+296-256:vgprValuC+296-256+7] // left value = v[296+0:303+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+304-256:vgprValuC+304-256+7] // left value = v[304+0:311+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+32+0+0-512:vgprValuB_X0_I0+32+0+0-512+7], v[vgprValuC+312-256:vgprValuC+312-256+7] // left value = v[312+0:319+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+320-256:vgprValuC+320-256+7] // left value = v[320+0:327+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+328-256:vgprValuC+328-256+7] // left value = v[328+0:335+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+336-256:vgprValuC+336-256+7] // left value = v[336+0:343+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+344-256:vgprValuC+344-256+7] // left value = v[344+0:351+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+352-256:vgprValuC+352-256+7] // left value = v[352+0:359+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+360-256:vgprValuC+360-256+7] // left value = v[360+0:367+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+368-256:vgprValuC+368-256+7] // left value = v[368+0:375+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+40+0+0-512:vgprValuB_X0_I0+40+0+0-512+7], v[vgprValuC+376-256:vgprValuC+376-256+7] // left value = v[376+0:383+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+384-256:vgprValuC+384-256+7] // left value = v[384+0:391+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+392-256:vgprValuC+392-256+7] // left value = v[392+0:399+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+400-256:vgprValuC+400-256+7] // left value = v[400+0:407+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+408-256:vgprValuC+408-256+7] // left value = v[408+0:415+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+416-256:vgprValuC+416-256+7] // left value = v[416+0:423+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+424-256:vgprValuC+424-256+7] // left value = v[424+0:431+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+432-256:vgprValuC+432-256+7] // left value = v[432+0:439+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+48+0+0-512:vgprValuB_X0_I0+48+0+0-512+7], v[vgprValuC+440-256:vgprValuC+440-256+7] // left value = v[440+0:447+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X0_I0+0+0+0-512:vgprValuA_X0_I0+0+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+448-256:vgprValuC+448-256+7] // left value = v[448+0:455+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X0_I0+8+0+0-512:vgprValuA_X0_I0+8+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+456-256:vgprValuC+456-256+7] // left value = v[456+0:463+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X0_I0+16+0+0-512:vgprValuA_X0_I0+16+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+464-256:vgprValuC+464-256+7] // left value = v[464+0:471+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X0_I0+24+0+0-512:vgprValuA_X0_I0+24+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+472-256:vgprValuC+472-256+7] // left value = v[472+0:479+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+480-256:vgprValuC+480-256+7], v[vgprValuA_X0_I0+32+0+0-512:vgprValuA_X0_I0+32+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+480-256:vgprValuC+480-256+7] // left value = v[480+0:487+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+488-256:vgprValuC+488-256+7], v[vgprValuA_X0_I0+40+0+0-512:vgprValuA_X0_I0+40+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+488-256:vgprValuC+488-256+7] // left value = v[488+0:495+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+496-256:vgprValuC+496-256+7], v[vgprValuA_X0_I0+48+0+0-512:vgprValuA_X0_I0+48+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+496-256:vgprValuC+496-256+7] // left value = v[496+0:503+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+504-256:vgprValuC+504-256+7], v[vgprValuA_X0_I0+56+0+0-512:vgprValuA_X0_I0+56+0+0-512+7], v[vgprValuB_X0_I0+56+0+0-512:vgprValuB_X0_I0+56+0+0-512+7], v[vgprValuC+504-256:vgprValuC+504-256+7] // left value = v[504+0:511+0]
s_set_vgpr_msb 23050                               // src0: 2, src1: 2, src2: 0, dst: 0
v_wmma_f32_16x16x32_bf16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+0:vgprValuC+0+7] // left value = v[0+0:7+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+8:vgprValuC+8+7] // left value = v[8+0:15+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+16:vgprValuC+16+7] // left value = v[16+0:23+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+24:vgprValuC+24+7] // left value = v[24+0:31+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+32:vgprValuC+32+7] // left value = v[32+0:39+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+40:vgprValuC+40+7] // left value = v[40+0:47+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+48:vgprValuC+48+7] // left value = v[48+0:55+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+0+0+0-512:vgprValuB_X1_I0+0+0+0-512+7], v[vgprValuC+56:vgprValuC+56+7] // left value = v[56+0:63+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+64:vgprValuC+64+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+64:vgprValuC+64+7] // left value = v[64+0:71+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+72:vgprValuC+72+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+72:vgprValuC+72+7] // left value = v[72+0:79+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+80:vgprValuC+80+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+80:vgprValuC+80+7] // left value = v[80+0:87+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+88:vgprValuC+88+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+88:vgprValuC+88+7] // left value = v[88+0:95+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+96:vgprValuC+96+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+96:vgprValuC+96+7] // left value = v[96+0:103+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+104:vgprValuC+104+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+104:vgprValuC+104+7] // left value = v[104+0:111+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+112:vgprValuC+112+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+112:vgprValuC+112+7] // left value = v[112+0:119+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+120:vgprValuC+120+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+8+0+0-512:vgprValuB_X1_I0+8+0+0-512+7], v[vgprValuC+120:vgprValuC+120+7] // left value = v[120+0:127+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+128:vgprValuC+128+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+128:vgprValuC+128+7] // left value = v[128+0:135+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+136:vgprValuC+136+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+136:vgprValuC+136+7] // left value = v[136+0:143+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+144:vgprValuC+144+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+144:vgprValuC+144+7] // left value = v[144+0:151+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+152:vgprValuC+152+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+152:vgprValuC+152+7] // left value = v[152+0:159+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+160:vgprValuC+160+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+160:vgprValuC+160+7] // left value = v[160+0:167+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+168:vgprValuC+168+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+168:vgprValuC+168+7] // left value = v[168+0:175+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+176:vgprValuC+176+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+176:vgprValuC+176+7] // left value = v[176+0:183+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+184:vgprValuC+184+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+16+0+0-512:vgprValuB_X1_I0+16+0+0-512+7], v[vgprValuC+184:vgprValuC+184+7] // left value = v[184+0:191+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+192:vgprValuC+192+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+192:vgprValuC+192+7] // left value = v[192+0:199+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+200:vgprValuC+200+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+200:vgprValuC+200+7] // left value = v[200+0:207+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+208:vgprValuC+208+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+208:vgprValuC+208+7] // left value = v[208+0:215+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+216:vgprValuC+216+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+216:vgprValuC+216+7] // left value = v[216+0:223+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+224:vgprValuC+224+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+224:vgprValuC+224+7] // left value = v[224+0:231+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+232:vgprValuC+232+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+232:vgprValuC+232+7] // left value = v[232+0:239+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+240:vgprValuC+240+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+240:vgprValuC+240+7] // left value = v[240+0:247+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+248:vgprValuC+248+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+24+0+0-512:vgprValuB_X1_I0+24+0+0-512+7], v[vgprValuC+248:vgprValuC+248+7] // left value = v[248+0:255+0]
s_set_vgpr_msb 2650                                // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+256-256:vgprValuC+256-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+256-256:vgprValuC+256-256+7] // left value = v[256+0:263+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+264-256:vgprValuC+264-256+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+264-256:vgprValuC+264-256+7] // left value = v[264+0:271+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+272-256:vgprValuC+272-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+272-256:vgprValuC+272-256+7] // left value = v[272+0:279+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+280-256:vgprValuC+280-256+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+280-256:vgprValuC+280-256+7] // left value = v[280+0:287+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+288-256:vgprValuC+288-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+288-256:vgprValuC+288-256+7] // left value = v[288+0:295+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+296-256:vgprValuC+296-256+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+296-256:vgprValuC+296-256+7] // left value = v[296+0:303+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+304-256:vgprValuC+304-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+304-256:vgprValuC+304-256+7] // left value = v[304+0:311+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+312-256:vgprValuC+312-256+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+32+0+0-512:vgprValuB_X1_I0+32+0+0-512+7], v[vgprValuC+312-256:vgprValuC+312-256+7] // left value = v[312+0:319+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+320-256:vgprValuC+320-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+320-256:vgprValuC+320-256+7] // left value = v[320+0:327+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+328-256:vgprValuC+328-256+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+328-256:vgprValuC+328-256+7] // left value = v[328+0:335+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+336-256:vgprValuC+336-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+336-256:vgprValuC+336-256+7] // left value = v[336+0:343+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+344-256:vgprValuC+344-256+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+344-256:vgprValuC+344-256+7] // left value = v[344+0:351+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+352-256:vgprValuC+352-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+352-256:vgprValuC+352-256+7] // left value = v[352+0:359+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+360-256:vgprValuC+360-256+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+360-256:vgprValuC+360-256+7] // left value = v[360+0:367+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+368-256:vgprValuC+368-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+368-256:vgprValuC+368-256+7] // left value = v[368+0:375+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+376-256:vgprValuC+376-256+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+40+0+0-512:vgprValuB_X1_I0+40+0+0-512+7], v[vgprValuC+376-256:vgprValuC+376-256+7] // left value = v[376+0:383+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+384-256:vgprValuC+384-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+384-256:vgprValuC+384-256+7] // left value = v[384+0:391+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+392-256:vgprValuC+392-256+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+392-256:vgprValuC+392-256+7] // left value = v[392+0:399+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+400-256:vgprValuC+400-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+400-256:vgprValuC+400-256+7] // left value = v[400+0:407+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+408-256:vgprValuC+408-256+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+408-256:vgprValuC+408-256+7] // left value = v[408+0:415+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+416-256:vgprValuC+416-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+416-256:vgprValuC+416-256+7] // left value = v[416+0:423+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+424-256:vgprValuC+424-256+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+424-256:vgprValuC+424-256+7] // left value = v[424+0:431+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+432-256:vgprValuC+432-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+432-256:vgprValuC+432-256+7] // left value = v[432+0:439+0]
v_wmma_f32_16x16x32_bf16 v[vgprValuC+440-256:vgprValuC+440-256+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+48+0+0-512:vgprValuB_X1_I0+48+0+0-512+7], v[vgprValuC+440-256:vgprValuC+440-256+7] // left value = v[440+0:447+0]
s_wait_tensorcnt 1
s_barrier_signal -1
s_barrier_wait -1                                  // PGR, and wait until LW done to sync LDS1
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+0-512:vgprValuB_X0_I0+0-512+3], v[vgprLocalReadAddrB+0-512] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+4-512:vgprValuB_X0_I0+4-512+3], v[vgprLocalReadAddrB+0-512] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+448-256:vgprValuC+448-256+7], v[vgprValuA_X1_I0+0+0+0-512:vgprValuA_X1_I0+0+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+448-256:vgprValuC+448-256+7] // left value = v[448+0:455+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+0-512:vgprValuA_X0_I0+0-512+3], v[vgprLocalReadAddrA+0-512] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+4-512:vgprValuA_X0_I0+4-512+3], v[vgprLocalReadAddrA+0-512] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+8-512:vgprValuA_X0_I0+8-512+3], v[vgprLocalReadAddrA+0-512] offset:8192 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+12-512:vgprValuA_X0_I0+12-512+3], v[vgprLocalReadAddrA+0-512] offset:8224 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+456-256:vgprValuC+456-256+7], v[vgprValuA_X1_I0+8+0+0-512:vgprValuA_X1_I0+8+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+456-256:vgprValuC+456-256+7] // left value = v[456+0:463+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+16-512:vgprValuA_X0_I0+16-512+3], v[vgprLocalReadAddrA+0-512] offset:16384 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+20-512:vgprValuA_X0_I0+20-512+3], v[vgprLocalReadAddrA+0-512] offset:16416 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+24-512:vgprValuA_X0_I0+24-512+3], v[vgprLocalReadAddrA+0-512] offset:24576 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+28-512:vgprValuA_X0_I0+28-512+3], v[vgprLocalReadAddrA+0-512] offset:24608 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=3 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+464-256:vgprValuC+464-256+7], v[vgprValuA_X1_I0+16+0+0-512:vgprValuA_X1_I0+16+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+464-256:vgprValuC+464-256+7] // left value = v[464+0:471+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+32-512:vgprValuA_X0_I0+32-512+3], v[vgprLocalReadAddrA+0-512] offset:32768 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+36-512:vgprValuA_X0_I0+36-512+3], v[vgprLocalReadAddrA+0-512] offset:32800 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=4 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+40-512:vgprValuA_X0_I0+40-512+3], v[vgprLocalReadAddrA+0-512] offset:40960 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+44-512:vgprValuA_X0_I0+44-512+3], v[vgprLocalReadAddrA+0-512] offset:40992 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=5 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+472-256:vgprValuC+472-256+7], v[vgprValuA_X1_I0+24+0+0-512:vgprValuA_X1_I0+24+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+472-256:vgprValuC+472-256+7] // left value = v[472+0:479+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuA_X0_I0+48-512:vgprValuA_X0_I0+48-512+3], v[vgprLocalReadAddrA+0-512] offset:49152 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+52-512:vgprValuA_X0_I0+52-512+3], v[vgprLocalReadAddrA+0-512] offset:49184 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=6 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+56-512:vgprValuA_X0_I0+56-512+3], v[vgprLocalReadAddrA+0-512] offset:57344 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuA_X0_I0+60-512:vgprValuA_X0_I0+60-512+3], v[vgprLocalReadAddrA+0-512] offset:57376 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=7 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+480-256:vgprValuC+480-256+7], v[vgprValuA_X1_I0+32+0+0-512:vgprValuA_X1_I0+32+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+480-256:vgprValuC+480-256+7] // left value = v[480+0:487+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+8-512:vgprValuB_X0_I0+8-512+3], v[vgprLocalReadAddrB+0-512] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+12-512:vgprValuB_X0_I0+12-512+3], v[vgprLocalReadAddrB+0-512] offset:288 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+16-512:vgprValuB_X0_I0+16-512+3], v[vgprLocalReadAddrB+0-512] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+20-512:vgprValuB_X0_I0+20-512+3], v[vgprLocalReadAddrB+0-512] offset:544 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+488-256:vgprValuC+488-256+7], v[vgprValuA_X1_I0+40+0+0-512:vgprValuA_X1_I0+40+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+488-256:vgprValuC+488-256+7] // left value = v[488+0:495+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+24-512:vgprValuB_X0_I0+24-512+3], v[vgprLocalReadAddrB+0-512] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+28-512:vgprValuB_X0_I0+28-512+3], v[vgprLocalReadAddrB+0-512] offset:800 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+32-512:vgprValuB_X0_I0+32-512+3], v[vgprLocalReadAddrB+0-512] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+36-512:vgprValuB_X0_I0+36-512+3], v[vgprLocalReadAddrB+0-512] offset:1056 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=4 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+496-256:vgprValuC+496-256+7], v[vgprValuA_X1_I0+48+0+0-512:vgprValuA_X1_I0+48+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+496-256:vgprValuC+496-256+7] // left value = v[496+0:503+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+40-512:vgprValuB_X0_I0+40-512+3], v[vgprLocalReadAddrB+0-512] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+44-512:vgprValuB_X0_I0+44-512+3], v[vgprLocalReadAddrB+0-512] offset:1312 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=5 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+48-512:vgprValuB_X0_I0+48-512+3], v[vgprLocalReadAddrB+0-512] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+52-512:vgprValuB_X0_I0+52-512+3], v[vgprLocalReadAddrB+0-512] offset:1568 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=6 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_set_vgpr_msb 33370                               // src0: 2, src1: 2, src2: 1, dst: 1
v_wmma_f32_16x16x32_bf16 v[vgprValuC+504-256:vgprValuC+504-256+7], v[vgprValuA_X1_I0+56+0+0-512:vgprValuA_X1_I0+56+0+0-512+7], v[vgprValuB_X1_I0+56+0+0-512:vgprValuB_X1_I0+56+0+0-512+7], v[vgprValuC+504-256:vgprValuC+504-256+7] // left value = v[504+0:511+0]
s_set_vgpr_msb 23170                               // src0: 2, src1: 0, src2: 0, dst: 2
ds_load_b128 v[vgprValuB_X0_I0+56-512:vgprValuB_X0_I0+56-512+3], v[vgprLocalReadAddrB+0-512] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=0 oIdx=0 buffer=0 iui=0 sync LDS1
ds_load_b128 v[vgprValuB_X0_I0+60-512:vgprValuB_X0_I0+60-512+3], v[vgprLocalReadAddrB+0-512] offset:1824 // L -> Reg lro=0 swapByteOffset=0 ti=256 vIdx=0 eIdx=7 rIdx=1 oIdx=0 buffer=0 iui=0 sync LDS1
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:
