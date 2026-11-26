
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT2x3x2048_Dot2_gfx1151
.globl RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT2x3x2048_Dot2_gfx1151
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT2x3x2048_Dot2_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT2x3x2048_Dot2_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 255 // vgprs
  .amdhsa_next_free_sgpr 82 // sgprs
  .amdhsa_group_segment_fixed_size 3640 // lds bytes
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
/* Num VGPR   =255 */
/* Num AccVGPR=0 */
/* Num SGPR   =82 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 2 x 3 */
/* SubGroup= 1 x 1 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=32, GlobalReadVectorWidthB=32 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 3
  StaggerU: 0
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT2x3x2048_Dot2_gfx1151
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT2x3x2048_Dot2_gfx1151.kd'
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
        .value_type:      i4
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
      - .name:            AddressScaleA
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleB
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleZeroA
        .size:            8
        .offset:          120
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            batchOffsetD
        .size:            8
        .offset:          128
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          136
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          144
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          152
        .value_kind:      by_value
        .value_type:      u64
      - .name:            groupSize
        .size:            4
        .offset:          160
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   3640
    .kernarg_segment_align:      8
    .kernarg_segment_size:       168
    .max_flat_workgroup_size:    32
    .private_segment_fixed_size: 0
    .sgpr_count:                 82
    .sgpr_spill_count:           0
    .vgpr_count:                 255
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT2x3x2048_Dot2_gfx1151:
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
/* ValuC range: [0-6),  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 60
.set vgprLocalWriteAddrA, 54
.set vgprLocalWriteAddrB, 55
.set vgprGlobalReadOffsetA, 6
.set vgprGlobalReadOffsetB, 10
.set vgprGlobalReadOffsetScaleA, 34
.set vgprG2LScaleA, 38
.set vgprG2LScaleZeroA, 42
.set vgprGlobalReadOffsetScaleZeroA, 46
.set vgprGlobalReadByteOffsetScaleZeroA, 50
.set vgprLocalReadAddrA, 56
.set vgprLocalReadAddrB, 57
.set vgprSerial, 221

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+65
.set vgprG2LA_BASE, vgprBase+0
.set vgprG2LB_BASE, vgprBase+65
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+2
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+3
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
.set sgprDot2TailRem, 14
.set sgprDot2TailMask, 15
.set sgprNumWorkGroups0, 17
.set sgprNumWorkGroups1, 18
.set sgprSizesFree, 20
.set sgprSizesSum, 23
.set sgprAddressA, 24
.set sgprAddressB, 26
.set sgprStridesA, 28
.set sgprStridesB, 30
.set sgprAlpha, 32
.set sgprBeta, 33
.set sgprAddressD, 34
.set sgprAddressC, 36
.set sgprStridesD, 38
.set sgprStridesC, 40
.set sgprGSU, 42
.set sgprAddressScaleA, 44
.set sgprAddressScaleB, 46
.set sgprStrideScaleA, 43
.set sgprSrdScaleA, 48
.set sgprScaleAGMinus1, 52
.set sgprScaleAGShift, 53
.set sgprScaleAKMask, 54
.set sgprScaleAInc, 55
.set sgprScaleAKCnt, 56
.set sgprScaleAPkMagic, 57
.set sgprAddressScaleZeroA, 60
.set sgprSrdScaleZeroA, 64

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

.set MT0, 2
.set MT1, 3
.set DepthU, 2048
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 32
.set SrdShiftLeftB, 32
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0xfffff000

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

/* Global Offset B */

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_b32 s19, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s62, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s63, s[sgprKernArgAddress:sgprKernArgAddress+1], 12

/* Load Kernel Args */
s_load_b512 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b128 s[36:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b64 s[40:41], s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa0
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s59, s19, 0x1e                          // Get arg type
s_and_b32 s19, 0x3fffffff, s19                     // Get nums of gemm
s_cmp_eq_u32 s59, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s59, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
label_LoadArgsEnd:
s_and_b32 s[sgprGSU], s62, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s59
s_mov_b32 m0, 0xe38                                // LDS clamp at 3640 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s72, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s72, s72                             // Get log(WGMXCC)
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s72, 0
s_cbranch_scc1 label_WGMXCC
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s59, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s59, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [60...281) to pool */
/* init: add vgpr [0...6) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 32 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_and_b32 v1, 31, v[vgprSerial]                    // 1 = Serial % 32
/* unroll *= glvw */
v_lshlrev_b32 v1, 5, v1                            // v1 = v1 * 32
/* LVCB = 32 */
/* v3 = B-unroll = serial%LVCB */
v_lshlrev_b32 v3, 5, v[vgprSerial]                 // B-unroll = serial * glvw
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args over preload
s_add_u32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 1 // size + MT0-1
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 1 // ceil(size / MT0)
s_mul_hi_u32 s69, s[sgprSizesFree+1], -1431655765  // STATIC_DIV: divisor=3
s_mul_i32 s68, s[sgprSizesFree+1], -1431655765     // tmp = dividend * magic
s_lshr_b64 s[68:69], s[68:69], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s68               // quotient
s_mul_i32 s68, s[sgprNumWorkGroups1], 3            // quotient*divisor
s_sub_u32 s62, s[sgprSizesFree+1], s68             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s62, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mov_b32 s[sgprWorkGroup2], 0
s_mov_b32 s[sgprWorkGroup1], 0
s_cmp_ge_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // wg2 or wg1 != 0?
s_cbranch_scc1 label_RemapWG
label_RemapWGDone:
label_MultiGemmEnd:
.set sgprSrdA, 68
.set sgprSrdB, 72
.set sgprShadowLimitA, 62
.set sgprShadowLimitB, 76
.set sgprGlobalReadIncsA, 19
.set sgprGlobalReadIncsB, 59
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 64 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift
label_Skip_Address_Prepad_For_Pointer_Array:  /// Skip pre-padding of address for pointer array case

/* Short circuit condition if Alpha == 0, then sumDims=0 */
s_and_b32 s19, s[sgprAlpha], 0x7fffffff            // SCC = Alpha != 0
s_cselect_b32 s[sgprSizesSum+0], s[sgprSizesSum+0], 0 // Set summation dim=0 if Alpha == 0

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 1
/* WGM Calculation */
s_sext_i32_i16 s19, s[sgprWGM]                     // Restore WGM
s_cmp_gt_u32 s19, 1                                // WGM != 0 and 1 ?
s_cbranch_scc1 label_WGMRemap                      // branch if remap
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
v_mov_b32 v60, v0                                  // groA0I_0
v_add_co_u32 v61, vcc_lo, 1, v60                   // groA0I_1 += LSPA

/* global read addresses: tile offsets b */

/* global read addresses: unroll offsets a */
v_mov_b32 v65, v1                                  // groAL_0
v_add_co_u32 v66, vcc_lo, 1024, v65                // groAL_1 + LSCA

/* global read addresses: unroll offsets b */
v_add_co_u32 v68, vcc_lo, 1024, v3                 // groBL_1 + LSCB

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_i32 s80, s[sgprWorkGroup0], 2                // WorkGroup[01] * MT
s_mul_hi_u32 s81, s80, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s80, s80, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mov_b32 s[sgprShadowLimitA+0], s[sgprSizeL]      // Init tensor size
s_mov_b32 s[sgprShadowLimitA+1], 0                 // init tensor size
s_sub_u32 s78, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s79, s[sgprStrideA0I], s78            // stride x (size-1)
s_mul_i32 s78, s[sgprStrideA0I], s78               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s78 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s79 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s80 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s81 // sub tileStart
s_lshr_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadA
s_mul_hi_u32 s79, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s78, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s80, s80, s78                            // accum wg term to tilestart
s_addc_u32 s81, s81, s79                           // accum wg term to tilestart
s_lshr_b64 s[80:81], s[80:81], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s80    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s81   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: block-scale A srd */
s_ff1_i32_b32 s[sgprScaleAGShift], s[sgprScaleAGMinus1]
s_lshr_b32 s[sgprScaleAKMask], s[sgprScaleAGMinus1], 11
s_max_u32 s[sgprScaleAKMask], s[sgprScaleAKMask], 1
s_sub_u32 s[sgprScaleAKMask], s[sgprScaleAKMask], 1
s_mov_b32 s[sgprScaleAInc], 2048
s_lshr_b32 s[sgprScaleAInc], s[sgprScaleAInc], s[sgprScaleAGShift]
s_max_u32 s[sgprScaleAInc], s[sgprScaleAInc], 1
s_lshl_b32 s[sgprScaleAInc], s[sgprScaleAInc], 1
s_mov_b32 s[sgprScaleAKCnt], 0
s_add_u32 s[sgprStrideScaleA], s[sgprSizeL], s[sgprScaleAGMinus1] // SizeL + G
s_sub_u32 s[sgprStrideScaleA], s[sgprStrideScaleA], 1 // SizeL + G-1
s_lshr_b32 s[sgprStrideScaleA], s[sgprStrideScaleA], s[sgprScaleAGShift] // StrideScaleA = ceil(SizeL/32) scale elements
s_mul_i32 s78, s[sgprWorkGroup0], s[sgprStrideScaleA] // scaleA: workgroup * row stride
s_mul_i32 s78, s78, 4                              // scaleA: * MT0 rows, elements -> bytes
s_mul_i32 s79, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s79, s79, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s79, s78             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s78 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD
s_mov_b32 s[sgprScaleAPkMagic+0], 0xf000f
s_mov_b32 s[sgprScaleAPkMagic+1], 0xf000f0

/* global read addresses: block-scale A zero-point srd */
s_mul_i32 s78, s[sgprWorkGroup0], 2                // scaleZeroA: workgroup row origin
s_lshr_b32 s78, s78, 3                             // scaleZeroA: row origin to words
s_lshl_b32 s78, s78, 2                             // scaleZeroA: words to bytes
s_mul_i32 s78, s78, s[sgprStrideScaleA]            // scaleZeroA: * kGroups
s_add_u32 s79, s[sgprSizeI], 7                     // scaleZeroA: SizeI + 7
s_lshr_b32 s79, s79, 3                             // scaleZeroA: ceil(SizeI/8) words
s_lshl_b32 s79, s79, 2                             // scaleZeroA: words to bytes
s_mul_i32 s79, s79, s[sgprStrideScaleA]            // scaleZeroA: total bytes
s_sub_u32 s[sgprSrdScaleZeroA+2], s79, s78         // scaleZeroA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprAddressScaleZeroA+0], s78 // scaleZeroA: SRD base lo
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprAddressScaleZeroA+1], 0 // scaleZeroA: SRD base hi
s_mov_b32 s[sgprSrdScaleZeroA+3], Srd127_96        // scaleZeroA: set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_i32 s80, s[sgprWorkGroup1], 3                // WorkGroup[01] * MT
s_mul_hi_u32 s81, s80, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s80, s80, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mov_b32 s[sgprShadowLimitB+0], s[sgprSizeL]      // Init tensor size
s_mov_b32 s[sgprShadowLimitB+1], 0                 // init tensor size
s_sub_u32 s78, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s79, s[sgprStrideB1J], s78            // stride x (size-1)
s_mul_i32 s78, s[sgprStrideB1J], s78               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s78 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s79 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s80 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s81 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 64 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadB
s_mul_hi_u32 s79, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s78, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s80, s80, s78                            // accum wg term to tilestart
s_addc_u32 s81, s81, s79                           // accum wg term to tilestart
s_lshl_b64 s[80:81], s[80:81], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s80    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s81   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v69, s[sgprStrideA0I], v[60]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[65], v[69+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x20, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v69, s[sgprScaleAGShift], v65        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v60 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v69, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
s_mul_i32 s78, s[sgprWorkGroup0], 2                // scaleZeroA: workgroup row origin
s_and_b32 s78, s78, 7                              // scaleZeroA: origin row within its word
v_add_nc_u32 v70, s78, v60                         // scaleZeroA: row from the word origin
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v70 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+0], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+0], v69, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: word to nibble index
v_and_b32 v69, 7, v70                              // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+0], v69, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+0], 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v69, s[sgprStrideA0I], v[60]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[66], v[69+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x20, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v69, s[sgprScaleAGShift], v66        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v60 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v69, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
s_mul_i32 s78, s[sgprWorkGroup0], 2                // scaleZeroA: workgroup row origin
s_and_b32 s78, s78, 7                              // scaleZeroA: origin row within its word
v_add_nc_u32 v70, s78, v60                         // scaleZeroA: row from the word origin
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v70 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+1], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+1], v69, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: word to nibble index
v_and_b32 v69, 7, v70                              // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+1], v69, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+1], 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
v_mul_lo_u32 v69, s[sgprStrideA0I], v[61]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+2+0], vcc_lo, v[65], v[69+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+2+0], 0x20, v[vgprGlobalReadOffsetA+2+0] // add prepad for pointer shift
v_lshrrev_b32 v69, s[sgprScaleAGShift], v65        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+2], s[sgprStrideScaleA], v61 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+2], v69, v[vgprGlobalReadOffsetScaleA+2] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+2], 1, v[vgprGlobalReadOffsetScaleA+2] // scaleA: elements -> bytes
s_mul_i32 s78, s[sgprWorkGroup0], 2                // scaleZeroA: workgroup row origin
s_and_b32 s78, s78, 7                              // scaleZeroA: origin row within its word
v_add_nc_u32 v70, s78, v61                         // scaleZeroA: row from the word origin
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+2], 3, v70 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+2], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+2], v69, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+2], 3, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: word to nibble index
v_and_b32 v69, 7, v70                              // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+2], v69, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+2], 1, v[vgprGlobalReadOffsetScaleZeroA+2] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+2], 1, v[vgprGlobalReadOffsetA+2] //  (multiple bpe)
v_mul_lo_u32 v69, s[sgprStrideA0I], v[61]          // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+3+0], vcc_lo, v[66], v[69+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+3+0], 0x20, v[vgprGlobalReadOffsetA+3+0] // add prepad for pointer shift
v_lshrrev_b32 v69, s[sgprScaleAGShift], v66        // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+3], s[sgprStrideScaleA], v61 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+3], v69, v[vgprGlobalReadOffsetScaleA+3] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+3], 1, v[vgprGlobalReadOffsetScaleA+3] // scaleA: elements -> bytes
s_mul_i32 s78, s[sgprWorkGroup0], 2                // scaleZeroA: workgroup row origin
s_and_b32 s78, s78, 7                              // scaleZeroA: origin row within its word
v_add_nc_u32 v70, s78, v61                         // scaleZeroA: row from the word origin
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+3], 3, v70 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+3], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+3], v69, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+3], 3, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: word to nibble index
v_and_b32 v69, 7, v70                              // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+3], v69, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+3], 1, v[vgprGlobalReadOffsetScaleZeroA+3] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+3], 1, v[vgprGlobalReadOffsetA+3] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_add_nc_u32 v[vgprGlobalReadOffsetB+0], 0x20, v3  // gROB_0: unroll + prepad
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+1], v[vgprGlobalReadOffsetB+0] // gROB_1: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+2], v[vgprGlobalReadOffsetB+0] // gROB_2: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+3], v[vgprGlobalReadOffsetB+0] // gROB_3: same row and unroll
v_add_nc_u32 v[vgprGlobalReadOffsetB+4], 0x20, v68 // gROB_4: unroll + prepad
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+5], v[vgprGlobalReadOffsetB+4] // gROB_5: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+6], v[vgprGlobalReadOffsetB+4] // gROB_6: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+7], v[vgprGlobalReadOffsetB+4] // gROB_7: same row and unroll
s_add_u32 s78, s[sgprStrideB1J], 0x20              // + prepad
v_add_nc_u32 v[vgprGlobalReadOffsetB+8], s78, v3   // gROB_8: + unroll
v_lshlrev_b32 v[vgprGlobalReadOffsetB+8], 1, v[vgprGlobalReadOffsetB+8] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+9], v[vgprGlobalReadOffsetB+8] // gROB_9: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+10], v[vgprGlobalReadOffsetB+8] // gROB_10: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+11], v[vgprGlobalReadOffsetB+8] // gROB_11: same row and unroll
s_add_u32 s78, s[sgprStrideB1J], 0x20              // + prepad
v_add_nc_u32 v[vgprGlobalReadOffsetB+12], s78, v68 // gROB_12: + unroll
v_lshlrev_b32 v[vgprGlobalReadOffsetB+12], 1, v[vgprGlobalReadOffsetB+12] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+13], v[vgprGlobalReadOffsetB+12] // gROB_13: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+14], v[vgprGlobalReadOffsetB+12] // gROB_14: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+15], v[vgprGlobalReadOffsetB+12] // gROB_15: same row and unroll
s_mul_i32 s78, s[sgprStrideB1J], 2                 // row(2) * stride
s_add_u32 s78, s78, 0x20                           // + prepad
v_add_nc_u32 v[vgprGlobalReadOffsetB+16], s78, v3  // gROB_16: + unroll
v_lshlrev_b32 v[vgprGlobalReadOffsetB+16], 1, v[vgprGlobalReadOffsetB+16] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+17], v[vgprGlobalReadOffsetB+16] // gROB_17: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+18], v[vgprGlobalReadOffsetB+16] // gROB_18: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+19], v[vgprGlobalReadOffsetB+16] // gROB_19: same row and unroll
s_mul_i32 s78, s[sgprStrideB1J], 2                 // row(2) * stride
s_add_u32 s78, s78, 0x20                           // + prepad
v_add_nc_u32 v[vgprGlobalReadOffsetB+20], s78, v68 // gROB_20: + unroll
v_lshlrev_b32 v[vgprGlobalReadOffsetB+20], 1, v[vgprGlobalReadOffsetB+20] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+21], v[vgprGlobalReadOffsetB+20] // gROB_21: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+22], v[vgprGlobalReadOffsetB+20] // gROB_22: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+23], v[vgprGlobalReadOffsetB+20] // gROB_23: same row and unroll
/* ============================================================= */
s_and_b32 s[sgprDot2TailRem], s[sgprSizesSum+0], 0x7ff // K tail: K % DepthU(2048)
v_and_b32 v0, 0x1f, v[vgprSerial]                  // K tail: lane
v_lshlrev_b32 v0, 0x5, v0                          // K tail: lane K block start
s_sub_i32 s78, s[sgprDot2TailRem], 0x0             // K tail: remainder past slot 0's start
v_cmp_gt_i32 s[sgprDot2TailMask+0], s78, v0        // K tail: slot 0 lanes inside K
s_sub_i32 s78, s[sgprDot2TailRem], 0x400           // K tail: remainder past slot 1's start
v_cmp_gt_i32 s[sgprDot2TailMask+1], s78, v0        // K tail: slot 1 lanes inside K

/* global read addresses: increments a */
s_mov_b32 s[sgprGlobalReadIncsA+0], 1024           // incrA = DepthU*Bpe*MI_dim(1)

/* global read addresses: increments b */
s_mov_b32 s[sgprGlobalReadIncsB+0], 4096           // incrB = DepthU*Bpe*MI_dim(1)
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...6) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [60...221) from pool */
v_mov_b32 v[vgprValuC+0], 0                        // initC
v_mov_b32 v[vgprValuC+1], 0                        // initC
v_mov_b32 v[vgprValuC+2], 0                        // initC
v_mov_b32 v[vgprValuC+3], 0                        // initC
v_mov_b32 v[vgprValuC+4], 0                        // initC
v_mov_b32 v[vgprValuC+5], 0                        // initC
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 11 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 2048
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
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
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 0_1_0_0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 0_2_0_0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 0_3_0_0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 1_0_0_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 1_1_0_0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 1_2_0_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 1_3_0_0
buffer_load_b128 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 0_1_1_0
buffer_load_b128 v[vgprG2LB+40:vgprG2LB+40+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 0_2_1_0
buffer_load_b128 v[vgprG2LB+44:vgprG2LB+44+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 0_3_1_0
buffer_load_b128 v[vgprG2LB+48:vgprG2LB+48+3], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 1_0_1_0
buffer_load_b128 v[vgprG2LB+52:vgprG2LB+52+3], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 1_1_1_0
buffer_load_b128 v[vgprG2LB+56:vgprG2LB+56+3], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 1_2_1_0
buffer_load_b128 v[vgprG2LB+60:vgprG2LB+60+3], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 1_3_1_0
buffer_load_b128 v[vgprG2LB+64:vgprG2LB+64+3], v[vgprGlobalReadOffsetB+16], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+68:vgprG2LB+68+3], v[vgprGlobalReadOffsetB+16], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 0_1_2_0
buffer_load_b128 v[vgprG2LB+72:vgprG2LB+72+3], v[vgprGlobalReadOffsetB+16], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 0_2_2_0
buffer_load_b128 v[vgprG2LB+76:vgprG2LB+76+3], v[vgprGlobalReadOffsetB+16], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 0_3_2_0
buffer_load_b128 v[vgprG2LB+80:vgprG2LB+80+3], v[vgprGlobalReadOffsetB+20], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 1_0_2_0
buffer_load_b128 v[vgprG2LB+84:vgprG2LB+84+3], v[vgprGlobalReadOffsetB+20], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 1_1_2_0
buffer_load_b128 v[vgprG2LB+88:vgprG2LB+88+3], v[vgprGlobalReadOffsetB+20], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 1_2_2_0
buffer_load_b128 v[vgprG2LB+92:vgprG2LB+92+3], v[vgprGlobalReadOffsetB+20], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 1_3_2_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_b16 v[vgprG2LScaleA+2], v[vgprGlobalReadOffsetScaleA+2], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 2
buffer_load_d16_b16 v[vgprG2LScaleA+3], v[vgprGlobalReadOffsetScaleA+3], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 3
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v[vgprGlobalReadByteOffsetScaleZeroA+0], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v[vgprGlobalReadByteOffsetScaleZeroA+1], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
buffer_load_d16_u8 v[vgprG2LScaleZeroA+2], v[vgprGlobalReadByteOffsetScaleZeroA+2], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 2
buffer_load_d16_u8 v[vgprG2LScaleZeroA+3], v[vgprGlobalReadByteOffsetScaleZeroA+3], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 3
buffer_load_b128 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 slc dlc // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 slc dlc // G -> Reg 1_0_0_0
buffer_load_b128 v[vgprG2LA+44:vgprG2LA+44+3], v[vgprGlobalReadOffsetA+2], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 slc dlc // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LA+60:vgprG2LA+60+3], v[vgprGlobalReadOffsetA+3], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 slc dlc // G -> Reg 1_0_1_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s78, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s78, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_lshl_b32 s79, s78, 1                             // scaleZeroA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s78 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s78 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s79 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s79 // scaleZeroA limit -= inc

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_waitcnt vmcnt(4)                                 // 5wait for global read B
// Skip barrier: NumThreads=32PGR=0, prior iter done reading lds, LW wait LR, sync

/* prefetch local a */

/* prefetch local b */
/* local read increment a */
/* N/A, lro->2 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
/* local read increment b */
/* N/A, lro->2 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* iter 0 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->4 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->4 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0
s_waitcnt vmcnt(1)                                 // A slot 0 loaded
v_pack_b32_f16 v225, v[vgprG2LScaleA+0], v[vgprG2LScaleA+0] // scaleA: duplicate fp16 scale
v_and_b32 v222, 1, v[vgprGlobalReadOffsetScaleZeroA+0]
v_lshlrev_b32 v222, 2, v222
v_bfe_u32 v222, v[vgprG2LScaleZeroA+0], v222, 4
v_lshl_or_b32 v222, v222, 16, v222
v_lshl_or_b32 v226, v222, 0, 0xe400e400
v_lshl_or_b32 v227, v222, 4, 0xd400d400

/* w4a16: save packed int4 dword 0 */
v_and_or_b32 v228, v[vgprG2LA+12+0], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+12+0], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+12+0]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+0+0], v229, v228, 0x5040100  // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+0+1], v231, v230, 0x5040100  // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+0+2], v229, v228, 0x7060302  // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+0+3], v231, v230, 0x7060302  // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_and_or_b32 v228, v[vgprG2LA+12+1], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+12+1], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+12+1]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+0+4], v229, v228, 0x5040100  // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+0+5], v231, v230, 0x5040100  // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+0+6], v229, v228, 0x7060302  // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+0+7], v231, v230, 0x7060302  // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 2 */
v_and_or_b32 v228, v[vgprG2LA+12+2], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+12+2], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+12+2]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+0+8], v229, v228, 0x5040100  // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+0+9], v231, v230, 0x5040100  // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+0+10], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+0+11], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 3 */
v_and_or_b32 v228, v[vgprG2LA+12+3], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+12+3], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+12+3]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+0+12], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+0+13], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+0+14], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+0+15], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3
v_pack_b32_f16 v225, v[vgprG2LScaleA+2], v[vgprG2LScaleA+2] // scaleA: duplicate fp16 scale
v_and_b32 v222, 1, v[vgprGlobalReadOffsetScaleZeroA+2]
v_lshlrev_b32 v222, 2, v222
v_bfe_u32 v222, v[vgprG2LScaleZeroA+2], v222, 4
v_lshl_or_b32 v222, v222, 16, v222
v_lshl_or_b32 v226, v222, 0, 0xe400e400
v_lshl_or_b32 v227, v222, 4, 0xd400d400

/* w4a16: save packed int4 dword 0 */
v_and_or_b32 v228, v[vgprG2LA+44+0], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+44+0], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+44+0]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+32+0], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+32+1], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+32+2], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+32+3], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_and_or_b32 v228, v[vgprG2LA+44+1], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+44+1], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+44+1]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+32+4], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+32+5], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+32+6], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+32+7], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 2 */
v_and_or_b32 v228, v[vgprG2LA+44+2], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+44+2], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+44+2]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+32+8], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+32+9], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+32+10], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+32+11], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 3 */
v_and_or_b32 v228, v[vgprG2LA+44+3], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+44+3], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+44+3]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+32+12], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+32+13], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+32+14], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+32+15], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+0], v[vgprG2LB+0], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+32], v[vgprG2LB+0], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+0], v[vgprG2LB+32], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+32], v[vgprG2LB+32], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+0], v[vgprG2LB+64], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+32], v[vgprG2LB+64], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 1 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->6 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->6 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+1], v[vgprG2LB+1], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+33], v[vgprG2LB+1], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+1], v[vgprG2LB+33], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+33], v[vgprG2LB+33], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+1], v[vgprG2LB+65], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+33], v[vgprG2LB+65], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 2 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->8 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->8 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+2], v[vgprG2LB+2], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+34], v[vgprG2LB+2], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+2], v[vgprG2LB+34], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+34], v[vgprG2LB+34], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+2], v[vgprG2LB+66], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+34], v[vgprG2LB+66], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 3 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->10 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->10 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+3], v[vgprG2LB+3], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+35], v[vgprG2LB+3], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+3], v[vgprG2LB+35], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+35], v[vgprG2LB+35], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+3], v[vgprG2LB+67], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+35], v[vgprG2LB+67], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 4 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->12 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->12 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+4], v[vgprG2LB+4], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+36], v[vgprG2LB+4], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+4], v[vgprG2LB+36], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+36], v[vgprG2LB+36], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+4], v[vgprG2LB+68], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+36], v[vgprG2LB+68], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 5 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->14 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->14 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+5], v[vgprG2LB+5], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+37], v[vgprG2LB+5], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+5], v[vgprG2LB+37], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+37], v[vgprG2LB+37], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+5], v[vgprG2LB+69], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+37], v[vgprG2LB+69], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 6 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->16 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->16 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+6], v[vgprG2LB+6], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+38], v[vgprG2LB+6], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+6], v[vgprG2LB+38], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+38], v[vgprG2LB+38], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+6], v[vgprG2LB+70], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+38], v[vgprG2LB+70], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 7 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->18 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->18 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+7], v[vgprG2LB+7], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+39], v[vgprG2LB+7], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+7], v[vgprG2LB+39], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+39], v[vgprG2LB+39], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+7], v[vgprG2LB+71], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+39], v[vgprG2LB+71], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 8 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->20 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->20 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+8], v[vgprG2LB+8], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+40], v[vgprG2LB+8], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+8], v[vgprG2LB+40], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+40], v[vgprG2LB+40], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+8], v[vgprG2LB+72], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+40], v[vgprG2LB+72], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 9 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->22 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->22 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+9], v[vgprG2LB+9], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+41], v[vgprG2LB+9], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+9], v[vgprG2LB+41], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+41], v[vgprG2LB+41], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+9], v[vgprG2LB+73], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+41], v[vgprG2LB+73], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 10 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->24 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->24 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+10], v[vgprG2LB+10], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+42], v[vgprG2LB+10], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+10], v[vgprG2LB+42], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+42], v[vgprG2LB+42], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+10], v[vgprG2LB+74], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+42], v[vgprG2LB+74], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 11 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->26 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->26 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+11], v[vgprG2LB+11], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+43], v[vgprG2LB+11], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+11], v[vgprG2LB+43], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+43], v[vgprG2LB+43], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+11], v[vgprG2LB+75], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+43], v[vgprG2LB+75], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 12 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->28 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->28 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+12], v[vgprG2LB+12], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+44], v[vgprG2LB+12], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+12], v[vgprG2LB+44], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+44], v[vgprG2LB+44], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+12], v[vgprG2LB+76], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+44], v[vgprG2LB+76], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 13 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->30 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->30 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+13], v[vgprG2LB+13], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+45], v[vgprG2LB+13], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+13], v[vgprG2LB+45], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+45], v[vgprG2LB+45], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+13], v[vgprG2LB+77], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+45], v[vgprG2LB+77], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 14 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1024 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1024 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+14], v[vgprG2LB+14], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+46], v[vgprG2LB+14], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+14], v[vgprG2LB+46], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+46], v[vgprG2LB+46], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+14], v[vgprG2LB+78], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+46], v[vgprG2LB+78], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 15 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1026 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1026 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+15], v[vgprG2LB+15], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+47], v[vgprG2LB+15], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+15], v[vgprG2LB+47], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+47], v[vgprG2LB+47], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+15], v[vgprG2LB+79], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+47], v[vgprG2LB+79], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 16 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1028 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1028 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
s_waitcnt vmcnt(0)                                 // A slot 1 loaded
v_pack_b32_f16 v225, v[vgprG2LScaleA+1], v[vgprG2LScaleA+1] // scaleA: duplicate fp16 scale
v_and_b32 v222, 1, v[vgprGlobalReadOffsetScaleZeroA+1]
v_lshlrev_b32 v222, 2, v222
v_bfe_u32 v222, v[vgprG2LScaleZeroA+1], v222, 4
v_lshl_or_b32 v222, v222, 16, v222
v_lshl_or_b32 v226, v222, 0, 0xe400e400
v_lshl_or_b32 v227, v222, 4, 0xd400d400

/* w4a16: save packed int4 dword 0 */
v_and_or_b32 v228, v[vgprG2LA+28+0], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+28+0], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+28+0]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+16+0], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+16+1], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+16+2], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+16+3], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_and_or_b32 v228, v[vgprG2LA+28+1], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+28+1], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+28+1]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+16+4], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+16+5], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+16+6], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+16+7], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 2 */
v_and_or_b32 v228, v[vgprG2LA+28+2], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+28+2], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+28+2]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+16+8], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+16+9], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+16+10], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+16+11], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 3 */
v_and_or_b32 v228, v[vgprG2LA+28+3], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+28+3], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+28+3]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+16+12], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+16+13], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+16+14], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+16+15], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3
v_pack_b32_f16 v225, v[vgprG2LScaleA+3], v[vgprG2LScaleA+3] // scaleA: duplicate fp16 scale
v_and_b32 v222, 1, v[vgprGlobalReadOffsetScaleZeroA+3]
v_lshlrev_b32 v222, 2, v222
v_bfe_u32 v222, v[vgprG2LScaleZeroA+3], v222, 4
v_lshl_or_b32 v222, v222, 16, v222
v_lshl_or_b32 v226, v222, 0, 0xe400e400
v_lshl_or_b32 v227, v222, 4, 0xd400d400

/* w4a16: save packed int4 dword 0 */
v_and_or_b32 v228, v[vgprG2LA+60+0], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+60+0], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+60+0]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+48+0], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+48+1], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+48+2], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+48+3], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 1 */
v_and_or_b32 v228, v[vgprG2LA+60+1], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+60+1], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+60+1]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+48+4], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+48+5], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+48+6], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+48+7], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 2 */
v_and_or_b32 v228, v[vgprG2LA+60+2], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+60+2], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+60+2]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+48+8], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+48+9], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+48+10], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+48+11], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3

/* w4a16: save packed int4 dword 3 */
v_and_or_b32 v228, v[vgprG2LA+60+3], s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #0 and #4 -> two fp16 holding 1024+q
v_and_or_b32 v229, v[vgprG2LA+60+3], s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #1 and #5 -> two fp16 holding 64+q
v_lshrrev_b32 v224, 0x8, v[vgprG2LA+60+3]          // w4a16: next two nibbles down to bits 0-7
v_and_or_b32 v230, v224, s[sgprScaleAPkMagic+0], 0x64006400 // w4a16: int4 #2 and #6 -> two fp16 holding 1024+q
v_and_or_b32 v231, v224, s[sgprScaleAPkMagic+1], 0x54005400 // w4a16: int4 #3 and #7 -> two fp16 holding 64+q
v_pk_add_f16 v228, v228, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v229, v229, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_add_f16 v230, v230, v226                      // w4a16: q - z (exact: both sides are integers near 1024)
v_pk_add_f16 v231, v231, v227                      // w4a16: q - z (exact: both sides are integers near 64)
v_pk_fma_f16 v228, v228, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v229, v229, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v230, v230, v225, 0                   // w4a16: (q - z)*s, packed
v_pk_fma_f16 v231, v231, v225, 0                   // w4a16: (q - z)*s, packed
v_perm_b32 v[vgprG2LA+48+12], v229, v228, 0x5040100 // w4a16: restore sequential fp16 pair 0
v_perm_b32 v[vgprG2LA+48+13], v231, v230, 0x5040100 // w4a16: restore sequential fp16 pair 1
v_perm_b32 v[vgprG2LA+48+14], v229, v228, 0x7060302 // w4a16: restore sequential fp16 pair 2
v_perm_b32 v[vgprG2LA+48+15], v231, v230, 0x7060302 // w4a16: restore sequential fp16 pair 3
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+16], v[vgprG2LB+16], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+48], v[vgprG2LB+16], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+16], v[vgprG2LB+48], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+48], v[vgprG2LB+48], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+16], v[vgprG2LB+80], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+48], v[vgprG2LB+80], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 17 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1030 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1030 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+17], v[vgprG2LB+17], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+49], v[vgprG2LB+17], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+17], v[vgprG2LB+49], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+49], v[vgprG2LB+49], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+17], v[vgprG2LB+81], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+49], v[vgprG2LB+81], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 18 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1032 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1032 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+18], v[vgprG2LB+18], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+50], v[vgprG2LB+18], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+18], v[vgprG2LB+50], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+50], v[vgprG2LB+50], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+18], v[vgprG2LB+82], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+50], v[vgprG2LB+82], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 19 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1034 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1034 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+19], v[vgprG2LB+19], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+51], v[vgprG2LB+19], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+19], v[vgprG2LB+51], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+51], v[vgprG2LB+51], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+19], v[vgprG2LB+83], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+51], v[vgprG2LB+83], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 20 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1036 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1036 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+20], v[vgprG2LB+20], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+52], v[vgprG2LB+20], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+20], v[vgprG2LB+52], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+52], v[vgprG2LB+52], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+20], v[vgprG2LB+84], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+52], v[vgprG2LB+84], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 21 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1038 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1038 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+21], v[vgprG2LB+21], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+53], v[vgprG2LB+21], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+21], v[vgprG2LB+53], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+53], v[vgprG2LB+53], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+21], v[vgprG2LB+85], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+53], v[vgprG2LB+85], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 22 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1040 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1040 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+22], v[vgprG2LB+22], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+54], v[vgprG2LB+22], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+22], v[vgprG2LB+54], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+54], v[vgprG2LB+54], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+22], v[vgprG2LB+86], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+54], v[vgprG2LB+86], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 23 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1042 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1042 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+23], v[vgprG2LB+23], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+55], v[vgprG2LB+23], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+23], v[vgprG2LB+55], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+55], v[vgprG2LB+55], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+23], v[vgprG2LB+87], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+55], v[vgprG2LB+87], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 24 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1044 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1044 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+24], v[vgprG2LB+24], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+56], v[vgprG2LB+24], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+24], v[vgprG2LB+56], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+56], v[vgprG2LB+56], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+24], v[vgprG2LB+88], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+56], v[vgprG2LB+88], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 25 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1046 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1046 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+25], v[vgprG2LB+25], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+57], v[vgprG2LB+25], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+25], v[vgprG2LB+57], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+57], v[vgprG2LB+57], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+25], v[vgprG2LB+89], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+57], v[vgprG2LB+89], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 26 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1048 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1048 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+26], v[vgprG2LB+26], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+58], v[vgprG2LB+26], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+26], v[vgprG2LB+58], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+58], v[vgprG2LB+58], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+26], v[vgprG2LB+90], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+58], v[vgprG2LB+90], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 27 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1050 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1050 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+27], v[vgprG2LB+27], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+59], v[vgprG2LB+27], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+27], v[vgprG2LB+59], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+59], v[vgprG2LB+59], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+27], v[vgprG2LB+91], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+59], v[vgprG2LB+91], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 28 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1052 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1052 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+28], v[vgprG2LB+28], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+60], v[vgprG2LB+28], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+28], v[vgprG2LB+60], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+60], v[vgprG2LB+60], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+28], v[vgprG2LB+92], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+60], v[vgprG2LB+92], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 29 */

/* local read a */

/* local read b */

/* local read increment a */
/* N/A, lro->1054 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */

/* local read increment b */
/* N/A, lro->1054 */
/* localReadDoCntA 0 localReadDoCntMXSA 0 localReadDoCntB 0 localReadDoCntMXSB 0 localReadDoCntM 0 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+29], v[vgprG2LB+29], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+61], v[vgprG2LB+29], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+29], v[vgprG2LB+61], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+61], v[vgprG2LB+61], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+29], v[vgprG2LB+93], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+61], v[vgprG2LB+93], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 30 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */

/* local read b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+30], v[vgprG2LB+30], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+62], v[vgprG2LB+30], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+30], v[vgprG2LB+62], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+62], v[vgprG2LB+62], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+30], v[vgprG2LB+94], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+62], v[vgprG2LB+94], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/* iter 31 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_F16_HPA_DOT2
v_dot2_f32_f16 v[vgprValuC+0+0], v[vgprG2LA+31], v[vgprG2LB+31], v[vgprValuC+0+0] // ValuC[0] iui=0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_f16 v[vgprValuC+1+0], v[vgprG2LA+63], v[vgprG2LB+31], v[vgprValuC+1+0] // ValuC[1] iui=0
v_dot2_f32_f16 v[vgprValuC+0+2], v[vgprG2LA+31], v[vgprG2LB+63], v[vgprValuC+0+2] // ValuC[2] iui=0
v_dot2_f32_f16 v[vgprValuC+1+2], v[vgprG2LA+63], v[vgprG2LB+63], v[vgprValuC+1+2] // ValuC[3] iui=0
v_dot2_f32_f16 v[vgprValuC+0+4], v[vgprG2LA+31], v[vgprG2LB+95], v[vgprValuC+0+4] // ValuC[4] iui=0
v_dot2_f32_f16 v[vgprValuC+1+4], v[vgprG2LA+63], v[vgprG2LB+95], v[vgprValuC+1+4] // ValuC[5] iui=0
s_setprio 0                                        // Reset priority after macs


/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:
s_cmp_eq_u32 s[sgprDot2TailRem], 0                 // K tail: none, or done
s_cbranch_scc1 label_Dot2TailDone                  // K tail: skip
s_mov_b32 s[sgprDot2TailRem], 0                    // K tail: run once
v_cndmask_b32 v[vgprGlobalReadOffsetA+0], BufferOOB, v[vgprGlobalReadOffsetA+0], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetA+1], BufferOOB, v[vgprGlobalReadOffsetA+1], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetA+2], BufferOOB, v[vgprGlobalReadOffsetA+2], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetA+3], BufferOOB, v[vgprGlobalReadOffsetA+3], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetScaleA+0], BufferOOB, v[vgprGlobalReadOffsetScaleA+0], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetScaleA+1], BufferOOB, v[vgprGlobalReadOffsetScaleA+1], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetScaleA+2], BufferOOB, v[vgprGlobalReadOffsetScaleA+2], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetScaleA+3], BufferOOB, v[vgprGlobalReadOffsetScaleA+3], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+0], BufferOOB, v[vgprGlobalReadOffsetB+0], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+1], BufferOOB, v[vgprGlobalReadOffsetB+1], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+2], BufferOOB, v[vgprGlobalReadOffsetB+2], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+3], BufferOOB, v[vgprGlobalReadOffsetB+3], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+4], BufferOOB, v[vgprGlobalReadOffsetB+4], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+5], BufferOOB, v[vgprGlobalReadOffsetB+5], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+6], BufferOOB, v[vgprGlobalReadOffsetB+6], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+7], BufferOOB, v[vgprGlobalReadOffsetB+7], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+8], BufferOOB, v[vgprGlobalReadOffsetB+8], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+9], BufferOOB, v[vgprGlobalReadOffsetB+9], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+10], BufferOOB, v[vgprGlobalReadOffsetB+10], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+11], BufferOOB, v[vgprGlobalReadOffsetB+11], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+12], BufferOOB, v[vgprGlobalReadOffsetB+12], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+13], BufferOOB, v[vgprGlobalReadOffsetB+13], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+14], BufferOOB, v[vgprGlobalReadOffsetB+14], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+15], BufferOOB, v[vgprGlobalReadOffsetB+15], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+16], BufferOOB, v[vgprGlobalReadOffsetB+16], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+17], BufferOOB, v[vgprGlobalReadOffsetB+17], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+18], BufferOOB, v[vgprGlobalReadOffsetB+18], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+19], BufferOOB, v[vgprGlobalReadOffsetB+19], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+20], BufferOOB, v[vgprGlobalReadOffsetB+20], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+21], BufferOOB, v[vgprGlobalReadOffsetB+21], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+22], BufferOOB, v[vgprGlobalReadOffsetB+22], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+23], BufferOOB, v[vgprGlobalReadOffsetB+23], s[sgprDot2TailMask+1] // K tail: slot 1, lanes past K read zeros
s_mov_b32 s[sgprLoopCounterL], 1                   // K tail: one more iteration
s_branch label_LoopBeginL                          // K tail: masked iteration
label_Dot2TailDone:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/* Tail: add ValuA/B vgpr buffer [60...221) to pool */

/* Tail: add address/G2L vgpr [221...221) to pool */
label_Summation_End_0:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStrideScaleA, UNDEF
.set sgprAddressScaleA, UNDEF
.set sgprAddressScaleB, UNDEF
.set sgprSrdScaleA, UNDEF
.set sgprScaleAGMinus1, UNDEF
.set sgprScaleAGShift, UNDEF
.set sgprScaleAKMask, UNDEF
.set sgprScaleAInc, UNDEF
.set sgprScaleAKCnt, UNDEF
.set sgprScaleAPkMagic, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprAddressScaleZeroA, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprSrdScaleZeroA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitB, UNDEF
/* load store sgprs */
.set sgprSrdC, 28
.set sgprSrdD, 24
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cselect_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0, s[sgprAddressD+0:sgprAddressD+0+1] // init SRD base address
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cselect_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0, s[sgprAddressC+0:sgprAddressC+0+1] // init SRD base address
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mul_i32 s46, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s45, s46, s[sgprStrideC1J]            // ScaleC s46 by Stride
s_mul_i32 s44, s46, s[sgprStrideC1J]               // ScaleC s46 by Stride
s_lshl_b64 s[44:45], s[44:45], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // add hi to SRD
s_mul_hi_u32 s45, s46, s[sgprStrideD1J]            // ScaleD s46 by Stride
s_mul_i32 s44, s46, s[sgprStrideD1J]               // ScaleD s46 by Stride
s_lshl_b64 s[44:45], s[44:45], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // add hi to SRD

s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s45, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s44, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[44:45], s[44:45], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // add hi to SRD
label_GeneralBatchedGemmLoadC_End:  /// End of label GeneralBatchedGemmLoadC
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for Strided Batched GEMM
s_mul_hi_u32 s45, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s44, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[44:45], s[44:45], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // add hi to SRD
label_GeneralBatchedGemmLoadD_End:  /// End of label GeneralBatchedGemmLoadD

.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF

/******************************************/
/* WaveSplitK Reduction                   */
/******************************************/
s_nop 0                                            // 1 wait states
s_nop 0                                            // 1 wait states
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v[vgprValuC+0] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v[vgprValuC+1] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v[vgprValuC+2] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v[vgprValuC+3] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v[vgprValuC+4] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v[vgprValuC+5] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v[vgprValuC+0] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v[vgprValuC+1] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v[vgprValuC+2] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v[vgprValuC+3] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v[vgprValuC+4] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v[vgprValuC+5] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v[vgprValuC+0] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v[vgprValuC+1] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v[vgprValuC+2] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v[vgprValuC+3] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v[vgprValuC+4] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v[vgprValuC+5] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v[vgprValuC+0] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v[vgprValuC+1] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v[vgprValuC+2] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v[vgprValuC+3] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v[vgprValuC+4] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v[vgprValuC+5] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_xor_b32 v60, 16, v[vgprSerial]                   // lane ^ 16
v_lshlrev_b32 v60, 2, v60                          // bpermute byte address
ds_bpermute_b32 v61, v60, v[vgprValuC+0]
ds_bpermute_b32 v62, v60, v[vgprValuC+1]
ds_bpermute_b32 v63, v60, v[vgprValuC+2]
ds_bpermute_b32 v64, v60, v[vgprValuC+3]
ds_bpermute_b32 v65, v60, v[vgprValuC+4]
ds_bpermute_b32 v66, v60, v[vgprValuC+5]
s_waitcnt lgkmcnt(0)                               // wait for bpermute
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v61      // Reduce 32 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v62      // Reduce 32 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v63      // Reduce 32 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v64      // Reduce 32 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v65      // Reduce 32 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v66      // Reduce 32 elements


/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_mov_b32 v62, 0                                   // rowStart vgpr
v_mov_b32 v63, 0                                   // rowStart vgpr

s_mul_i32 s44, 0x2, s[sgprWorkGroup0]              // s44 = wg0*MT0
v_mov_b32 v60, s44                                 // coord0 = wg0*MT0
s_mul_i32 s46, 0x3, s[sgprWorkGroup1]              // <- wg1*MT1
v_mov_b32 v61, s46                                 // coord1 = wg1*MT1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
label_GSU:
s_cmpk_eq_u32 s[sgprBeta], 0                       // Beta == 0
s_cbranch_scc0 label_GW_B1_GSU1                    // Branch if Beta is not zero

label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 2 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s44, 1, s[sgprSizeI]                     // s44 = s[sgprSizeI] % 2
s_add_u32 s45, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s45                // wg0 >= nwg0-1 ?
s_cselect_b32 s44, s44, 0                          // set rem
s_cmpk_gt_u32 s44, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 3 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s47, s[sgprSizeJ], -1431655765        // STATIC_DIV: divisor=3
s_mul_i32 s46, s[sgprSizeJ], -1431655765           // tmp = dividend * magic
s_lshr_b64 s[46:47], s[46:47], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s45, s46                                 // quotient
s_mul_i32 s46, s45, 3                              // quotient*divisor
s_sub_u32 s44, s[sgprSizeJ], s46                   // rReg = dividend - quotient*divisor
s_add_u32 s45, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s45                // wg1 >= nwg1-1
s_cselect_b32 s44, s44, 0                          // set rem
s_cmpk_gt_u32 s44, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=6 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v67, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_lshl_u32 v72, v63, v60, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=60, coord0Vgpr=60 (multiple bpe)
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v72, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v0.l, v[vgprValuC+0]                 // convert C to fp16
buffer_store_b16 v0, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v1.l, v[vgprValuC+1]                 // convert C to fp16
buffer_store_b16 v1, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
v_cvt_f16_f32 v2.l, v[vgprValuC+2]                 // convert C to fp16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v2, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v3.l, v[vgprValuC+3]                 // convert C to fp16
buffer_store_b16 v3, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
v_cvt_f16_f32 v4.l, v[vgprValuC+4]                 // convert C to fp16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v4, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v5.l, v[vgprValuC+5]                 // convert C to fp16
buffer_store_b16 v5, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=182 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v67, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v72, v63, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v67, v72, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v72, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v73, v63, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v67, v73, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v73, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v74, v63, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v74, v67, v74, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v74, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v75, v63, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v67, v75, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v75, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v76, v63, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v76, v67, v76, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v76, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v77, v63, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v67, v77, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v77, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v0.l, v[vgprValuC+0]                 // convert C to fp16
buffer_store_b16 v0, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v1.l, v[vgprValuC+1]                 // convert C to fp16
buffer_store_b16 v1, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v2.l, v[vgprValuC+2]                 // convert C to fp16
buffer_store_b16 v2, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v3.l, v[vgprValuC+3]                 // convert C to fp16
buffer_store_b16 v3, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v4.l, v[vgprValuC+4]                 // convert C to fp16
buffer_store_b16 v4, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f16_f32 v5.l, v[vgprValuC+5]                 // convert C to fp16
buffer_store_b16 v5, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_branch label_GW_End                              // jump to end
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:

/* Edge/NonEdge store path check (M): Size % 2 > 0 -> Edge store; else -> NonEdge store */
s_and_b32 s44, 1, s[sgprSizeI]                     // s44 = s[sgprSizeI] % 2
s_add_u32 s45, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s45                // wg0 >= nwg0-1 ?
s_cselect_b32 s44, s44, 0                          // set rem
s_cmpk_gt_u32 s44, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else       // jump if edges required

/* Edge/NonEdge store path check (N (isSize1)): Size % 3 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s47, s[sgprSizeJ], -1431655765        // STATIC_DIV: divisor=3
s_mul_i32 s46, s[sgprSizeJ], -1431655765           // tmp = dividend * magic
s_lshr_b64 s[46:47], s[46:47], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s45, s46                                 // quotient
s_mul_i32 s46, s45, 3                              // quotient*divisor
s_sub_u32 s44, s[sgprSizeJ], s46                   // rReg = dividend - quotient*divisor
s_add_u32 s45, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s45                // wg1 >= nwg1-1
s_cselect_b32 s44, s44, 0                          // set rem
s_cmpk_gt_u32 s44, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=180 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v67, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v73, v62, v60, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=60, coord0Vgpr=60 (multiple bpe)
buffer_load_d16_b16 v74, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v75, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:2 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_d16_b16 v76, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_d16_b16 v77, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:2 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_d16_b16 v78, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_d16_b16 v79, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:2 // load C
v_add_lshl_u32 v72, v63, v60, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=60, coord0Vgpr=60 (multiple bpe)
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v72, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 6 - 1 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+0], s[sgprBeta], v74, v[vgprValuC+0] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v0.l, v[vgprValuC+0]                 // convert C to fp16
buffer_store_b16 v0, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 6 - 2 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+1], s[sgprBeta], v75, v[vgprValuC+1] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v1.l, v[vgprValuC+1]                 // convert C to fp16
buffer_store_b16 v1, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 6 - 3 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+2], s[sgprBeta], v76, v[vgprValuC+2] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v2.l, v[vgprValuC+2]                 // convert C to fp16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v2, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 6 - 4 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+3], s[sgprBeta], v77, v[vgprValuC+3] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v3.l, v[vgprValuC+3]                 // convert C to fp16
buffer_store_b16 v3, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 6 - 5 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+4], s[sgprBeta], v78, v[vgprValuC+4] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v4.l, v[vgprValuC+4]                 // convert C to fp16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v4, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 6 - 6 (beta) (interleaved)
v_fma_mix_f32 v[vgprValuC+5], s[sgprBeta], v79, v[vgprValuC+5] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v5.l, v[vgprValuC+5]                 // convert C to fp16
buffer_store_b16 v5, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
s_branch label_GW_End                              // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=90 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v67, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v73, v62, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v67, v73, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v72, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v73, v63, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v67, v73, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v73, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v75, v62, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v67, v75, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v74, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v75, v63, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v67, v75, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v75, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v77, v62, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v67, v77, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v76, v77, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v77, v63, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v67, v77, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v77, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v79, v62, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v79, v67, v79, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v78, v79, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v79, v63, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v79, v67, v79, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v79, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v61, vcc_lo, v61, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v62, v62, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v63, v63, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v81, v62, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v67, v81, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v80, v81, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v81, v63, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v67, v81, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v81, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v64, vcc_lo, v60, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v64, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v61, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v83, v62, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v67, v83, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v82, v83, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v83, v63, v64, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v67, v83, s46                   // LDD clip if OOB. offset
v_and_b32 v66, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v66                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v83, v67                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_fma_mix_f32 v[vgprValuC+0], s[sgprBeta], v72, v[vgprValuC+0] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v0.l, v[vgprValuC+0]                 // convert C to fp16
buffer_store_b16 v0, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+1], s[sgprBeta], v74, v[vgprValuC+1] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v1.l, v[vgprValuC+1]                 // convert C to fp16
buffer_store_b16 v1, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+2], s[sgprBeta], v76, v[vgprValuC+2] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v2.l, v[vgprValuC+2]                 // convert C to fp16
buffer_store_b16 v2, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+3], s[sgprBeta], v78, v[vgprValuC+3] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v3.l, v[vgprValuC+3]                 // convert C to fp16
buffer_store_b16 v3, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+4], s[sgprBeta], v80, v[vgprValuC+4] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v4.l, v[vgprValuC+4]                 // convert C to fp16
buffer_store_b16 v4, v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_fma_mix_f32 v[vgprValuC+5], s[sgprBeta], v82, v[vgprValuC+5] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_cvt_f16_f32 v5.l, v[vgprValuC+5]                 // convert C to fp16
buffer_store_b16 v5, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_branch label_GW_End                              // jump to end
label_GW_End:
label_KernelEnd:
s_endpgm                                           // Kernel End
.set sgprKernArgAddress, 0
label_HBMArgs:

/* Load address of kernel arguments */
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)                               // wait for args to load
s_branch label_LoadArgsEnd
.set sgprWGM, 11
.set sgprWorkGroup0, 2
label_WGMXCC:
s_lshr_b32 s73, s[sgprWGM], 0x16                   // Get CU_Count
/* only remap WGs in the range */
s_lshr_b32 s69, s63, s72
s_lshl_b32 s69, s69, s72
s_cmp_ge_u32 s[sgprWorkGroup0], s69
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s73, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s69, s[sgprWorkGroup0], s72
s_bfm_b32 s70, s72, 0
s_and_b32 s70, s[sgprWorkGroup0], s70
s_lshr_b32 s71, s63, s72
s_mul_i32 s70, s70, s71
s_add_u32 s[sgprWorkGroup0], s69, s70
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[60:61], s73                        // s69 = s[sgprWorkGroup0] / s73
v_rcp_f64 v[60:61], v[60:61]                       // s69 = s[sgprWorkGroup0] / s73
v_cvt_f64_u32 v[62:63], s[sgprWorkGroup0]          // s69 = s[sgprWorkGroup0] / s73
v_mul_f64 v[60:61], v[60:61], v[62:63]             // s69 = s[sgprWorkGroup0] / s73
v_cvt_u32_f64 v60, v[60:61]                        // s69 = s[sgprWorkGroup0] / s73
v_mul_lo_u32 v61, v60, s73                         // s69 = s[sgprWorkGroup0] / s73
v_sub_nc_u32 v62, s[sgprWorkGroup0], v61           // s69 = s[sgprWorkGroup0] / s73
v_cmp_ge_u32 vcc_lo, v62, s73                      // s69 = s[sgprWorkGroup0] / s73
s_mov_b32 exec_lo, vcc_lo                          // s69 = s[sgprWorkGroup0] / s73
v_add_nc_u32 v60, v60, 1                           // s69 = s[sgprWorkGroup0] / s73
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v61, v60, s73                         // s69 = s[sgprWorkGroup0] / s73
v_sub_nc_u32 v62, s[sgprWorkGroup0], v61           // s69 = s[sgprWorkGroup0] / s73
v_readfirstlane_b32 s69, v60                       // quotient
v_readfirstlane_b32 s70, v62                       // remainder
s_mul_i32 s69, s69, s73
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s70, s70, s72
/* temp0 = temp0 + temp1 */
s_add_u32 s69, s69, s70
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[60:61], s73                        // s70 = s63 / s73
v_rcp_f64 v[60:61], v[60:61]                       // s70 = s63 / s73
v_cvt_f64_u32 v[62:63], s63                        // s70 = s63 / s73
v_mul_f64 v[60:61], v[60:61], v[62:63]             // s70 = s63 / s73
v_cvt_u32_f64 v60, v[60:61]                        // s70 = s63 / s73
v_mul_lo_u32 v61, v60, s73                         // s70 = s63 / s73
v_sub_nc_u32 v62, s63, v61                         // s70 = s63 / s73
v_cmp_ge_u32 vcc_lo, v62, s73                      // s70 = s63 / s73
s_mov_b32 exec_lo, vcc_lo                          // s70 = s63 / s73
v_add_nc_u32 v60, v60, 1                           // s70 = s63 / s73
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s70, v60                       // quotient
s_mul_i32 s70, s70, s73
s_sub_u32 s71, s63, s70
s_cmp_gt_u32 s[sgprWorkGroup0], s70
s_cselect_b32 s70, s71, s73
s_lshr_b32 s70, s70, s72
s_bfm_b32 s71, s72, 0
s_and_b32 s71, s[sgprWorkGroup0], s71
s_mul_i32 s70, s70, s71
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s69, s70
s_branch label_skip_WGMXCC
.set sgprNumWorkGroups0, 17
.set sgprNumWorkGroups1, 18
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
label_RemapWG:
s_mul_i32 s62, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_cmp_lt_u32 s[sgprWorkGroup0], s62                // wg2 == 0?
s_cbranch_scc1 label_RemapWG2Done
v_cvt_f32_u32 v60, s62                             // s62 = s[sgprWorkGroup0] / s62
v_rcp_iflag_f32 v60, v60                           // s62 = s[sgprWorkGroup0] / s62
v_mov_b32 v61, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v60, v61, v60                            // 2^32 / divisor
v_cvt_u32_f32 v60, v60                             // s62 = s[sgprWorkGroup0] / s62
v_mul_lo_u32 v61, v60, s62                         // refine 2^32 / divisor
v_sub_nc_u32 v61, 0, v61                           // refine 2^32 / divisor
v_mul_hi_u32 v61, v60, v61                         // refine 2^32 / divisor
v_add_nc_u32 v60, v60, v61                         // refine 2^32 / divisor
v_mul_hi_u32 v60, s[sgprWorkGroup0], v60           // s62 = s[sgprWorkGroup0] / s62
v_mul_lo_u32 v61, v60, s62                         // s62 = s[sgprWorkGroup0] / s62
v_sub_nc_u32 v61, s[sgprWorkGroup0], v61           // s62 = s[sgprWorkGroup0] / s62
v_cmp_ge_u32 vcc_lo, v61, s62                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v60, 1, v60                           // quotient + 1
v_sub_nc_u32 v61, v61, s62                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v61, s62                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v60, 1, v60                           // quotient + 1
v_sub_nc_u32 v61, v61, s62                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s62, v60                       // quotient
s_mov_b32 s[sgprWorkGroup2], s62
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s62, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s62, s62, s[sgprWorkGroup2]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s62
label_RemapWG2Done:
s_cmp_lt_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // wg1 == 0?
s_cbranch_scc1 label_RemapWG1Done
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v60, s[sgprNumWorkGroups0]           // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v60, v60                           // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mov_b32 v61, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v60, v61, v60                            // 2^32 / divisor
v_cvt_u32_f32 v60, v60                             // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_lo_u32 v61, v60, s[sgprNumWorkGroups0]       // refine 2^32 / divisor
v_sub_nc_u32 v61, 0, v61                           // refine 2^32 / divisor
v_mul_hi_u32 v61, v60, v61                         // refine 2^32 / divisor
v_add_nc_u32 v60, v60, v61                         // refine 2^32 / divisor
v_mul_hi_u32 v60, s[sgprWorkGroup0], v60           // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_lo_u32 v61, v60, s[sgprNumWorkGroups0]       // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v61, s[sgprWorkGroup0], v61           // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_ge_u32 vcc_lo, v61, s[sgprNumWorkGroups0]    // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v60, 1, v60                           // quotient + 1
v_sub_nc_u32 v61, v61, s[sgprNumWorkGroups0]       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v61, s[sgprNumWorkGroups0]    // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v60, 1, v60                           // quotient + 1
v_sub_nc_u32 v61, v61, s[sgprNumWorkGroups0]       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s62, v60                       // quotient
s_mov_b32 s[sgprWorkGroup1], s62
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s62, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s62
label_RemapWG1Done:
s_branch label_RemapWGDone
.set sgprNumWorkGroups0, 17
.set sgprNumWorkGroups1, 18
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
label_RemapWG_1:
s_mul_i32 s62, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_cmp_lt_u32 s[sgprWorkGroup0], s62                // wg2 == 0?
s_cbranch_scc1 label_RemapWG2Done_1
v_cvt_f32_u32 v60, s62                             // s62 = s[sgprWorkGroup0] / s62
v_rcp_iflag_f32 v60, v60                           // s62 = s[sgprWorkGroup0] / s62
v_mov_b32 v61, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v60, v61, v60                            // 2^32 / divisor
v_cvt_u32_f32 v60, v60                             // s62 = s[sgprWorkGroup0] / s62
v_mul_lo_u32 v61, v60, s62                         // refine 2^32 / divisor
v_sub_nc_u32 v61, 0, v61                           // refine 2^32 / divisor
v_mul_hi_u32 v61, v60, v61                         // refine 2^32 / divisor
v_add_nc_u32 v60, v60, v61                         // refine 2^32 / divisor
v_mul_hi_u32 v60, s[sgprWorkGroup0], v60           // s62 = s[sgprWorkGroup0] / s62
v_mul_lo_u32 v61, v60, s62                         // s62 = s[sgprWorkGroup0] / s62
v_sub_nc_u32 v61, s[sgprWorkGroup0], v61           // s62 = s[sgprWorkGroup0] / s62
v_cmp_ge_u32 vcc_lo, v61, s62                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v60, 1, v60                           // quotient + 1
v_sub_nc_u32 v61, v61, s62                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v61, s62                      // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v60, 1, v60                           // quotient + 1
v_sub_nc_u32 v61, v61, s62                         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s62, v60                       // quotient
s_mov_b32 s[sgprWorkGroup2], s62
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s62, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s62, s62, s[sgprWorkGroup2]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s62
label_RemapWG2Done_1:
s_cmp_lt_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // wg1 == 0?
s_cbranch_scc1 label_RemapWG1Done_1
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v60, s[sgprNumWorkGroups0]           // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v60, v60                           // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mov_b32 v61, 0x4f7ffffe                          // 2^32 - 512
v_mul_f32 v60, v61, v60                            // 2^32 / divisor
v_cvt_u32_f32 v60, v60                             // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_lo_u32 v61, v60, s[sgprNumWorkGroups0]       // refine 2^32 / divisor
v_sub_nc_u32 v61, 0, v61                           // refine 2^32 / divisor
v_mul_hi_u32 v61, v60, v61                         // refine 2^32 / divisor
v_add_nc_u32 v60, v60, v61                         // refine 2^32 / divisor
v_mul_hi_u32 v60, s[sgprWorkGroup0], v60           // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_lo_u32 v61, v60, s[sgprNumWorkGroups0]       // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v61, s[sgprWorkGroup0], v61           // s62 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_ge_u32 vcc_lo, v61, s[sgprNumWorkGroups0]    // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v60, 1, v60                           // quotient + 1
v_sub_nc_u32 v61, v61, s[sgprNumWorkGroups0]       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v61, s[sgprNumWorkGroups0]    // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v60, 1, v60                           // quotient + 1
v_sub_nc_u32 v61, v61, s[sgprNumWorkGroups0]       // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s62, v60                       // quotient
s_mov_b32 s[sgprWorkGroup1], s62
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s62, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s62
label_RemapWG1Done_1:
s_branch label_RemapWGDone_1
.set sgprAddressA, 24
.set sgprAddressB, 26
.set sgprAddressC, 36
.set sgprAddressD, 34
.set sgprAddressScaleA, 44
.set sgprAddressScaleB, 46
.set sgprAddressScaleZeroA, 60
.set sgprAlpha, 32
.set sgprArgType, 5
.set sgprKernArgAddress, 0
.set sgprNumWorkGroups0, 17
.set sgprNumWorkGroups1, 18
.set sgprScaleAGMinus1, 52
.set sgprSizesFree, 20
.set sgprStridesA, 28
.set sgprStridesB, 30
.set sgprStridesC, 40
.set sgprStridesD, 38
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s18, 112                                 // KernArgAddressOffset
s_mul_i32 s74, s19, 4
s_mov_b64 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s18, 232
s_mov_b32 s74, 0
s_mov_b64 s[68:69], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s17, 1
s_mov_b32 s75, 0
s_load_b128 s[20:23], s[68:69], s74
s_cmpk_eq_u32 s19, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s72, s20, 1                             // s72 = s20 / 2
s_and_b32 s70, 1, s20                              // s70 = s20 % 2
s_addc_u32 s72, s72, 0
s_mul_hi_u32 s71, s21, -1431655765                 // STATIC_DIV: divisor=3
s_mul_i32 s70, s21, -1431655765                    // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp0 = quotient
s_mul_i32 s71, s70, 3                              // tmp1 = quotient * divisor
s_cmp_lg_u32 s71, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s73, s70, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s72, s72, s73
s_mul_i32 s72, s72, s22
s_add_u32 s75, s75, s72
s_cmp_lt_u32 s[sgprWorkGroup0], s75
s_cbranch_scc1 label_FOUND
s_add_u32 s74, s74, s18
s_load_b128 s[20:23], s[68:69], s74
s_add_u32 s17, s17, 1
s_cmp_lt_u32 s17, s19
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s72, s20, 1                             // s72 = s20 / 2
s_and_b32 s70, 1, s20                              // s70 = s20 % 2
s_addc_u32 s72, s72, 0
s_mul_hi_u32 s71, s21, -1431655765                 // STATIC_DIV: divisor=3
s_mul_i32 s70, s21, -1431655765                    // tmp = dividend * magic
s_lshr_b64 s[70:71], s[70:71], 33                  // tmp0 = quotient
s_mul_i32 s71, s70, 3                              // tmp1 = quotient * divisor
s_cmp_lg_u32 s71, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s73, s70, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s72, s72, s73
s_mul_i32 s72, s72, s22
s_add_u32 s75, s75, s72

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s69, s17, 1
s_sub_u32 s68, s75, s72
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s68
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s19, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s69, s69, 112                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s69
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0

/* Load Kernel Args */
s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b64 s[40:41], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s69, s69, 232
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s69
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b64 s[sgprAddressD:sgprAddressD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_b64 s[sgprAddressC:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18
s_load_b64 s[sgprAddressA:sgprAddressA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20
s_load_b64 s[sgprAddressB:sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28
s_load_b64 s[sgprStridesD:sgprStridesD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_b64 s[sgprStridesC:sgprStridesC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38
s_load_b64 s[sgprStridesA:sgprStridesA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_b64 s[sgprStridesB:sgprStridesB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48
s_load_b32 s[sgprAlpha], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
// Read Beta
s_load_b32 s33, s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x64
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x6c
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x74
label_LoadExternalStructEnd:
/* init: add vgpr [60...281) to pool */
/* init: add vgpr [0...6) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 32 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 5, v[vgprSerial]                 // 0 = Serial / 32
v_and_b32 v1, 31, v[vgprSerial]                    // 1 = Serial % 32
/* unroll *= glvw */
v_lshlrev_b32 v1, 5, v1                            // v1 = v1 * 32
/* LVCB = 32 */
/* v3 = B-unroll = serial%LVCB */
v_lshlrev_b32 v3, 5, v[vgprSerial]                 // B-unroll = serial * glvw
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */
s_waitcnt lgkmcnt(0)                               // wait for 88/0 bytes of kern args over preload
s_add_u32 s[sgprNumWorkGroups0], s[sgprSizesFree+0], 1 // size + MT0-1
s_lshr_b32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 1 // ceil(size / MT0)
s_mul_hi_u32 s69, s[sgprSizesFree+1], -1431655765  // STATIC_DIV: divisor=3
s_mul_i32 s68, s[sgprSizesFree+1], -1431655765     // tmp = dividend * magic
s_lshr_b64 s[68:69], s[68:69], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s68               // quotient
s_mul_i32 s68, s[sgprNumWorkGroups1], 3            // quotient*divisor
s_sub_u32 s62, s[sgprSizesFree+1], s68             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s62, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mov_b32 s[sgprWorkGroup2], 0
s_mov_b32 s[sgprWorkGroup1], 0
s_cmp_ge_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // wg2 or wg1 != 0?
s_cbranch_scc1 label_RemapWG_1
label_RemapWGDone_1:

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

s_branch label_MultiGemmEnd
.set sgprNumWorkGroups0, 17
.set sgprNumWorkGroups1, 18
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
label_WGMRemap:
s_cmp_gt_i32 s19, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s19, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s19, s19                                 // abs(WGM)
v_cvt_f64_u32 v[60:61], s19                        // s59 = s[sgprWorkGroup0] / s19
v_rcp_f64 v[60:61], v[60:61]                       // s59 = s[sgprWorkGroup0] / s19
v_cvt_f64_u32 v[62:63], s[sgprWorkGroup0]          // s59 = s[sgprWorkGroup0] / s19
v_mul_f64 v[60:61], v[60:61], v[62:63]             // s59 = s[sgprWorkGroup0] / s19
v_cvt_u32_f64 v60, v[60:61]                        // s59 = s[sgprWorkGroup0] / s19
v_mul_lo_u32 v61, v60, s19                         // s59 = s[sgprWorkGroup0] / s19
v_sub_nc_u32 v62, s[sgprWorkGroup0], v61           // s59 = s[sgprWorkGroup0] / s19
v_cmp_ge_u32 vcc_lo, v62, s19                      // s59 = s[sgprWorkGroup0] / s19
s_mov_b32 exec_lo, vcc_lo                          // s59 = s[sgprWorkGroup0] / s19
v_add_nc_u32 v60, v60, 1                           // s59 = s[sgprWorkGroup0] / s19
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s59, v60                       // quotient
s_mul_i32 s68, s59, s19                            // quotient * non-magic divisor
s_sub_u32 s68, s[sgprWorkGroup0], s68              // WorkGroup0=remainder
s_mul_i32 s68, s68, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s68, s68, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[60:61], s19                        // s62 = s[sgprNumWorkGroups0] / s19
v_rcp_f64 v[60:61], v[60:61]                       // s62 = s[sgprNumWorkGroups0] / s19
v_cvt_f64_u32 v[62:63], s[sgprNumWorkGroups0]      // s62 = s[sgprNumWorkGroups0] / s19
v_mul_f64 v[60:61], v[60:61], v[62:63]             // s62 = s[sgprNumWorkGroups0] / s19
v_cvt_u32_f64 v60, v[60:61]                        // s62 = s[sgprNumWorkGroups0] / s19
v_mul_lo_u32 v61, v60, s19                         // s62 = s[sgprNumWorkGroups0] / s19
v_sub_nc_u32 v62, s[sgprNumWorkGroups0], v61       // s62 = s[sgprNumWorkGroups0] / s19
v_cmp_ge_u32 vcc_lo, v62, s19                      // s62 = s[sgprNumWorkGroups0] / s19
s_mov_b32 exec_lo, vcc_lo                          // s62 = s[sgprNumWorkGroups0] / s19
v_add_nc_u32 v60, v60, 1                           // s62 = s[sgprNumWorkGroups0] / s19
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s62, v60                       // quotient
s_mul_i32 s63, s19, s62                            // quotient * non-magic divisor
s_sub_u32 s63, s[sgprNumWorkGroups0], s63          // NumWorkGroups0=remainder
s_cmp_eq_u32 s63, 0                                // remainder == 0 ?
s_cmov_b32 s63, s19                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s59, s62                              // blockId >= numFullBlocks ?
s_cselect_b32 s62, s63, s19
v_cvt_f64_u32 v[60:61], s62                        // s[sgprWorkGroup1] = s68 / s62
v_rcp_f64 v[60:61], v[60:61]                       // s[sgprWorkGroup1] = s68 / s62
v_cvt_f64_u32 v[62:63], s68                        // s[sgprWorkGroup1] = s68 / s62
v_mul_f64 v[60:61], v[60:61], v[62:63]             // s[sgprWorkGroup1] = s68 / s62
v_cvt_u32_f64 v60, v[60:61]                        // s[sgprWorkGroup1] = s68 / s62
v_mul_lo_u32 v61, v60, s62                         // s[sgprWorkGroup1] = s68 / s62
v_sub_nc_u32 v62, s68, v61                         // s[sgprWorkGroup1] = s68 / s62
v_cmp_ge_u32 vcc_lo, v62, s62                      // s[sgprWorkGroup1] = s68 / s62
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s68 / s62
v_add_nc_u32 v60, v60, 1                           // s[sgprWorkGroup1] = s68 / s62
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v61, v60, s62                         // s[sgprWorkGroup1] = s68 / s62
v_sub_nc_u32 v62, s68, v61                         // s[sgprWorkGroup1] = s68 / s62
v_readfirstlane_b32 s[sgprWorkGroup1], v60         // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v62         // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s62 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s68, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s59, s59, s19                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s59 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s19, s19                                 // WGM
v_cvt_f64_u32 v[60:61], s19                        // s59 = s[sgprWorkGroup1] / s19
v_rcp_f64 v[60:61], v[60:61]                       // s59 = s[sgprWorkGroup1] / s19
v_cvt_f64_u32 v[62:63], s[sgprWorkGroup1]          // s59 = s[sgprWorkGroup1] / s19
v_mul_f64 v[60:61], v[60:61], v[62:63]             // s59 = s[sgprWorkGroup1] / s19
v_cvt_u32_f64 v60, v[60:61]                        // s59 = s[sgprWorkGroup1] / s19
v_mul_lo_u32 v61, v60, s19                         // s59 = s[sgprWorkGroup1] / s19
v_sub_nc_u32 v62, s[sgprWorkGroup1], v61           // s59 = s[sgprWorkGroup1] / s19
v_cmp_ge_u32 vcc_lo, v62, s19                      // s59 = s[sgprWorkGroup1] / s19
s_mov_b32 exec_lo, vcc_lo                          // s59 = s[sgprWorkGroup1] / s19
v_add_nc_u32 v60, v60, 1                           // s59 = s[sgprWorkGroup1] / s19
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s59, v60                       // quotient
s_mul_i32 s68, s59, s19                            // quotient * non-magic divisor
s_sub_u32 s68, s[sgprWorkGroup1], s68              // WorkGroup1=remainder
s_mul_i32 s68, s68, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s68, s68, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[60:61], s19                        // s62 = s[sgprNumWorkGroups1] / s19
v_rcp_f64 v[60:61], v[60:61]                       // s62 = s[sgprNumWorkGroups1] / s19
v_cvt_f64_u32 v[62:63], s[sgprNumWorkGroups1]      // s62 = s[sgprNumWorkGroups1] / s19
v_mul_f64 v[60:61], v[60:61], v[62:63]             // s62 = s[sgprNumWorkGroups1] / s19
v_cvt_u32_f64 v60, v[60:61]                        // s62 = s[sgprNumWorkGroups1] / s19
v_mul_lo_u32 v61, v60, s19                         // s62 = s[sgprNumWorkGroups1] / s19
v_sub_nc_u32 v62, s[sgprNumWorkGroups1], v61       // s62 = s[sgprNumWorkGroups1] / s19
v_cmp_ge_u32 vcc_lo, v62, s19                      // s62 = s[sgprNumWorkGroups1] / s19
s_mov_b32 exec_lo, vcc_lo                          // s62 = s[sgprNumWorkGroups1] / s19
v_add_nc_u32 v60, v60, 1                           // s62 = s[sgprNumWorkGroups1] / s19
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s62, v60                       // quotient
s_mul_i32 s63, s19, s62                            // quotient * non-magic divisor
s_sub_u32 s63, s[sgprNumWorkGroups1], s63          // NumWorkGroups1=remainder
s_cmp_eq_u32 s63, 0                                // remainder == 0 ?
s_cmov_b32 s63, s19                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s59, s62                              // blockId >= numFullBlocks ?
s_cselect_b32 s62, s63, s19
v_cvt_f64_u32 v[60:61], s62                        // s[sgprWorkGroup0] = s68 / s62
v_rcp_f64 v[60:61], v[60:61]                       // s[sgprWorkGroup0] = s68 / s62
v_cvt_f64_u32 v[62:63], s68                        // s[sgprWorkGroup0] = s68 / s62
v_mul_f64 v[60:61], v[60:61], v[62:63]             // s[sgprWorkGroup0] = s68 / s62
v_cvt_u32_f64 v60, v[60:61]                        // s[sgprWorkGroup0] = s68 / s62
v_mul_lo_u32 v61, v60, s62                         // s[sgprWorkGroup0] = s68 / s62
v_sub_nc_u32 v62, s68, v61                         // s[sgprWorkGroup0] = s68 / s62
v_cmp_ge_u32 vcc_lo, v62, s62                      // s[sgprWorkGroup0] = s68 / s62
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s68 / s62
v_add_nc_u32 v60, v60, 1                           // s[sgprWorkGroup0] = s68 / s62
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v61, v60, s62                         // s[sgprWorkGroup0] = s68 / s62
v_sub_nc_u32 v62, s68, v61                         // s[sgprWorkGroup0] = s68 / s62
v_readfirstlane_b32 s[sgprWorkGroup0], v60         // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v62         // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s62 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s68, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s59, s59, s19                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s59 // wg1 += blockId * WGM
s_branch label_WGM
.set sgprAddressA, 24
.set sgprKernArgAddress, 0
.set sgprSizesSum, 23
.set sgprSrdA, 68
.set sgprWorkGroup2, 4
label_GeneralBatchedGemmLoadA:
s_mul_i32 s78, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s78, s78, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s79, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdA:sgprSrdA+1], s[78:79], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[78:79], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s78        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s79       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshr_b64 s[80:81], s[80:81], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s80, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s81, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
.set sgprAddressB, 26
.set sgprKernArgAddress, 0
.set sgprSizesSum, 23
.set sgprSrdB, 72
.set sgprWorkGroup2, 4
label_GeneralBatchedGemmLoadB:
s_mul_i32 s78, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s78, s78, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s79, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdB:sgprSrdB+1], s[78:79], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[78:79], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s78        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s79       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 64         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[80:81], s[80:81], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s80, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s81, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadB_End
.set sgprAddressC, 36
.set sgprKernArgAddress, 0
.set sgprSrdC, 28
.set sgprWorkGroup2, 4
label_GeneralBatchedGemmLoadC:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s44, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s44, s44, s[sgprAddressC+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s45, s[sgprAddressC+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[44:45], s[44:45], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // Load batchOffsetC from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s44        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s45       // Add matrix address to SRD (high)
s_branch label_GeneralBatchedGemmLoadC_End
.set sgprAddressD, 34
.set sgprKernArgAddress, 0
.set sgprSrdD, 24
.set sgprWorkGroup2, 4
label_GeneralBatchedGemmLoadD:  /// Computing the Batch Matrix's base address for General Batched GEMM
s_mul_i32 s44, 8, s[sgprWorkGroup2]                // Compute stride in bytes into Pointer Array
s_add_u32 s44, s44, s[sgprAddressD+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s45, s[sgprAddressD+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[44:45], s[44:45], 0                   // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for the Matrix Address Load from the Pointer Array
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // Offsetting within the Batch Matrix [Lower half of address]
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // Offsetting within the Batch Matrix [Higher half of address]
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70 // Load batchOffsetD from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s44        // Add matrix address to SRD (low)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s45       // Add matrix address to SRD (high)
s_branch label_GeneralBatchedGemmLoadD_End
label_ASM_End:  /// The end of the kernel
