
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1150"
.text
.protected RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT2x5x1024_Dot2_gfx1150
.globl RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT2x5x1024_Dot2_gfx1150
.p2align 8
.type RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT2x5x1024_Dot2_gfx1150,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT2x5x1024_Dot2_gfx1150
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 255 // vgprs
  .amdhsa_next_free_sgpr 85 // sgprs
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
/* Num SGPR   =85 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 2 x 5 */
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
  - .name: RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT2x5x1024_Dot2_gfx1150
    .symbol: 'RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT2x5x1024_Dot2_gfx1150.kd'
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
        .value_type:      bf16
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
        .value_type:      bf16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          80
        .value_kind:      global_buffer
        .value_type:      bf16
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
    .sgpr_count:                 85
    .sgpr_spill_count:           0
    .vgpr_count:                 255
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
RuntimeGroup_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGZPU8_UserArgs_MT2x5x1024_Dot2_gfx1150:
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
/* ValuC range: [0-10),  */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprBase, 56
.set vgprLocalWriteAddrA, 52
.set vgprLocalWriteAddrB, 53
.set vgprGlobalReadOffsetA, 10
.set vgprGlobalReadOffsetB, 12
.set vgprGlobalReadOffsetScaleA, 32
.set vgprG2LScaleA, 34
.set vgprDot2PS, 36
.set vgprG2LScaleZeroA, 46
.set vgprGlobalReadOffsetScaleZeroA, 48
.set vgprGlobalReadByteOffsetScaleZeroA, 50
.set vgprLocalReadAddrA, 54
.set vgprLocalReadAddrB, 55
.set vgprSerial, 169

/******************************************/
/* VGPR Macro Assignments                 */
/******************************************/
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+33
.set vgprG2LA_BASE, vgprBase+0
.set vgprG2LB_BASE, vgprBase+33
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuA_X1_I0, vgprValuA_X0_I0_BASE+2
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprValuB_X1_I0, vgprValuB_X0_I0_BASE+5
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
.set sgprNumWorkGroups0, 16
.set sgprNumWorkGroups1, 17
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
.set sgprScaleALiftMagic, 57
.set sgprAddressScaleZeroA, 64
.set sgprSrdScaleZeroA, 68

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
.set MT1, 5
.set DepthU, 1024
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
s_load_b32 s18, s[sgprKernArgAddress:sgprKernArgAddress+1], 0

/* Load packed kernel args (StaggerU/GSU) */
s_load_b32 s63, s[sgprKernArgAddress:sgprKernArgAddress+1], 4

/* Load WGM data */
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8

/* Load num of WGs */
s_load_b32 s66, s[sgprKernArgAddress:sgprKernArgAddress+1], 12

/* Load Kernel Args */
s_load_b512 s[20:35], s[sgprKernArgAddress:sgprKernArgAddress+1], 16 // 16
s_load_b128 s[36:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 80 // 80
s_load_b64 s[40:41], s[sgprKernArgAddress:sgprKernArgAddress+1], 96 // 96
s_load_b32 s[sgprScaleAGMinus1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa0
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78
s_waitcnt lgkmcnt(0)                               // load args
s_lshr_b32 s19, s18, 0x1e                          // Get arg type
s_and_b32 s18, 0x3fffffff, s18                     // Get nums of gemm
s_cmp_eq_u32 s19, 3                                // Is kernel argType == 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s19, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
label_LoadArgsEnd:
s_and_b32 s[sgprGSU], s63, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s19
s_mov_b32 m0, 0xe38                                // LDS clamp at 3640 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_mov_b32 vcc_hi, 0                                // Ensure hi bits are zero

/* remap workgroup to XCCs */
s_lshr_b32 s76, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s76, s76                             // Get log(WGMXCC)
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s76, 0
s_cbranch_scc1 label_WGMXCC
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s19, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s19, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
/* init: add vgpr [56...225) to pool */
/* init: add vgpr [0...10) to pool */
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
s_mul_hi_u32 s73, s[sgprSizesFree+1], 1717986919   // STATIC_DIV: divisor=5
s_mul_i32 s72, s[sgprSizesFree+1], 1717986919      // tmp = dividend * magic
s_lshr_b64 s[72:73], s[72:73], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s72               // quotient
s_mul_i32 s72, s[sgprNumWorkGroups1], 5            // quotient*divisor
s_sub_u32 s63, s[sgprSizesFree+1], s72             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s63, 0                                // Check if remainder > 0
s_addc_u32 s[sgprNumWorkGroups1], s[sgprNumWorkGroups1], 0 // Increment if remainder > 0

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mov_b32 s[sgprWorkGroup2], 0
s_mov_b32 s[sgprWorkGroup1], 0
s_cmp_ge_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // wg2 or wg1 != 0?
s_cbranch_scc1 label_RemapWG
label_RemapWGDone:
label_MultiGemmEnd:
.set sgprSrdA, 72
.set sgprSrdB, 76
.set sgprShadowLimitA, 18
.set sgprShadowLimitB, 66
.set sgprGlobalReadIncsA, 63
.set sgprGlobalReadIncsB, 80
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 64 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift
label_Skip_Address_Prepad_For_Pointer_Array:  /// Skip pre-padding of address for pointer array case

/* Short circuit condition if Alpha == 0, then sumDims=0 */
s_and_b32 s18, s[sgprAlpha], 0x7fffffff            // SCC = Alpha != 0
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
s_sext_i32_i16 s18, s[sgprWGM]                     // Restore WGM
s_cmp_gt_u32 s18, 1                                // WGM != 0 and 1 ?
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
v_mov_b32 v6, v0                                   // groA0I_0
v_add_co_u32 v7, vcc_lo, 1, v6                     // groA0I_1 += LSPA

/* global read addresses: tile offsets b */

/* global read addresses: unroll offsets a */
v_mov_b32 v8, v1                                   // groAL_0

/* global read addresses: unroll offsets b */

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_i32 s82, s[sgprWorkGroup0], 2                // WorkGroup[01] * MT
s_mul_hi_u32 s83, s82, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s82, s82, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mov_b32 s[sgprShadowLimitA+0], s[sgprSizeL]      // Init tensor size
s_mov_b32 s[sgprShadowLimitA+1], 0                 // init tensor size
s_sub_u32 s80, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s81, s[sgprStrideA0I], s80            // stride x (size-1)
s_mul_i32 s80, s[sgprStrideA0I], s80               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s80 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s81 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s82 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s83 // sub tileStart
s_lshr_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadA
s_mul_hi_u32 s81, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s80, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s82, s82, s80                            // accum wg term to tilestart
s_addc_u32 s83, s83, s81                           // accum wg term to tilestart
s_lshr_b64 s[82:83], s[82:83], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s82    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s83   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadA_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: block-scale A srd */
s_ff1_i32_b32 s[sgprScaleAGShift], s[sgprScaleAGMinus1]
s_lshr_b32 s[sgprScaleAKMask], s[sgprScaleAGMinus1], 10
s_max_u32 s[sgprScaleAKMask], s[sgprScaleAKMask], 1
s_sub_u32 s[sgprScaleAKMask], s[sgprScaleAKMask], 1
s_mov_b32 s[sgprScaleAInc], 1024
s_lshr_b32 s[sgprScaleAInc], s[sgprScaleAInc], s[sgprScaleAGShift]
s_max_u32 s[sgprScaleAInc], s[sgprScaleAInc], 1
s_lshl_b32 s[sgprScaleAInc], s[sgprScaleAInc], 1
s_mov_b32 s[sgprScaleAKCnt], 0
s_add_u32 s[sgprStrideScaleA], s[sgprSizeL], s[sgprScaleAGMinus1] // SizeL + G
s_sub_u32 s[sgprStrideScaleA], s[sgprStrideScaleA], 1 // SizeL + G-1
s_lshr_b32 s[sgprStrideScaleA], s[sgprStrideScaleA], s[sgprScaleAGShift] // StrideScaleA = ceil(SizeL/32) scale elements
s_mul_i32 s80, s[sgprWorkGroup0], s[sgprStrideScaleA] // scaleA: workgroup * row stride
s_mul_i32 s80, s80, 4                              // scaleA: * MT0 rows, elements -> bytes
s_mul_i32 s81, s[sgprSizeI], s[sgprStrideScaleA]   // scaleA: SizeI * row stride
s_mul_i32 s81, s81, 2                              // scaleA: tensor bytes
s_sub_u32 s[sgprSrdScaleA+2], s81, s80             // scaleA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s80 // scaleA: SRD base lo
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0 // scaleA: SRD base hi
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96            // scaleA: set bits 127_96 in SRD
s_mov_b32 s[sgprScaleALiftMagic+0], 0x4b000000     // w4a16: 2^23, lsb of nibble 0 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+1], 0x49000000     // w4a16: 2^19, lsb of nibble 1 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+2], 0x47000000     // w4a16: 2^15, lsb of nibble 2 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+3], 0x45000000     // w4a16: 2^11, lsb of nibble 3 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+4], 0x43000000     // w4a16: 2^7, lsb of nibble 4 at 1.0
s_mov_b32 s[sgprScaleALiftMagic+5], 0x8000         // w4a16: bf16 rounding bias

/* global read addresses: block-scale A zero-point srd */
s_mul_i32 s80, s[sgprWorkGroup0], 2                // scaleZeroA: workgroup row origin
s_lshr_b32 s80, s80, 3                             // scaleZeroA: row origin to words
s_lshl_b32 s80, s80, 2                             // scaleZeroA: words to bytes
s_mul_i32 s80, s80, s[sgprStrideScaleA]            // scaleZeroA: * kGroups
s_add_u32 s81, s[sgprSizeI], 7                     // scaleZeroA: SizeI + 7
s_lshr_b32 s81, s81, 3                             // scaleZeroA: ceil(SizeI/8) words
s_lshl_b32 s81, s81, 2                             // scaleZeroA: words to bytes
s_mul_i32 s81, s81, s[sgprStrideScaleA]            // scaleZeroA: total bytes
s_sub_u32 s[sgprSrdScaleZeroA+2], s81, s80         // scaleZeroA: buffer limit from the workgroup origin
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprAddressScaleZeroA+0], s80 // scaleZeroA: SRD base lo
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprAddressScaleZeroA+1], 0 // scaleZeroA: SRD base hi
s_mov_b32 s[sgprSrdScaleZeroA+3], Srd127_96        // scaleZeroA: set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_i32 s82, s[sgprWorkGroup1], 5                // WorkGroup[01] * MT
s_mul_hi_u32 s83, s82, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s82, s82, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mov_b32 s[sgprShadowLimitB+0], s[sgprSizeL]      // Init tensor size
s_mov_b32 s[sgprShadowLimitB+1], 0                 // init tensor size
s_sub_u32 s80, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s81, s[sgprStrideB1J], s80            // stride x (size-1)
s_mul_i32 s80, s[sgprStrideB1J], s80               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s80 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s81 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s82 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s83 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1 // Set limit to use bytes (multiple bpe)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 64 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_cmp_eq_u32 s[sgprArgType], 3                     // ArgType == 3 for General Batched GEMM
s_cbranch_scc1 label_GeneralBatchedGemmLoadB
s_mul_hi_u32 s81, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s80, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s82, s82, s80                            // accum wg term to tilestart
s_addc_u32 s83, s83, s81                           // accum wg term to tilestart
s_lshl_b64 s[82:83], s[82:83], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s82    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s83   // SRD base = Address+ tileStart1
label_StridedBatchedGemmLoadB_End:  /// End Computing the Batch Matrix's base address for Strided Batched
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: final offsets a */
/* ============================================================= */
v_mul_lo_u32 v61, s[sgprStrideA0I], v[6]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[8], v[61+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x20, v[vgprGlobalReadOffsetA+0+0] // add prepad for pointer shift
v_lshrrev_b32 v61, s[sgprScaleAGShift], v8         // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v61, v[vgprGlobalReadOffsetScaleA+0] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0] // scaleA: elements -> bytes
s_mul_i32 s81, s[sgprWorkGroup0], 2                // scaleZeroA: workgroup row origin
s_and_b32 s81, s81, 7                              // scaleZeroA: origin row within its word
v_add_nc_u32 v62, s81, v6                          // scaleZeroA: row from the word origin
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v62 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+0], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+0], v61, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 3, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: word to nibble index
v_and_b32 v61, 7, v62                              // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+0], v61, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+0], 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0] //  (multiple bpe)
v_mul_lo_u32 v61, s[sgprStrideA0I], v[7]           // mul d1 lower
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[8], v[61+0] // accumulate K lower
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x20, v[vgprGlobalReadOffsetA+1+0] // add prepad for pointer shift
v_lshrrev_b32 v61, s[sgprScaleAGShift], v8         // scaleA: kGroup = k/32
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7 // scaleA: row * StrideScaleA
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v61, v[vgprGlobalReadOffsetScaleA+1] // scaleA: + kGroup
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1] // scaleA: elements -> bytes
s_mul_i32 s81, s[sgprWorkGroup0], 2                // scaleZeroA: workgroup row origin
s_and_b32 s81, s81, 7                              // scaleZeroA: origin row within its word
v_add_nc_u32 v62, s81, v7                          // scaleZeroA: row from the word origin
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v62 // scaleZeroA: row / 8
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+1], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (row/8) * kGroups
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+1], v61, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: + kGroup -> word offset
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 3, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: word to nibble index
v_and_b32 v61, 7, v62                              // scaleZeroA: nibble = row & 7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+1], v61, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: (word << 3) | nibble
v_lshrrev_b32 v[vgprGlobalReadByteOffsetScaleZeroA+1], 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: save loop-invariant byte offset
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1] //  (multiple bpe)
/* ============================================================= */

/* global read addresses: final offsets b */
/* ============================================================= */
v_add_nc_u32 v[vgprGlobalReadOffsetB+0], 0x20, v3  // gROB_0: unroll + prepad
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+1], v[vgprGlobalReadOffsetB+0] // gROB_1: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+2], v[vgprGlobalReadOffsetB+0] // gROB_2: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+3], v[vgprGlobalReadOffsetB+0] // gROB_3: same row and unroll
s_add_u32 s81, s[sgprStrideB1J], 0x20              // + prepad
v_add_nc_u32 v[vgprGlobalReadOffsetB+4], s81, v3   // gROB_4: + unroll
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+5], v[vgprGlobalReadOffsetB+4] // gROB_5: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+6], v[vgprGlobalReadOffsetB+4] // gROB_6: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+7], v[vgprGlobalReadOffsetB+4] // gROB_7: same row and unroll
s_mul_i32 s81, s[sgprStrideB1J], 2                 // row(2) * stride
s_add_u32 s81, s81, 0x20                           // + prepad
v_add_nc_u32 v[vgprGlobalReadOffsetB+8], s81, v3   // gROB_8: + unroll
v_lshlrev_b32 v[vgprGlobalReadOffsetB+8], 1, v[vgprGlobalReadOffsetB+8] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+9], v[vgprGlobalReadOffsetB+8] // gROB_9: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+10], v[vgprGlobalReadOffsetB+8] // gROB_10: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+11], v[vgprGlobalReadOffsetB+8] // gROB_11: same row and unroll
s_mul_i32 s81, s[sgprStrideB1J], 3                 // row(3) * stride
s_add_u32 s81, s81, 0x20                           // + prepad
v_add_nc_u32 v[vgprGlobalReadOffsetB+12], s81, v3  // gROB_12: + unroll
v_lshlrev_b32 v[vgprGlobalReadOffsetB+12], 1, v[vgprGlobalReadOffsetB+12] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+13], v[vgprGlobalReadOffsetB+12] // gROB_13: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+14], v[vgprGlobalReadOffsetB+12] // gROB_14: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+15], v[vgprGlobalReadOffsetB+12] // gROB_15: same row and unroll
s_mul_i32 s81, s[sgprStrideB1J], 4                 // row(4) * stride
s_add_u32 s81, s81, 0x20                           // + prepad
v_add_nc_u32 v[vgprGlobalReadOffsetB+16], s81, v3  // gROB_16: + unroll
v_lshlrev_b32 v[vgprGlobalReadOffsetB+16], 1, v[vgprGlobalReadOffsetB+16] //  (multiple bpe)
v_mov_b32 v[vgprGlobalReadOffsetB+17], v[vgprGlobalReadOffsetB+16] // gROB_17: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+18], v[vgprGlobalReadOffsetB+16] // gROB_18: same row and unroll
v_mov_b32 v[vgprGlobalReadOffsetB+19], v[vgprGlobalReadOffsetB+16] // gROB_19: same row and unroll
/* ============================================================= */
s_and_b32 s[sgprDot2TailRem], s[sgprSizesSum+0], 0x3ff // K tail: K % DepthU(1024)
v_and_b32 v0, 0x1f, v[vgprSerial]                  // K tail: lane
v_lshlrev_b32 v0, 0x5, v0                          // K tail: lane K block start
s_sub_i32 s81, s[sgprDot2TailRem], 0x0             // K tail: remainder past slot 0's start
v_cmp_gt_i32 s[sgprDot2TailMask+0], s81, v0        // K tail: slot 0 lanes inside K

/* global read addresses: increments a */
s_mov_b32 s[sgprGlobalReadIncsA+0], 512            // incrA = DepthU*Bpe*MI_dim(1)

/* global read addresses: increments b */
s_mov_b32 s[sgprGlobalReadIncsB+0], 2048           // incrB = DepthU*Bpe*MI_dim(1)
/* declare loop num iterations */

/* initC: remove ValuC vgpr buffer [0...10) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [56...169) from pool */
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
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 10 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 1024
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
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_1_0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 0_1_1_0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 0_2_1_0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 0_3_1_0
buffer_load_b128 v[vgprG2LB+32:vgprG2LB+32+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_2_0
buffer_load_b128 v[vgprG2LB+36:vgprG2LB+36+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 0_1_2_0
buffer_load_b128 v[vgprG2LB+40:vgprG2LB+40+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 0_2_2_0
buffer_load_b128 v[vgprG2LB+44:vgprG2LB+44+3], v[vgprGlobalReadOffsetB+8], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 0_3_2_0
buffer_load_b128 v[vgprG2LB+48:vgprG2LB+48+3], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_3_0
buffer_load_b128 v[vgprG2LB+52:vgprG2LB+52+3], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 0_1_3_0
buffer_load_b128 v[vgprG2LB+56:vgprG2LB+56+3], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 0_2_3_0
buffer_load_b128 v[vgprG2LB+60:vgprG2LB+60+3], v[vgprGlobalReadOffsetB+12], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 0_3_3_0
buffer_load_b128 v[vgprG2LB+64:vgprG2LB+64+3], v[vgprGlobalReadOffsetB+16], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_4_0
buffer_load_b128 v[vgprG2LB+68:vgprG2LB+68+3], v[vgprGlobalReadOffsetB+16], s[sgprSrdB:sgprSrdB+3], 0 offen offset:16 // G -> Reg 0_1_4_0
buffer_load_b128 v[vgprG2LB+72:vgprG2LB+72+3], v[vgprGlobalReadOffsetB+16], s[sgprSrdB:sgprSrdB+3], 0 offen offset:32 // G -> Reg 0_2_4_0
buffer_load_b128 v[vgprG2LB+76:vgprG2LB+76+3], v[vgprGlobalReadOffsetB+16], s[sgprSrdB:sgprSrdB+3], 0 offen offset:48 // G -> Reg 0_3_4_0

/* global read block-scale A */
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0 // load block scale for A load 1
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v[vgprGlobalReadByteOffsetScaleZeroA+0], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 0
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v[vgprGlobalReadByteOffsetScaleZeroA+1], s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0 // load packed zero-point pair for A load 1
buffer_load_b128 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 slc dlc // G -> Reg 0_0_0_0
buffer_load_b128 v[vgprG2LA+28:vgprG2LA+28+3], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 slc dlc // G -> Reg 0_0_1_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc block-scale A (2 bytes every 1 iters) */
s_add_u32 s[sgprScaleAKCnt], s[sgprScaleAKCnt], 1  // scaleA: one more K iteration done
s_and_b32 s82, s[sgprScaleAKCnt], s[sgprScaleAKMask] // scaleA: SCC = counter has not wrapped
s_cselect_b32 s82, 0, s[sgprScaleAInc]             // scaleA: advance only on a wrap
s_lshl_b32 s83, s82, 1                             // scaleZeroA: advance only on a wrap
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], s82 // scaleA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0 // scaleA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], s82 // scaleA limit -= inc
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], s83 // scaleZeroA SRD += inc(lower)
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0 // scaleZeroA SRD += inc(upper)
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], s83 // scaleZeroA limit -= inc

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0x0       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0x0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
s_waitcnt vmcnt(2)                                 // 5wait for global read B
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
s_waitcnt vmcnt(0)                                 // A slot 0 loaded
v_lshlrev_b32 v[vgprG2LScaleA+0], 16, v[vgprG2LScaleA+0] // scaleA: bf16 -> f32, for the MAC
v_and_b32 v179, 1, v[vgprGlobalReadOffsetScaleZeroA+0] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v179, 2, v179                        // scaleZeroA: nibble shift
v_bfe_u32 v179, v[vgprG2LScaleZeroA+0], v179, 0x4  // scaleZeroA: z
v_cvt_f32_i32 v179, v179                           // scaleZeroA: c to f32
v_mov_b32 v178, v[vgprG2LA+12+0]                   // w4a16: save packed int4 dword 0
v_and_or_b32 v172, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v173, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v175, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v174, v178, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+0], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+1], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+2], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+3], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+0+0], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+0+1], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_and_or_b32 v172, v178, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v178, 20, v178                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v173, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v175, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v174, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+4], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+0], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+1], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+2], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+0+2], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+0+3], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_mov_b32 v178, v[vgprG2LA+12+1]                   // w4a16: save packed int4 dword 1
v_and_or_b32 v172, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v173, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v175, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v174, v178, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+0], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+1], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+2], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+3], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+0+4], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+0+5], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_and_or_b32 v172, v178, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v178, 20, v178                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v173, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v175, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v174, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+4], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+0], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+1], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+2], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+0+6], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+0+7], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_mov_b32 v178, v[vgprG2LA+12+2]                   // w4a16: save packed int4 dword 2
v_and_or_b32 v172, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v173, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v175, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v174, v178, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+0], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+1], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+2], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+3], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+0+8], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+0+9], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_and_or_b32 v172, v178, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v178, 20, v178                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v173, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v175, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v174, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+4], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+0], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+1], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+2], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+0+10], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+0+11], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_mov_b32 v178, v[vgprG2LA+12+3]                   // w4a16: save packed int4 dword 3
v_and_or_b32 v172, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v173, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v175, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v174, v178, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+0], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+1], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+2], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+3], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+0+12], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+0+13], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_and_or_b32 v172, v178, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v178, 20, v178                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v173, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v175, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v174, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+4], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+0], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+1], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+2], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+0+14], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+0+15], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_lshlrev_b32 v[vgprG2LScaleA+1], 16, v[vgprG2LScaleA+1] // scaleA: bf16 -> f32, for the MAC
v_and_b32 v179, 1, v[vgprGlobalReadOffsetScaleZeroA+1] // scaleZeroA: 0 or 1 from row parity
v_lshlrev_b32 v179, 2, v179                        // scaleZeroA: nibble shift
v_bfe_u32 v179, v[vgprG2LScaleZeroA+1], v179, 0x4  // scaleZeroA: z
v_cvt_f32_i32 v179, v179                           // scaleZeroA: c to f32
v_mov_b32 v178, v[vgprG2LA+28+0]                   // w4a16: save packed int4 dword 0
v_and_or_b32 v172, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v173, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v175, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v174, v178, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+0], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+1], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+2], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+3], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+16+0], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+16+1], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_and_or_b32 v172, v178, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v178, 20, v178                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v173, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v175, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v174, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+4], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+0], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+1], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+2], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+16+2], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+16+3], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_mov_b32 v178, v[vgprG2LA+28+1]                   // w4a16: save packed int4 dword 1
v_and_or_b32 v172, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v173, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v175, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v174, v178, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+0], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+1], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+2], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+3], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+16+4], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+16+5], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_and_or_b32 v172, v178, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v178, 20, v178                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v173, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v175, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v174, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+4], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+0], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+1], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+2], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+16+6], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+16+7], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_mov_b32 v178, v[vgprG2LA+28+2]                   // w4a16: save packed int4 dword 2
v_and_or_b32 v172, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v173, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v175, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v174, v178, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+0], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+1], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+2], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+3], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+16+8], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+16+9], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_and_or_b32 v172, v178, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v178, 20, v178                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v173, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v175, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v174, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+4], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+0], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+1], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+2], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+16+10], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+16+11], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_mov_b32 v178, v[vgprG2LA+28+3]                   // w4a16: save packed int4 dword 3
v_and_or_b32 v172, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #0 -> f32 2^23 + n
v_and_or_b32 v173, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #1 -> f32 2^19 + n
v_and_or_b32 v175, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #2 -> f32 2^15 + n
v_and_or_b32 v174, v178, 0xf000, s[sgprScaleALiftMagic+3] // w4a16: int4 #3 -> f32 2^11 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+0], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+1], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+2], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+3], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+16+12], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+16+13], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_and_or_b32 v172, v178, 0xf0000, s[sgprScaleALiftMagic+4] // w4a16: int4 #4 -> f32 2^7 + n
v_lshrrev_b32 v178, 20, v178                       // w4a16: nibbles 5-7 down to 0-2
v_and_or_b32 v173, v178, 0xf, s[sgprScaleALiftMagic+0] // w4a16: int4 #5 -> f32 2^23 + n
v_and_or_b32 v175, v178, 0xf0, s[sgprScaleALiftMagic+1] // w4a16: int4 #6 -> f32 2^19 + n
v_and_or_b32 v174, v178, 0xf00, s[sgprScaleALiftMagic+2] // w4a16: int4 #7 -> f32 2^15 + n
v_dual_subrev_f32 v172, s[sgprScaleALiftMagic+4], v172 :: v_dual_subrev_f32 v173, s[sgprScaleALiftMagic+0], v173 // w4a16: n - c, exact
v_dual_subrev_f32 v175, s[sgprScaleALiftMagic+1], v175 :: v_dual_subrev_f32 v174, s[sgprScaleALiftMagic+2], v174 // w4a16: n - c, exact
v_sub_f32 v172, v172, v179                         // w4a16: - zero-point, exact
v_sub_f32 v173, v173, v179                         // w4a16: - zero-point, exact
v_sub_f32 v175, v175, v179                         // w4a16: - zero-point, exact
v_sub_f32 v174, v174, v179                         // w4a16: - zero-point, exact
v_perm_b32 v[vgprG2LA+16+14], v173, v172, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
v_perm_b32 v[vgprG2LA+16+15], v174, v175, 0x07060302 // w4a16: pack 2 bf16 (n - c), exact
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+0], v[vgprG2LB+0], 0 // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+16], v[vgprG2LB+0], 0 // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+0], v[vgprG2LB+16], 0 // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+16], v[vgprG2LB+16], 0 // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+0], v[vgprG2LB+32], 0 // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+16], v[vgprG2LB+32], 0 // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+0], v[vgprG2LB+48], 0 // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+16], v[vgprG2LB+48], 0 // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+0], v[vgprG2LB+64], 0 // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+16], v[vgprG2LB+64], 0 // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+1], v[vgprG2LB+1], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+17], v[vgprG2LB+1], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+1], v[vgprG2LB+17], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+17], v[vgprG2LB+17], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+1], v[vgprG2LB+33], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+17], v[vgprG2LB+33], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+1], v[vgprG2LB+49], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+17], v[vgprG2LB+49], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+1], v[vgprG2LB+65], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+17], v[vgprG2LB+65], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+2], v[vgprG2LB+2], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+18], v[vgprG2LB+2], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+2], v[vgprG2LB+18], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+18], v[vgprG2LB+18], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+2], v[vgprG2LB+34], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+18], v[vgprG2LB+34], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+2], v[vgprG2LB+50], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+18], v[vgprG2LB+50], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+2], v[vgprG2LB+66], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+18], v[vgprG2LB+66], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+3], v[vgprG2LB+3], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+19], v[vgprG2LB+3], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+3], v[vgprG2LB+19], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+19], v[vgprG2LB+19], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+3], v[vgprG2LB+35], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+19], v[vgprG2LB+35], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+3], v[vgprG2LB+51], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+19], v[vgprG2LB+51], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+3], v[vgprG2LB+67], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+19], v[vgprG2LB+67], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+4], v[vgprG2LB+4], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+20], v[vgprG2LB+4], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+4], v[vgprG2LB+20], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+20], v[vgprG2LB+20], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+4], v[vgprG2LB+36], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+20], v[vgprG2LB+36], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+4], v[vgprG2LB+52], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+20], v[vgprG2LB+52], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+4], v[vgprG2LB+68], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+20], v[vgprG2LB+68], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+5], v[vgprG2LB+5], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+21], v[vgprG2LB+5], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+5], v[vgprG2LB+21], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+21], v[vgprG2LB+21], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+5], v[vgprG2LB+37], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+21], v[vgprG2LB+37], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+5], v[vgprG2LB+53], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+21], v[vgprG2LB+53], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+5], v[vgprG2LB+69], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+21], v[vgprG2LB+69], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+6], v[vgprG2LB+6], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+22], v[vgprG2LB+6], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+6], v[vgprG2LB+22], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+22], v[vgprG2LB+22], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+6], v[vgprG2LB+38], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+22], v[vgprG2LB+38], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+6], v[vgprG2LB+54], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+22], v[vgprG2LB+54], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+6], v[vgprG2LB+70], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+22], v[vgprG2LB+70], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+7], v[vgprG2LB+7], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+23], v[vgprG2LB+7], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+7], v[vgprG2LB+23], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+23], v[vgprG2LB+23], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+7], v[vgprG2LB+39], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+23], v[vgprG2LB+39], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+7], v[vgprG2LB+55], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+23], v[vgprG2LB+55], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+7], v[vgprG2LB+71], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+23], v[vgprG2LB+71], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+8], v[vgprG2LB+8], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+24], v[vgprG2LB+8], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+8], v[vgprG2LB+24], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+24], v[vgprG2LB+24], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+8], v[vgprG2LB+40], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+24], v[vgprG2LB+40], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+8], v[vgprG2LB+56], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+24], v[vgprG2LB+56], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+8], v[vgprG2LB+72], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+24], v[vgprG2LB+72], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+9], v[vgprG2LB+9], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+25], v[vgprG2LB+9], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+9], v[vgprG2LB+25], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+25], v[vgprG2LB+25], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+9], v[vgprG2LB+41], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+25], v[vgprG2LB+41], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+9], v[vgprG2LB+57], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+25], v[vgprG2LB+57], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+9], v[vgprG2LB+73], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+25], v[vgprG2LB+73], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+10], v[vgprG2LB+10], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+26], v[vgprG2LB+10], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+10], v[vgprG2LB+26], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+26], v[vgprG2LB+26], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+10], v[vgprG2LB+42], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+26], v[vgprG2LB+42], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+10], v[vgprG2LB+58], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+26], v[vgprG2LB+58], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+10], v[vgprG2LB+74], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+26], v[vgprG2LB+74], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+11], v[vgprG2LB+11], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+27], v[vgprG2LB+11], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+11], v[vgprG2LB+27], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+27], v[vgprG2LB+27], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+11], v[vgprG2LB+43], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+27], v[vgprG2LB+43], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+11], v[vgprG2LB+59], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+27], v[vgprG2LB+59], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+11], v[vgprG2LB+75], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+27], v[vgprG2LB+75], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+12], v[vgprG2LB+12], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+28], v[vgprG2LB+12], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+12], v[vgprG2LB+28], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+28], v[vgprG2LB+28], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+12], v[vgprG2LB+44], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+28], v[vgprG2LB+44], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+12], v[vgprG2LB+60], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+28], v[vgprG2LB+60], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+12], v[vgprG2LB+76], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+28], v[vgprG2LB+76], v[vgprDot2PS+9] // Dot2PS[9] slot 0
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
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+13], v[vgprG2LB+13], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+29], v[vgprG2LB+13], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+13], v[vgprG2LB+29], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+29], v[vgprG2LB+29], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+13], v[vgprG2LB+45], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+29], v[vgprG2LB+45], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+13], v[vgprG2LB+61], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+29], v[vgprG2LB+61], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+13], v[vgprG2LB+77], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+29], v[vgprG2LB+77], v[vgprDot2PS+9] // Dot2PS[9] slot 0
s_setprio 0                                        // Reset priority after macs


/* iter 14 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */

/* local read a */

/* local read b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+14], v[vgprG2LB+14], v[vgprDot2PS+0] // Dot2PS[0] slot 0
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+30], v[vgprG2LB+14], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+14], v[vgprG2LB+30], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+30], v[vgprG2LB+30], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+14], v[vgprG2LB+46], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+30], v[vgprG2LB+46], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+14], v[vgprG2LB+62], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+30], v[vgprG2LB+62], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+14], v[vgprG2LB+78], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+30], v[vgprG2LB+78], v[vgprDot2PS+9] // Dot2PS[9] slot 0
s_setprio 0                                        // Reset priority after macs


/* iter 15 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
// Component.MAC.FMA_BF16_HPA_DOT2
v_dot2_f32_bf16 v[vgprDot2PS+0], v[vgprG2LA+15], v[vgprG2LB+15], v[vgprDot2PS+0] // Dot2PS[0] slot 0
v_fma_f32 v[vgprValuC+0+0], v[vgprG2LScaleA+0], v[vgprDot2PS+0], v[vgprValuC+0+0] // ValuC[0] += s * Dot2PS
s_setprio 1                                        // Raise priority while processing macs
v_dot2_f32_bf16 v[vgprDot2PS+1], v[vgprG2LA+31], v[vgprG2LB+15], v[vgprDot2PS+1] // Dot2PS[1] slot 0
v_fma_f32 v[vgprValuC+1+0], v[vgprG2LScaleA+1], v[vgprDot2PS+1], v[vgprValuC+1+0] // ValuC[1] += s * Dot2PS
v_dot2_f32_bf16 v[vgprDot2PS+2], v[vgprG2LA+15], v[vgprG2LB+31], v[vgprDot2PS+2] // Dot2PS[2] slot 0
v_fma_f32 v[vgprValuC+0+2], v[vgprG2LScaleA+0], v[vgprDot2PS+2], v[vgprValuC+0+2] // ValuC[2] += s * Dot2PS
v_dot2_f32_bf16 v[vgprDot2PS+3], v[vgprG2LA+31], v[vgprG2LB+31], v[vgprDot2PS+3] // Dot2PS[3] slot 0
v_fma_f32 v[vgprValuC+1+2], v[vgprG2LScaleA+1], v[vgprDot2PS+3], v[vgprValuC+1+2] // ValuC[3] += s * Dot2PS
v_dot2_f32_bf16 v[vgprDot2PS+4], v[vgprG2LA+15], v[vgprG2LB+47], v[vgprDot2PS+4] // Dot2PS[4] slot 0
v_fma_f32 v[vgprValuC+0+4], v[vgprG2LScaleA+0], v[vgprDot2PS+4], v[vgprValuC+0+4] // ValuC[4] += s * Dot2PS
v_dot2_f32_bf16 v[vgprDot2PS+5], v[vgprG2LA+31], v[vgprG2LB+47], v[vgprDot2PS+5] // Dot2PS[5] slot 0
v_fma_f32 v[vgprValuC+1+4], v[vgprG2LScaleA+1], v[vgprDot2PS+5], v[vgprValuC+1+4] // ValuC[5] += s * Dot2PS
v_dot2_f32_bf16 v[vgprDot2PS+6], v[vgprG2LA+15], v[vgprG2LB+63], v[vgprDot2PS+6] // Dot2PS[6] slot 0
v_fma_f32 v[vgprValuC+0+6], v[vgprG2LScaleA+0], v[vgprDot2PS+6], v[vgprValuC+0+6] // ValuC[6] += s * Dot2PS
v_dot2_f32_bf16 v[vgprDot2PS+7], v[vgprG2LA+31], v[vgprG2LB+63], v[vgprDot2PS+7] // Dot2PS[7] slot 0
v_fma_f32 v[vgprValuC+1+6], v[vgprG2LScaleA+1], v[vgprDot2PS+7], v[vgprValuC+1+6] // ValuC[7] += s * Dot2PS
v_dot2_f32_bf16 v[vgprDot2PS+8], v[vgprG2LA+15], v[vgprG2LB+79], v[vgprDot2PS+8] // Dot2PS[8] slot 0
v_fma_f32 v[vgprValuC+0+8], v[vgprG2LScaleA+0], v[vgprDot2PS+8], v[vgprValuC+0+8] // ValuC[8] += s * Dot2PS
v_dot2_f32_bf16 v[vgprDot2PS+9], v[vgprG2LA+31], v[vgprG2LB+79], v[vgprDot2PS+9] // Dot2PS[9] slot 0
v_fma_f32 v[vgprValuC+1+8], v[vgprG2LScaleA+1], v[vgprDot2PS+9], v[vgprValuC+1+8] // ValuC[9] += s * Dot2PS
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
v_cndmask_b32 v[vgprGlobalReadOffsetA+1], BufferOOB, v[vgprGlobalReadOffsetA+1], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetScaleA+0], BufferOOB, v[vgprGlobalReadOffsetScaleA+0], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetScaleA+1], BufferOOB, v[vgprGlobalReadOffsetScaleA+1], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+0], BufferOOB, v[vgprGlobalReadOffsetB+0], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+1], BufferOOB, v[vgprGlobalReadOffsetB+1], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+2], BufferOOB, v[vgprGlobalReadOffsetB+2], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+3], BufferOOB, v[vgprGlobalReadOffsetB+3], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+4], BufferOOB, v[vgprGlobalReadOffsetB+4], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+5], BufferOOB, v[vgprGlobalReadOffsetB+5], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+6], BufferOOB, v[vgprGlobalReadOffsetB+6], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+7], BufferOOB, v[vgprGlobalReadOffsetB+7], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+8], BufferOOB, v[vgprGlobalReadOffsetB+8], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+9], BufferOOB, v[vgprGlobalReadOffsetB+9], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+10], BufferOOB, v[vgprGlobalReadOffsetB+10], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+11], BufferOOB, v[vgprGlobalReadOffsetB+11], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+12], BufferOOB, v[vgprGlobalReadOffsetB+12], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+13], BufferOOB, v[vgprGlobalReadOffsetB+13], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+14], BufferOOB, v[vgprGlobalReadOffsetB+14], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+15], BufferOOB, v[vgprGlobalReadOffsetB+15], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+16], BufferOOB, v[vgprGlobalReadOffsetB+16], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+17], BufferOOB, v[vgprGlobalReadOffsetB+17], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+18], BufferOOB, v[vgprGlobalReadOffsetB+18], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
v_cndmask_b32 v[vgprGlobalReadOffsetB+19], BufferOOB, v[vgprGlobalReadOffsetB+19], s[sgprDot2TailMask+0] // K tail: slot 0, lanes past K read zeros
s_mov_b32 s[sgprLoopCounterL], 1                   // K tail: one more iteration
s_branch label_LoopBeginL                          // K tail: masked iteration
label_Dot2TailDone:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/* Tail: add ValuA/B vgpr buffer [56...169) to pool */

/* Tail: add address/G2L vgpr [169...169) to pool */
label_Summation_End_0:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprShadowLimitA, UNDEF
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
.set sgprScaleALiftMagic, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprAddressScaleZeroA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprSrdScaleZeroA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprGlobalReadIncsB, UNDEF
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
v_add_f32 v[vgprValuC+6], v[vgprValuC+6], v[vgprValuC+6] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+7], v[vgprValuC+7], v[vgprValuC+7] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+8], v[vgprValuC+8], v[vgprValuC+8] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+9], v[vgprValuC+9], v[vgprValuC+9] row_shr:1 bound_ctrl:0 // Reduce 2 elements
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v[vgprValuC+0] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v[vgprValuC+1] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v[vgprValuC+2] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v[vgprValuC+3] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v[vgprValuC+4] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v[vgprValuC+5] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+6], v[vgprValuC+6], v[vgprValuC+6] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+7], v[vgprValuC+7], v[vgprValuC+7] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+8], v[vgprValuC+8], v[vgprValuC+8] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+9], v[vgprValuC+9], v[vgprValuC+9] row_shr:2 bound_ctrl:0 // Reduce 4 elements
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v[vgprValuC+0] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v[vgprValuC+1] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v[vgprValuC+2] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v[vgprValuC+3] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v[vgprValuC+4] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v[vgprValuC+5] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+6], v[vgprValuC+6], v[vgprValuC+6] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+7], v[vgprValuC+7], v[vgprValuC+7] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+8], v[vgprValuC+8], v[vgprValuC+8] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+9], v[vgprValuC+9], v[vgprValuC+9] row_shr:4 bound_ctrl:0 // Reduce 8 elements
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v[vgprValuC+0] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v[vgprValuC+1] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v[vgprValuC+2] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v[vgprValuC+3] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v[vgprValuC+4] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v[vgprValuC+5] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+6], v[vgprValuC+6], v[vgprValuC+6] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+7], v[vgprValuC+7], v[vgprValuC+7] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+8], v[vgprValuC+8], v[vgprValuC+8] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_add_f32 v[vgprValuC+9], v[vgprValuC+9], v[vgprValuC+9] row_shr:8 bound_ctrl:0 // Reduce 16 elements
v_xor_b32 v56, 16, v[vgprSerial]                   // lane ^ 16
v_lshlrev_b32 v56, 2, v56                          // bpermute byte address
ds_bpermute_b32 v57, v56, v[vgprValuC+0]
ds_bpermute_b32 v58, v56, v[vgprValuC+1]
ds_bpermute_b32 v59, v56, v[vgprValuC+2]
ds_bpermute_b32 v60, v56, v[vgprValuC+3]
ds_bpermute_b32 v61, v56, v[vgprValuC+4]
ds_bpermute_b32 v62, v56, v[vgprValuC+5]
ds_bpermute_b32 v63, v56, v[vgprValuC+6]
ds_bpermute_b32 v64, v56, v[vgprValuC+7]
ds_bpermute_b32 v65, v56, v[vgprValuC+8]
ds_bpermute_b32 v66, v56, v[vgprValuC+9]
s_waitcnt lgkmcnt(0)                               // wait for bpermute
v_add_f32 v[vgprValuC+0], v[vgprValuC+0], v57      // Reduce 32 elements
v_add_f32 v[vgprValuC+1], v[vgprValuC+1], v58      // Reduce 32 elements
v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v59      // Reduce 32 elements
v_add_f32 v[vgprValuC+3], v[vgprValuC+3], v60      // Reduce 32 elements
v_add_f32 v[vgprValuC+4], v[vgprValuC+4], v61      // Reduce 32 elements
v_add_f32 v[vgprValuC+5], v[vgprValuC+5], v62      // Reduce 32 elements
v_add_f32 v[vgprValuC+6], v[vgprValuC+6], v63      // Reduce 32 elements
v_add_f32 v[vgprValuC+7], v[vgprValuC+7], v64      // Reduce 32 elements
v_add_f32 v[vgprValuC+8], v[vgprValuC+8], v65      // Reduce 32 elements
v_add_f32 v[vgprValuC+9], v[vgprValuC+9], v66      // Reduce 32 elements


/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_mov_b32 v58, 0                                   // rowStart vgpr
v_mov_b32 v59, 0                                   // rowStart vgpr

s_mul_i32 s44, 0x2, s[sgprWorkGroup0]              // s44 = wg0*MT0
v_mov_b32 v56, s44                                 // coord0 = wg0*MT0
s_mul_i32 s46, 0x5, s[sgprWorkGroup1]              // <- wg1*MT1
v_mov_b32 v57, s46                                 // coord1 = wg1*MT1

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

/* Edge/NonEdge store path check (N (isSize1)): Size % 5 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s47, s[sgprSizeJ], 1717986919         // STATIC_DIV: divisor=5
s_mul_i32 s46, s[sgprSizeJ], 1717986919            // tmp = dividend * magic
s_lshr_b64 s[46:47], s[46:47], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s45, s46                                 // quotient
s_mul_i32 s46, s45, 5                              // quotient*divisor
s_sub_u32 s44, s[sgprSizeJ], s46                   // rReg = dividend - quotient*divisor
s_add_u32 s45, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s45                // wg1 >= nwg1-1
s_cselect_b32 s44, s44, 0                          // set rem
s_cmpk_gt_u32 s44, 0                               // rem > 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B0_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=10 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v63, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_lshl_u32 v68, v59, v56, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=56, coord0Vgpr=56 (multiple bpe)
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v68, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v65, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v66, 0x7fff0000                          // fp32 Nan
v_mov_b32 v67, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s8, v[vgprValuC+0], v[vgprValuC+0]     // check Nan
v_bfe_u32 v64, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+0], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v64, v66, s8
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_b16 v0, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+1], v[vgprValuC+1]     // check Nan
v_bfe_u32 v64, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+1], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v64, v66, s8
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
buffer_store_b16 v1, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
v_cmp_u_f32 s8, v[vgprValuC+2], v[vgprValuC+2]     // check Nan
v_bfe_u32 v64, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+2], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v64, v66, s8
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v2, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+3], v[vgprValuC+3]     // check Nan
v_bfe_u32 v64, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+3], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v64, v66, s8
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
buffer_store_b16 v3, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
v_cmp_u_f32 s8, v[vgprValuC+4], v[vgprValuC+4]     // check Nan
v_bfe_u32 v64, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+4], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v64, v66, s8
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v4, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+5], v[vgprValuC+5]     // check Nan
v_bfe_u32 v64, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+5], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v64, v66, s8
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_b16 v5, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
v_cmp_u_f32 s8, v[vgprValuC+6], v[vgprValuC+6]     // check Nan
v_bfe_u32 v64, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+6], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v64, v66, s8
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v6, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+7], v[vgprValuC+7]     // check Nan
v_bfe_u32 v64, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+7], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v64, v66, s8
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_b16 v7, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
v_cmp_u_f32 s8, v[vgprValuC+8], v[vgprValuC+8]     // check Nan
v_bfe_u32 v64, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+8], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v64, v66, s8
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v8, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s8, v[vgprValuC+9], v[vgprValuC+9]     // check Nan
v_bfe_u32 v64, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+9], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v64, v66, s8
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_b16 v9, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
s_branch label_GW_End                              // jump to end
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=186 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v63, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v68, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v68, v63, v68, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v68, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v69, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v69, v63, v69, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v69, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v57, vcc_lo, v57, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v58, v58, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v59, v59, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v70, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v70, v63, v70, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v70, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v71, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v63, v71, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v71, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v57, vcc_lo, v57, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v58, v58, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v59, v59, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v72, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v72, v63, v72, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v72, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v73, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v63, v73, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v73, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v57, vcc_lo, v57, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v58, v58, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v59, v59, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v74, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v74, v63, v74, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v74, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v75, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v63, v75, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v75, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v57, vcc_lo, v57, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v58, v58, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v59, v59, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v76, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v76, v63, v76, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v76, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v77, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v63, v77, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v77, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v65, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v66, 0x7fff0000                          // fp32 Nan
v_mov_b32 v67, 0x7fff                              // rounding bias for bfloat16
v_cmp_u_f32 s44, v[vgprValuC+0], v[vgprValuC+0]    // check Nan
v_bfe_u32 v64, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+0], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v64, v66, s44
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_b16 v0, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+1], v[vgprValuC+1]    // check Nan
v_bfe_u32 v64, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+1], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v64, v66, s44
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
buffer_store_b16 v1, v69, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+2], v[vgprValuC+2]    // check Nan
v_bfe_u32 v64, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+2], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v64, v66, s44
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
buffer_store_b16 v2, v70, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+3], v[vgprValuC+3]    // check Nan
v_bfe_u32 v64, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+3], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v64, v66, s44
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
buffer_store_b16 v3, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+4], v[vgprValuC+4]    // check Nan
v_bfe_u32 v64, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+4], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v64, v66, s44
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
buffer_store_b16 v4, v72, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+5], v[vgprValuC+5]    // check Nan
v_bfe_u32 v64, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+5], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v64, v66, s44
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_b16 v5, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+6], v[vgprValuC+6]    // check Nan
v_bfe_u32 v64, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+6], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v64, v66, s44
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
buffer_store_b16 v6, v74, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+7], v[vgprValuC+7]    // check Nan
v_bfe_u32 v64, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+7], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v64, v66, s44
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_b16 v7, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+8], v[vgprValuC+8]    // check Nan
v_bfe_u32 v64, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+8], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v64, v66, s44
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
buffer_store_b16 v8, v76, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_u_f32 s44, v[vgprValuC+9], v[vgprValuC+9]    // check Nan
v_bfe_u32 v64, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+9], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v64, v66, s44
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_b16 v9, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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

/* Edge/NonEdge store path check (N (isSize1)): Size % 5 > 0 -> Edge store; else -> NonEdge store */
s_mul_hi_u32 s47, s[sgprSizeJ], 1717986919         // STATIC_DIV: divisor=5
s_mul_i32 s46, s[sgprSizeJ], 1717986919            // tmp = dividend * magic
s_lshr_b64 s[46:47], s[46:47], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s45, s46                                 // quotient
s_mul_i32 s46, s45, 5                              // quotient*divisor
s_sub_u32 s44, s[sgprSizeJ], s46                   // rReg = dividend - quotient*divisor
s_add_u32 s45, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s45                // wg1 >= nwg1-1
s_cselect_b32 s44, s44, 0                          // set rem
s_cmpk_gt_u32 s44, 0                               // rem > 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then       // jump if edges required
label_GW_B1_FD0_VW1_GSU1_NonEdge:

/* edge=0, allocate 1 sgpr. perBatchTmpS=1 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=184 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v63, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v69, v58, v56, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=56, coord0Vgpr=56 (multiple bpe)
buffer_load_d16_b16 v70, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
buffer_load_d16_b16 v71, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:2 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_d16_b16 v72, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
buffer_load_d16_b16 v73, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:2 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_d16_b16 v74, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
buffer_load_d16_b16 v75, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:2 // load C
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_d16_b16 v76, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
buffer_load_d16_b16 v77, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:2 // load C
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
s_lshl_b32 s8, s[sgprStrideC1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_load_d16_b16 v78, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
buffer_load_d16_b16 v79, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:2 // load C
v_add_lshl_u32 v68, v59, v56, 1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=56, coord0Vgpr=56 (multiple bpe)
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v68, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)

/* apply mask, calc new C and issue writes */
v_mov_b32 v65, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v66, 0x7fff0000                          // fp32 Nan
v_mov_b32 v67, 0x7fff                              // rounding bias for bfloat16

s_waitcnt vmcnt(9)                                 // vlcnt(9) = 10 - 1 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v70                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+0], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+0], v[vgprValuC+0]     // check Nan
v_bfe_u32 v64, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+0], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v64, v66, s8
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_b16 v0, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(8)                                 // vlcnt(8) = 10 - 2 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v71                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+1], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+1], v[vgprValuC+1]     // check Nan
v_bfe_u32 v64, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+1], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v64, v66, s8
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
buffer_store_b16 v1, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D

s_waitcnt vmcnt(7)                                 // vlcnt(7) = 10 - 3 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v72                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+2], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+2], v[vgprValuC+2]     // check Nan
v_bfe_u32 v64, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+2], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v64, v66, s8
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v2, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(6)                                 // vlcnt(6) = 10 - 4 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v73                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+3], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+3], v[vgprValuC+3]     // check Nan
v_bfe_u32 v64, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+3], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v64, v66, s8
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
buffer_store_b16 v3, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D

s_waitcnt vmcnt(5)                                 // vlcnt(5) = 10 - 5 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v74                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+4], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+4], v[vgprValuC+4]     // check Nan
v_bfe_u32 v64, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+4], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v64, v66, s8
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v4, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(4)                                 // vlcnt(4) = 10 - 6 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v75                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+5], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+5], v[vgprValuC+5]     // check Nan
v_bfe_u32 v64, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+5], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v64, v66, s8
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_b16 v5, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D

s_waitcnt vmcnt(3)                                 // vlcnt(3) = 10 - 7 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v76                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+6], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+6], v[vgprValuC+6]     // check Nan
v_bfe_u32 v64, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+6], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v64, v66, s8
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v6, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(2)                                 // vlcnt(2) = 10 - 8 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v77                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+7], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+7], v[vgprValuC+7]     // check Nan
v_bfe_u32 v64, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+7], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v64, v66, s8
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_b16 v7, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D

s_waitcnt vmcnt(1)                                 // vlcnt(1) = 10 - 9 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v78                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+8], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+8], v[vgprValuC+8]     // check Nan
v_bfe_u32 v64, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+8], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v64, v66, s8
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
s_lshl_b32 s8, s[sgprStrideD1J], 1                 // incToNextRow(1): Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8         // incToNextRow(1): gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow(1): gra SRD += inc(upper)
buffer_store_b16 v8, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(0)                                 // vlcnt(0) = 10 - 10 (beta) (interleaved)
v_lshlrev_b32 v60, 16, v79                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+9], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s8, v[vgprValuC+9], v[vgprValuC+9]     // check Nan
v_bfe_u32 v64, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+9], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v64, v66, s8
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_b16 v9, v68, s[sgprSrdD:sgprSrdD+3], 0 offen offset:2 // store D
s_branch label_GW_End                              // jump to end
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:

/* edge=1, allocate 3 sgpr. perBatchTmpS=2 perBatchMaskS=1 perElementMaskS=0 elementsPerBatch=92 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,1,0,0:vw1); (1,0,0,0:vw1); (1,1,0,0:vw1); (2,0,0,0:vw1); (2,1,0,0:vw1); (3,0,0,0:vw1); (3,1,0,0:vw1); (4,0,0,0:vw1); (4,1,0,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v63, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v69, v58, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v69, v63, v69, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v68, v69, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v69, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v69, v63, v69, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v69, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v71, v58, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v63, v71, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v70, v71, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v71, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v71, v63, v71, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v71, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v57, vcc_lo, v57, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v58, v58, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v59, v59, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v73, v58, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v63, v73, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v72, v73, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v73, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v73, v63, v73, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v73, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v75, v58, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v63, v75, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v74, v75, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v75, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v75, v63, v75, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v75, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v57, vcc_lo, v57, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v58, v58, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v59, v59, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v77, v58, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v63, v77, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v76, v77, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v77, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v77, v63, v77, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v77, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v79, v58, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v79, v63, v79, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v78, v79, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v79, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v79, v63, v79, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v79, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
v_add_co_u32 v57, vcc_lo, v57, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v58, v58, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v59, v59, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v81, v58, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v63, v81, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v80, v81, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v81, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v81, v63, v81, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v81, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v83, v58, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v63, v83, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v82, v83, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v83, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v83, v63, v83, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v83, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(4,0,0,0) */
v_add_co_u32 v57, vcc_lo, v57, 1                   // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_nc_u32 v58, v58, s[sgprStrideC1J]            // ROWINC- Move cinRowPtr to next row
v_add_nc_u32 v59, v59, s[sgprStrideD1J]            // Move coutRowPtrD to next row
v_cmp_lt_u32 s44, v56, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v85, v58, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v85, v63, v85, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v84, v85, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v85, v59, v56, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v85, v63, v85, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v85, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask
/* (d1,vc1,d0,vc0)=(4,0,1,0) */
v_add_co_u32 v60, vcc_lo, v56, 1                   // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s44, v60, s[sgprSizeI]                // coord0 < size0
v_cmp_lt_u32 s46, v57, s[sgprSizeJ]                // coord1 < size1
s_and_b32 s46, s44, s46                            // in0 && in1
v_add_lshl_u32 v87, v58, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v63, v87, s46                   // LDC clip if OOB. offset
buffer_load_d16_b16 v86, v87, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v87, v59, v60, 1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr (multiple bpe)
v_cndmask_b32 v87, v63, v87, s46                   // LDD clip if OOB. offset
v_and_b32 v62, 31, v[vgprSerial]                   // kidx = Serial % NumWaveSplitK
v_cmp_ne_u32 vcc_lo, 31, v62                       // kidx == NumWaveSplitK-1
s_mov_b32 exec_lo, vcc_lo                          // kidx == NumWaveSplitK-1
v_mov_b32 v87, v63                                 // LDD clip if kidx != NumWaveSplitK-1.
s_mov_b32 exec_lo, -1                              // Reset exec mask

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (3, 0, 0, 0), (3, 1, 0, 0), (4, 0, 0, 0), (4, 1, 0, 0)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha (pk)
v_mul_f32 v[vgprValuC+0+1], s[sgprAlpha], v[vgprValuC+0+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha (pk)
v_mul_f32 v[vgprValuC+2+1], s[sgprAlpha], v[vgprValuC+2+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha (pk)
v_mul_f32 v[vgprValuC+4+1], s[sgprAlpha], v[vgprValuC+4+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha (pk)
v_mul_f32 v[vgprValuC+6+1], s[sgprAlpha], v[vgprValuC+6+1] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha (pk)
v_mul_f32 v[vgprValuC+8+1], s[sgprAlpha], v[vgprValuC+8+1] // *= alpha (pk)
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v65, 0xffff0000                          // mask for pack two bfloat16 element to 32bit
v_mov_b32 v66, 0x7fff0000                          // fp32 Nan
v_mov_b32 v67, 0x7fff                              // rounding bias for bfloat16
v_lshlrev_b32 v60, 16, v68                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+0], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+0], v[vgprValuC+0]    // check Nan
v_bfe_u32 v64, v[vgprValuC+0], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+0], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+0], v64, v66, s44
v_lshrrev_b32 v0, 16, v[vgprValuC+0]               // convert C to bf16
buffer_store_b16 v0, v69, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v70                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+1], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+1], v[vgprValuC+1]    // check Nan
v_bfe_u32 v64, v[vgprValuC+1], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+1], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+1], v64, v66, s44
v_lshrrev_b32 v1, 16, v[vgprValuC+1]               // convert C to bf16
buffer_store_b16 v1, v71, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v72                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+2], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+2], v[vgprValuC+2]    // check Nan
v_bfe_u32 v64, v[vgprValuC+2], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+2], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+2], v64, v66, s44
v_lshrrev_b32 v2, 16, v[vgprValuC+2]               // convert C to bf16
buffer_store_b16 v2, v73, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v74                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+3], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+3], v[vgprValuC+3]    // check Nan
v_bfe_u32 v64, v[vgprValuC+3], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+3], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+3], v64, v66, s44
v_lshrrev_b32 v3, 16, v[vgprValuC+3]               // convert C to bf16
buffer_store_b16 v3, v75, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v76                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+4], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+4], v[vgprValuC+4]    // check Nan
v_bfe_u32 v64, v[vgprValuC+4], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+4], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+4], v64, v66, s44
v_lshrrev_b32 v4, 16, v[vgprValuC+4]               // convert C to bf16
buffer_store_b16 v4, v77, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v78                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+5], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+5], v[vgprValuC+5]    // check Nan
v_bfe_u32 v64, v[vgprValuC+5], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+5], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+5], v64, v66, s44
v_lshrrev_b32 v5, 16, v[vgprValuC+5]               // convert C to bf16
buffer_store_b16 v5, v79, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v80                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+6], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+6], v[vgprValuC+6]    // check Nan
v_bfe_u32 v64, v[vgprValuC+6], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+6], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+6], v64, v66, s44
v_lshrrev_b32 v6, 16, v[vgprValuC+6]               // convert C to bf16
buffer_store_b16 v6, v81, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v82                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+7], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+7], v[vgprValuC+7]    // check Nan
v_bfe_u32 v64, v[vgprValuC+7], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+7], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+7], v64, v66, s44
v_lshrrev_b32 v7, 16, v[vgprValuC+7]               // convert C to bf16
buffer_store_b16 v7, v83, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v84                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+8], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+8], v[vgprValuC+8]    // check Nan
v_bfe_u32 v64, v[vgprValuC+8], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+8], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+8], v64, v66, s44
v_lshrrev_b32 v8, 16, v[vgprValuC+8]               // convert C to bf16
buffer_store_b16 v8, v85, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_lshlrev_b32 v60, 16, v86                         // cvt bf16 to fp32. 
v_fmac_f32 v[vgprValuC+9], v60, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_cmp_u_f32 s44, v[vgprValuC+9], v[vgprValuC+9]    // check Nan
v_bfe_u32 v64, v[vgprValuC+9], 16, 1               // Non-Nan case: store lsb of bf16
v_add3_u32 v64, v[vgprValuC+9], v64, v67           // Non-Nan case: add lsb and the increment for rounding
v_cndmask_b32 v[vgprValuC+9], v64, v66, s44
v_lshrrev_b32 v9, 16, v[vgprValuC+9]               // convert C to bf16
buffer_store_b16 v9, v87, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
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
s_lshr_b32 s77, s[sgprWGM], 0x16                   // Get CU_Count
/* only remap WGs in the range */
s_lshr_b32 s73, s66, s76
s_lshl_b32 s73, s73, s76
s_cmp_ge_u32 s[sgprWorkGroup0], s73
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s77, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s73, s[sgprWorkGroup0], s76
s_bfm_b32 s74, s76, 0
s_and_b32 s74, s[sgprWorkGroup0], s74
s_lshr_b32 s75, s66, s76
s_mul_i32 s74, s74, s75
s_add_u32 s[sgprWorkGroup0], s73, s74
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f64_u32 v[6:7], s77                          // s73 = s[sgprWorkGroup0] / s77
v_rcp_f64 v[6:7], v[6:7]                           // s73 = s[sgprWorkGroup0] / s77
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s73 = s[sgprWorkGroup0] / s77
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s73 = s[sgprWorkGroup0] / s77
v_cvt_u32_f64 v6, v[6:7]                           // s73 = s[sgprWorkGroup0] / s77
v_mul_lo_u32 v7, v6, s77                           // s73 = s[sgprWorkGroup0] / s77
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s73 = s[sgprWorkGroup0] / s77
v_cmp_ge_u32 vcc_lo, v8, s77                       // s73 = s[sgprWorkGroup0] / s77
s_mov_b32 exec_lo, vcc_lo                          // s73 = s[sgprWorkGroup0] / s77
v_add_nc_u32 v6, v6, 1                             // s73 = s[sgprWorkGroup0] / s77
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s77                           // s73 = s[sgprWorkGroup0] / s77
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s73 = s[sgprWorkGroup0] / s77
v_readfirstlane_b32 s73, v6                        // quotient
v_readfirstlane_b32 s74, v8                        // remainder
s_mul_i32 s73, s73, s77
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s74, s74, s76
/* temp0 = temp0 + temp1 */
s_add_u32 s73, s73, s74
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f64_u32 v[6:7], s77                          // s74 = s66 / s77
v_rcp_f64 v[6:7], v[6:7]                           // s74 = s66 / s77
v_cvt_f64_u32 v[8:9], s66                          // s74 = s66 / s77
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s74 = s66 / s77
v_cvt_u32_f64 v6, v[6:7]                           // s74 = s66 / s77
v_mul_lo_u32 v7, v6, s77                           // s74 = s66 / s77
v_sub_nc_u32 v8, s66, v7                           // s74 = s66 / s77
v_cmp_ge_u32 vcc_lo, v8, s77                       // s74 = s66 / s77
s_mov_b32 exec_lo, vcc_lo                          // s74 = s66 / s77
v_add_nc_u32 v6, v6, 1                             // s74 = s66 / s77
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s74, v6                        // quotient
s_mul_i32 s74, s74, s77
s_sub_u32 s75, s66, s74
s_cmp_gt_u32 s[sgprWorkGroup0], s74
s_cselect_b32 s74, s75, s77
s_lshr_b32 s74, s74, s76
s_bfm_b32 s75, s76, 0
s_and_b32 s75, s[sgprWorkGroup0], s75
s_mul_i32 s74, s74, s75
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s73, s74
s_branch label_skip_WGMXCC
.set sgprNumWorkGroups0, 16
.set sgprNumWorkGroups1, 17
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
label_RemapWG:
s_mul_i32 s66, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_cmp_lt_u32 s[sgprWorkGroup0], s66                // wg2 == 0?
s_cbranch_scc1 label_RemapWG2Done
v_cvt_f32_u32 v6, s66                              // s66 = s[sgprWorkGroup0] / s66
v_rcp_iflag_f32 v6, v6                             // s66 = s[sgprWorkGroup0] / s66
v_mov_b32 v7, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v6, v7, v6                               // 2^32 / divisor
v_cvt_u32_f32 v6, v6                               // s66 = s[sgprWorkGroup0] / s66
v_mul_lo_u32 v7, v6, s66                           // refine 2^32 / divisor
v_sub_nc_u32 v7, 0, v7                             // refine 2^32 / divisor
v_mul_hi_u32 v7, v6, v7                            // refine 2^32 / divisor
v_add_nc_u32 v6, v6, v7                            // refine 2^32 / divisor
v_mul_hi_u32 v6, s[sgprWorkGroup0], v6             // s66 = s[sgprWorkGroup0] / s66
v_mul_lo_u32 v7, v6, s66                           // s66 = s[sgprWorkGroup0] / s66
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s66 = s[sgprWorkGroup0] / s66
v_cmp_ge_u32 vcc_lo, v7, s66                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v6, 1, v6                             // quotient + 1
v_sub_nc_u32 v7, v7, s66                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v7, s66                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v6, 1, v6                             // quotient + 1
v_sub_nc_u32 v7, v7, s66                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s66, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s66
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s66, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s66, s66, s[sgprWorkGroup2]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s66
label_RemapWG2Done:
s_cmp_lt_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // wg1 == 0?
s_cbranch_scc1 label_RemapWG1Done
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mov_b32 v7, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v6, v7, v6                               // 2^32 / divisor
v_cvt_u32_f32 v6, v6                               // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_lo_u32 v7, v6, s[sgprNumWorkGroups0]         // refine 2^32 / divisor
v_sub_nc_u32 v7, 0, v7                             // refine 2^32 / divisor
v_mul_hi_u32 v7, v6, v7                            // refine 2^32 / divisor
v_add_nc_u32 v6, v6, v7                            // refine 2^32 / divisor
v_mul_hi_u32 v6, s[sgprWorkGroup0], v6             // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_lo_u32 v7, v6, s[sgprNumWorkGroups0]         // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_ge_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v6, 1, v6                             // quotient + 1
v_sub_nc_u32 v7, v7, s[sgprNumWorkGroups0]         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v6, 1, v6                             // quotient + 1
v_sub_nc_u32 v7, v7, s[sgprNumWorkGroups0]         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s66, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s66
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s66, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s66
label_RemapWG1Done:
s_branch label_RemapWGDone
.set sgprNumWorkGroups0, 16
.set sgprNumWorkGroups1, 17
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
label_RemapWG_1:
s_mul_i32 s66, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_cmp_lt_u32 s[sgprWorkGroup0], s66                // wg2 == 0?
s_cbranch_scc1 label_RemapWG2Done_1
v_cvt_f32_u32 v6, s66                              // s66 = s[sgprWorkGroup0] / s66
v_rcp_iflag_f32 v6, v6                             // s66 = s[sgprWorkGroup0] / s66
v_mov_b32 v7, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v6, v7, v6                               // 2^32 / divisor
v_cvt_u32_f32 v6, v6                               // s66 = s[sgprWorkGroup0] / s66
v_mul_lo_u32 v7, v6, s66                           // refine 2^32 / divisor
v_sub_nc_u32 v7, 0, v7                             // refine 2^32 / divisor
v_mul_hi_u32 v7, v6, v7                            // refine 2^32 / divisor
v_add_nc_u32 v6, v6, v7                            // refine 2^32 / divisor
v_mul_hi_u32 v6, s[sgprWorkGroup0], v6             // s66 = s[sgprWorkGroup0] / s66
v_mul_lo_u32 v7, v6, s66                           // s66 = s[sgprWorkGroup0] / s66
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s66 = s[sgprWorkGroup0] / s66
v_cmp_ge_u32 vcc_lo, v7, s66                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v6, 1, v6                             // quotient + 1
v_sub_nc_u32 v7, v7, s66                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v7, s66                       // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v6, 1, v6                             // quotient + 1
v_sub_nc_u32 v7, v7, s66                           // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s66, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s66
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s66, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s66, s66, s[sgprWorkGroup2]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s66
label_RemapWG2Done_1:
s_cmp_lt_u32 s[sgprWorkGroup0], s[sgprNumWorkGroups0] // wg1 == 0?
s_cbranch_scc1 label_RemapWG1Done_1
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mov_b32 v7, 0x4f7ffffe                           // 2^32 - 512
v_mul_f32 v6, v7, v6                               // 2^32 / divisor
v_cvt_u32_f32 v6, v6                               // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_lo_u32 v7, v6, s[sgprNumWorkGroups0]         // refine 2^32 / divisor
v_sub_nc_u32 v7, 0, v7                             // refine 2^32 / divisor
v_mul_hi_u32 v7, v6, v7                            // refine 2^32 / divisor
v_add_nc_u32 v6, v6, v7                            // refine 2^32 / divisor
v_mul_hi_u32 v6, s[sgprWorkGroup0], v6             // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_lo_u32 v7, v6, s[sgprNumWorkGroups0]         // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7             // s66 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmp_ge_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v6, 1, v6                             // quotient + 1
v_sub_nc_u32 v7, v7, s[sgprNumWorkGroups0]         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_cmp_ge_u32 vcc_lo, v7, s[sgprNumWorkGroups0]     // remainder >= divisor?
s_mov_b32 exec_lo, vcc_lo                          // remainder >= divisor?
v_add_nc_u32 v6, 1, v6                             // quotient + 1
v_sub_nc_u32 v7, v7, s[sgprNumWorkGroups0]         // remainder - divisor
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s66, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s66
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s66, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s66
label_RemapWG1Done_1:
s_branch label_RemapWGDone_1
.set sgprAddressA, 24
.set sgprAddressB, 26
.set sgprAddressC, 36
.set sgprAddressD, 34
.set sgprAddressScaleA, 44
.set sgprAddressScaleB, 46
.set sgprAddressScaleZeroA, 64
.set sgprAlpha, 32
.set sgprArgType, 5
.set sgprKernArgAddress, 0
.set sgprNumWorkGroups0, 16
.set sgprNumWorkGroups1, 17
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
s_mov_b32 s17, 112                                 // KernArgAddressOffset
s_mul_i32 s78, s18, 4
s_mov_b64 s[72:73], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s17, 232
s_mov_b32 s78, 0
s_mov_b64 s[72:73], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s16, 1
s_mov_b32 s79, 0
s_load_b128 s[20:23], s[72:73], s78
s_cmpk_eq_u32 s18, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s76, s20, 1                             // s76 = s20 / 2
s_and_b32 s74, 1, s20                              // s74 = s20 % 2
s_addc_u32 s76, s76, 0
s_mul_hi_u32 s75, s21, 1717986919                  // STATIC_DIV: divisor=5
s_mul_i32 s74, s21, 1717986919                     // tmp = dividend * magic
s_lshr_b64 s[74:75], s[74:75], 33                  // tmp0 = quotient
s_mul_i32 s75, s74, 5                              // tmp1 = quotient * divisor
s_cmp_lg_u32 s75, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s77, s74, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s76, s76, s77
s_mul_i32 s76, s76, s22
s_add_u32 s79, s79, s76
s_cmp_lt_u32 s[sgprWorkGroup0], s79
s_cbranch_scc1 label_FOUND
s_add_u32 s78, s78, s17
s_load_b128 s[20:23], s[72:73], s78
s_add_u32 s16, s16, 1
s_cmp_lt_u32 s16, s18
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s76, s20, 1                             // s76 = s20 / 2
s_and_b32 s74, 1, s20                              // s74 = s20 % 2
s_addc_u32 s76, s76, 0
s_mul_hi_u32 s75, s21, 1717986919                  // STATIC_DIV: divisor=5
s_mul_i32 s74, s21, 1717986919                     // tmp = dividend * magic
s_lshr_b64 s[74:75], s[74:75], 33                  // tmp0 = quotient
s_mul_i32 s75, s74, 5                              // tmp1 = quotient * divisor
s_cmp_lg_u32 s75, s21                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s77, s74, 0                             // if (quotient * divisor != dividend), result+=1
s_mul_i32 s76, s76, s77
s_mul_i32 s76, s76, s22
s_add_u32 s79, s79, s76

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s73, s16, 1
s_sub_u32 s72, s79, s76
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s72
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s18, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s73, s73, 112                            // KernArgAddressOffset
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s73
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
s_mul_i32 s73, s73, 232
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s73
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
/* init: add vgpr [56...225) to pool */
/* init: add vgpr [0...10) to pool */
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
s_mul_hi_u32 s73, s[sgprSizesFree+1], 1717986919   // STATIC_DIV: divisor=5
s_mul_i32 s72, s[sgprSizesFree+1], 1717986919      // tmp = dividend * magic
s_lshr_b64 s[72:73], s[72:73], 33                  // tmp = (dividend * magic) >> shift
s_mov_b32 s[sgprNumWorkGroups1], s72               // quotient
s_mul_i32 s72, s[sgprNumWorkGroups1], 5            // quotient*divisor
s_sub_u32 s63, s[sgprSizesFree+1], s72             // rReg = dividend - quotient*divisor
s_cmp_gt_u32 s63, 0                                // Check if remainder > 0
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
.set sgprNumWorkGroups0, 16
.set sgprNumWorkGroups1, 17
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
label_WGMRemap:
s_cmp_gt_i32 s18, 1                                // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s18, 0                                // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s18, s18                                 // abs(WGM)
v_cvt_f64_u32 v[6:7], s18                          // s19 = s[sgprWorkGroup0] / s18
v_rcp_f64 v[6:7], v[6:7]                           // s19 = s[sgprWorkGroup0] / s18
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]            // s19 = s[sgprWorkGroup0] / s18
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s19 = s[sgprWorkGroup0] / s18
v_cvt_u32_f64 v6, v[6:7]                           // s19 = s[sgprWorkGroup0] / s18
v_mul_lo_u32 v7, v6, s18                           // s19 = s[sgprWorkGroup0] / s18
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7             // s19 = s[sgprWorkGroup0] / s18
v_cmp_ge_u32 vcc_lo, v8, s18                       // s19 = s[sgprWorkGroup0] / s18
s_mov_b32 exec_lo, vcc_lo                          // s19 = s[sgprWorkGroup0] / s18
v_add_nc_u32 v6, v6, 1                             // s19 = s[sgprWorkGroup0] / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s19, v6                        // quotient
s_mul_i32 s63, s19, s18                            // quotient * non-magic divisor
s_sub_u32 s63, s[sgprWorkGroup0], s63              // WorkGroup0=remainder
s_mul_i32 s63, s63, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s63, s63, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f64_u32 v[6:7], s18                          // s66 = s[sgprNumWorkGroups0] / s18
v_rcp_f64 v[6:7], v[6:7]                           // s66 = s[sgprNumWorkGroups0] / s18
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups0]        // s66 = s[sgprNumWorkGroups0] / s18
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s66 = s[sgprNumWorkGroups0] / s18
v_cvt_u32_f64 v6, v[6:7]                           // s66 = s[sgprNumWorkGroups0] / s18
v_mul_lo_u32 v7, v6, s18                           // s66 = s[sgprNumWorkGroups0] / s18
v_sub_nc_u32 v8, s[sgprNumWorkGroups0], v7         // s66 = s[sgprNumWorkGroups0] / s18
v_cmp_ge_u32 vcc_lo, v8, s18                       // s66 = s[sgprNumWorkGroups0] / s18
s_mov_b32 exec_lo, vcc_lo                          // s66 = s[sgprNumWorkGroups0] / s18
v_add_nc_u32 v6, v6, 1                             // s66 = s[sgprNumWorkGroups0] / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s66, v6                        // quotient
s_mul_i32 s67, s18, s66                            // quotient * non-magic divisor
s_sub_u32 s67, s[sgprNumWorkGroups0], s67          // NumWorkGroups0=remainder
s_cmp_eq_u32 s67, 0                                // remainder == 0 ?
s_cmov_b32 s67, s18                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s19, s66                              // blockId >= numFullBlocks ?
s_cselect_b32 s66, s67, s18
v_cvt_f64_u32 v[6:7], s66                          // s[sgprWorkGroup1] = s63 / s66
v_rcp_f64 v[6:7], v[6:7]                           // s[sgprWorkGroup1] = s63 / s66
v_cvt_f64_u32 v[8:9], s63                          // s[sgprWorkGroup1] = s63 / s66
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s[sgprWorkGroup1] = s63 / s66
v_cvt_u32_f64 v6, v[6:7]                           // s[sgprWorkGroup1] = s63 / s66
v_mul_lo_u32 v7, v6, s66                           // s[sgprWorkGroup1] = s63 / s66
v_sub_nc_u32 v8, s63, v7                           // s[sgprWorkGroup1] = s63 / s66
v_cmp_ge_u32 vcc_lo, v8, s66                       // s[sgprWorkGroup1] = s63 / s66
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup1] = s63 / s66
v_add_nc_u32 v6, v6, 1                             // s[sgprWorkGroup1] = s63 / s66
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s66                           // s[sgprWorkGroup1] = s63 / s66
v_sub_nc_u32 v8, s63, v7                           // s[sgprWorkGroup1] = s63 / s66
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v8          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s66 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s63, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s19, s19, s18                            // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s19 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s18, s18                                 // WGM
v_cvt_f64_u32 v[6:7], s18                          // s19 = s[sgprWorkGroup1] / s18
v_rcp_f64 v[6:7], v[6:7]                           // s19 = s[sgprWorkGroup1] / s18
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup1]            // s19 = s[sgprWorkGroup1] / s18
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s19 = s[sgprWorkGroup1] / s18
v_cvt_u32_f64 v6, v[6:7]                           // s19 = s[sgprWorkGroup1] / s18
v_mul_lo_u32 v7, v6, s18                           // s19 = s[sgprWorkGroup1] / s18
v_sub_nc_u32 v8, s[sgprWorkGroup1], v7             // s19 = s[sgprWorkGroup1] / s18
v_cmp_ge_u32 vcc_lo, v8, s18                       // s19 = s[sgprWorkGroup1] / s18
s_mov_b32 exec_lo, vcc_lo                          // s19 = s[sgprWorkGroup1] / s18
v_add_nc_u32 v6, v6, 1                             // s19 = s[sgprWorkGroup1] / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s19, v6                        // quotient
s_mul_i32 s63, s19, s18                            // quotient * non-magic divisor
s_sub_u32 s63, s[sgprWorkGroup1], s63              // WorkGroup1=remainder
s_mul_i32 s63, s63, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s63, s63, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f64_u32 v[6:7], s18                          // s66 = s[sgprNumWorkGroups1] / s18
v_rcp_f64 v[6:7], v[6:7]                           // s66 = s[sgprNumWorkGroups1] / s18
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups1]        // s66 = s[sgprNumWorkGroups1] / s18
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s66 = s[sgprNumWorkGroups1] / s18
v_cvt_u32_f64 v6, v[6:7]                           // s66 = s[sgprNumWorkGroups1] / s18
v_mul_lo_u32 v7, v6, s18                           // s66 = s[sgprNumWorkGroups1] / s18
v_sub_nc_u32 v8, s[sgprNumWorkGroups1], v7         // s66 = s[sgprNumWorkGroups1] / s18
v_cmp_ge_u32 vcc_lo, v8, s18                       // s66 = s[sgprNumWorkGroups1] / s18
s_mov_b32 exec_lo, vcc_lo                          // s66 = s[sgprNumWorkGroups1] / s18
v_add_nc_u32 v6, v6, 1                             // s66 = s[sgprNumWorkGroups1] / s18
s_mov_b32 exec_lo, -1                              // Reset exec
v_readfirstlane_b32 s66, v6                        // quotient
s_mul_i32 s67, s18, s66                            // quotient * non-magic divisor
s_sub_u32 s67, s[sgprNumWorkGroups1], s67          // NumWorkGroups1=remainder
s_cmp_eq_u32 s67, 0                                // remainder == 0 ?
s_cmov_b32 s67, s18                                // remainder = WGM if remainder == 0
s_cmp_ge_u32 s19, s66                              // blockId >= numFullBlocks ?
s_cselect_b32 s66, s67, s18
v_cvt_f64_u32 v[6:7], s66                          // s[sgprWorkGroup0] = s63 / s66
v_rcp_f64 v[6:7], v[6:7]                           // s[sgprWorkGroup0] = s63 / s66
v_cvt_f64_u32 v[8:9], s63                          // s[sgprWorkGroup0] = s63 / s66
v_mul_f64 v[6:7], v[6:7], v[8:9]                   // s[sgprWorkGroup0] = s63 / s66
v_cvt_u32_f64 v6, v[6:7]                           // s[sgprWorkGroup0] = s63 / s66
v_mul_lo_u32 v7, v6, s66                           // s[sgprWorkGroup0] = s63 / s66
v_sub_nc_u32 v8, s63, v7                           // s[sgprWorkGroup0] = s63 / s66
v_cmp_ge_u32 vcc_lo, v8, s66                       // s[sgprWorkGroup0] = s63 / s66
s_mov_b32 exec_lo, vcc_lo                          // s[sgprWorkGroup0] = s63 / s66
v_add_nc_u32 v6, v6, 1                             // s[sgprWorkGroup0] = s63 / s66
s_mov_b32 exec_lo, -1                              // Reset exec
v_mul_lo_u32 v7, v6, s66                           // s[sgprWorkGroup0] = s63 / s66
v_sub_nc_u32 v8, s63, v7                           // s[sgprWorkGroup0] = s63 / s66
v_readfirstlane_b32 s[sgprWorkGroup0], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v8          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s66 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s63, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s19, s19, s18                            // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s19 // wg1 += blockId * WGM
s_branch label_WGM
.set sgprAddressA, 24
.set sgprKernArgAddress, 0
.set sgprSizesSum, 23
.set sgprSrdA, 72
.set sgprWorkGroup2, 4
label_GeneralBatchedGemmLoadA:
s_mul_i32 s80, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s80, s80, s[sgprAddressA+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s81, s[sgprAddressA+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdA:sgprSrdA+1], s[80:81], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[80:81], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // Load batchOffsetA from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s80        // Add batch offset to A address (low)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s81       // Add batch offset to A address (high)
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 16         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_lshr_b64 s[82:83], s[82:83], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdA+0], s82, s[sgprSrdA+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s83, s[sgprSrdA+1]       // SRD base = Address+ tileStart1
s_branch label_StridedBatchedGemmLoadA_End
.set sgprAddressB, 26
.set sgprKernArgAddress, 0
.set sgprSizesSum, 23
.set sgprSrdB, 76
.set sgprWorkGroup2, 4
label_GeneralBatchedGemmLoadB:
s_mul_i32 s80, 8, s[sgprWorkGroup2]                // Compute Offset into Pointer Array
s_cmp_eq_u32 s[sgprSizesSum], 0x0                  // Don't dereference Pointer array if SizesSum == 0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s80, s80, s[sgprAddressB+0]              // Offsetting to the location [Lower half of address]
s_addc_u32 s81, s[sgprAddressB+1], 0               // Offsetting to the location [Higher half of address]
s_load_b64 s[sgprSrdB:sgprSrdB+1], s[80:81], 0     // Load the Matrix Address in the Pointer Array
s_waitcnt lgkmcnt(0)                               // Wait for pointer-array SRD load before reusing base SGPR
s_load_b64 s[80:81], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88 // Load batchOffsetB from kernel args
s_waitcnt lgkmcnt(0)                               // Wait for Matrix Address and Batch Offset Loads
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s80        // Add batch offset to B address (low)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s81       // Add batch offset to B address (high)
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 64         // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_lshl_b64 s[82:83], s[82:83], 1                   // tileStart (multiple bpe)
s_add_u32 s[sgprSrdB+0], s82, s[sgprSrdB+0]        // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s83, s[sgprSrdB+1]       // SRD base = Address+ tileStart1
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
