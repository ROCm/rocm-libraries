// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.BF8.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.FP6.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
 // T_LOAD_TILED 3 Source 2
 // Tensor.E8M0.d2 4, (base=&80, sizes={&88 &96 }, strides={&104 &112 })
 // T_LOAD_TILED 5 Source 4
 // BlockScale(Separate, {1, 32}): Data: 1, Scale: 5
 // Tensor.E8M0.d2 7, (base=&120, sizes={&128 &136 }, strides={&144 &152 })
 // T_LOAD_TILED 8 Source 7
 // BlockScale(Separate, {32, 1}): Data: 3, Scale: 8
 // Tensor.Half.d2 10, (base=&160, sizes={&168 &176 }, strides={&184 &192 })
 // T_LOAD_TILED 11 Source 10
 // Scalar.Float.12
 // T_LOAD_SCALAR 13Source 12
 // Scalar.Float.14
 // T_LOAD_SCALAR 15Source 14
 // T_Mul 6 9 Value: Float
 // T_EXECUTE 13 16 15 11
 //   E_Mul 17, 15, 11
 //   E_Mul 18, 13, 16
 //   E_Add 19, 17, 18
 // Tensor.Half.d2 21, (base=&208, sizes={&216 &224 }, strides={&232 &240 })
 // T_EXECUTE 19
 //   E_Cvt 22, 19
 // T_STORE_TILED 24 Source 22 Dest 21

 // Command arguments:
 // Tensor_0_pointer: PointerGlobal: BF8(offset: 0, size: 8, read_write)
 // Tensor_0_size_0: Value: Int64(offset: 8, size: 8, read_only)
 // Tensor_0_size_1: Value: Int64(offset: 16, size: 8, read_only)
 // Tensor_0_stride_0: Value: Int64(offset: 24, size: 8, read_only)
 // Tensor_0_stride_1: Value: Int64(offset: 32, size: 8, read_only)
 // Tensor_2_pointer: PointerGlobal: FP6(offset: 40, size: 8, read_write)
 // Tensor_2_size_0: Value: Int64(offset: 48, size: 8, read_only)
 // Tensor_2_size_1: Value: Int64(offset: 56, size: 8, read_only)
 // Tensor_2_stride_0: Value: Int64(offset: 64, size: 8, read_only)
 // Tensor_2_stride_1: Value: Int64(offset: 72, size: 8, read_only)
 // Tensor_4_pointer: PointerGlobal: E8M0(offset: 80, size: 8, read_write)
 // Tensor_4_size_0: Value: Int64(offset: 88, size: 8, read_only)
 // Tensor_4_size_1: Value: Int64(offset: 96, size: 8, read_only)
 // Tensor_4_stride_0: Value: Int64(offset: 104, size: 8, read_only)
 // Tensor_4_stride_1: Value: Int64(offset: 112, size: 8, read_only)
 // Tensor_7_pointer: PointerGlobal: E8M0(offset: 120, size: 8, read_write)
 // Tensor_7_size_0: Value: Int64(offset: 128, size: 8, read_only)
 // Tensor_7_size_1: Value: Int64(offset: 136, size: 8, read_only)
 // Tensor_7_stride_0: Value: Int64(offset: 144, size: 8, read_only)
 // Tensor_7_stride_1: Value: Int64(offset: 152, size: 8, read_only)
 // Tensor_10_pointer: PointerGlobal: Half(offset: 160, size: 8, read_write)
 // Tensor_10_size_0: Value: Int64(offset: 168, size: 8, read_only)
 // Tensor_10_size_1: Value: Int64(offset: 176, size: 8, read_only)
 // Tensor_10_stride_0: Value: Int64(offset: 184, size: 8, read_only)
 // Tensor_10_stride_1: Value: Int64(offset: 192, size: 8, read_only)
 // user_Float_Value_12: Value: Float(offset: 200, size: 4, read_only)
 // user_Float_Value_14: Value: Float(offset: 204, size: 4, read_only)
 // Tensor_21_pointer: PointerGlobal: Half(offset: 208, size: 8, read_write)
 // Tensor_21_size_0: Value: Int64(offset: 216, size: 8, read_only)
 // Tensor_21_size_1: Value: Int64(offset: 224, size: 8, read_only)
 // Tensor_21_stride_0: Value: Int64(offset: 232, size: 8, read_only)
 // Tensor_21_stride_1: Value: Int64(offset: 240, size: 8, read_only)

 // Kernel Arguments:
 // KernelArg{Tensor_0_stride_0_0, Value: Int64, CommandArgument(Tensor_0_stride_0)I64(c 0), o:0, s:8, preloaded}
 // KernelArg{Tensor_0_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP6, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
 // KernelArg{Tensor_2_size_1_4, Value: Int64, CommandArgument(Tensor_2_size_1)I64(c 0), o:40, s:8, preloaded}
 // KernelArg{user_Float_Value_14, Value: Float, CommandArgument(user_Float_Value_14)S(c 0), o:48, s:4, preloaded}
 // KernelArg{user_Float_Value_12, Value: Float, CommandArgument(user_Float_Value_12)S(c 0), o:52, s:4, preloaded}
 // KernelArg{Tensor_4_stride_0_6, Value: Int64, CommandArgument(Tensor_4_stride_0)I64(c 0), o:56, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_4_pointer)PG(c 0), o:64, s:8}
 // KernelArg{Tensor_7_stride_1_8, Value: Int64, CommandArgument(Tensor_7_stride_1)I64(c 0), o:72, s:8}
 // KernelArg{Tensor_7_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_7_pointer)PG(c 0), o:80, s:8}
 // KernelArg{Tensor_0_size_1_14, Value: Int64, CommandArgument(Tensor_0_size_1)I64(c 0), o:88, s:8}
 // KernelArg{Tensor_21_stride_1_16, Value: Int64, CommandArgument(Tensor_21_stride_1)I64(c 0), o:96, s:8}
 // KernelArg{Tensor_21_pointer, PointerGlobal: Half, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: Half, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}
.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
 // Pad to make room for arg loading preamble.
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
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:
 // Kernel execution starts here. Use this label instead of the kernel name when setting a breakpoint.
 // Kernel Options:
 //   logLevel:                             Verbose
 //   alwaysWaitAfterLoad:                    false
 //   alwaysWaitAfterStore:                   false
 //   alwaysWaitBeforeBranch:                 false
 //   alwaysWaitZeroBeforeBarrier:            false
 //   systemPreloadedKernelArguments:            -1
 //   lazyLoadKernelArguments:                false
 //   maxACCVGPRs:                              256
 //   maxSGPRs:                                 102
 //   maxVGPRs:                                 256
 //   loadLocalWidth:                             4
 //   loadGlobalWidth:                            8
 //   storeLocalWidth:                            4
 //   storeGlobalWidth:                           4
 //   assertWaitCntState:                      true
 //   setNextFreeVGPRToMax:                   false
 //   deduplicateArguments:                    true
 //   minLaunchTimeExpressionComplexity:         20
 //   maxConcurrentSubExpressions:                1
 //   maxConcurrentControlOps:                 none
 //   dsObserver:                        DSMEMObserver
 //   enableFullDivision:                     false
 //   scaleSkipPermlane:                       None
 //   assertOpKind:                            NoOp
 //   removeSetCoordinate:                    false
 //   ldsSwizzleMode:                          None
 //   coexecutionEnabled:                      true
 //   favourF8F6F4OverF8MatrixInstruction:      true

 // Loading Kernel Arguments: 
 // KernelArg{Tensor_0_stride_0_0, Value: Int64, CommandArgument(Tensor_0_stride_0)I64(c 0), o:0, s:8, preloaded}
 // KernelArg{Tensor_0_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP6, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
 // KernelArg{Tensor_2_size_1_4, Value: Int64, CommandArgument(Tensor_2_size_1)I64(c 0), o:40, s:8, preloaded}
 // KernelArg{user_Float_Value_14, Value: Float, CommandArgument(user_Float_Value_14)S(c 0), o:48, s:4, preloaded}
 // KernelArg{user_Float_Value_12, Value: Float, CommandArgument(user_Float_Value_12)S(c 0), o:52, s:4, preloaded}
 // KernelArg{Tensor_4_stride_0_6, Value: Int64, CommandArgument(Tensor_4_stride_0)I64(c 0), o:56, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_4_pointer)PG(c 0), o:64, s:8}
 // KernelArg{Tensor_7_stride_1_8, Value: Int64, CommandArgument(Tensor_7_stride_1)I64(c 0), o:72, s:8}
 // KernelArg{Tensor_7_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_7_pointer)PG(c 0), o:80, s:8}
 // KernelArg{Tensor_0_size_1_14, Value: Int64, CommandArgument(Tensor_0_size_1)I64(c 0), o:88, s:8}
 // KernelArg{Tensor_21_stride_1_16, Value: Int64, CommandArgument(Tensor_21_stride_1)I64(c 0), o:96, s:8}
 // KernelArg{Tensor_21_pointer, PointerGlobal: Half, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: Half, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}

 // Kernel arguments: Loading 72 bytes
s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value
 // Freeing Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15

 // Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]
 // Tensor_0_pointer: SGPR PointerGlobal: BF8 x 1: s[4:5]
 // Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]
 // Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]
 // Tensor_2_pointer: SGPR PointerGlobal: FP6 x 1: s[10:11]
 // Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]
 // user_Float_Value_14: SGPR Value: Float x 1: s14
 // user_Float_Value_12: SGPR Value: Float x 1: s15
 // Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]
 // Tensor_4_pointer: SGPR PointerGlobal: E8M0 x 1: s[22:23]
 // Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]
 // Tensor_7_pointer: SGPR PointerGlobal: E8M0 x 1: s[26:27]
 // Tensor_0_size_1_14: SGPR Value: Int64 x 1: s[28:29]
 // Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]
 // Tensor_21_pointer: SGPR PointerGlobal: Half x 1: s[32:33]
 // Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]
 // Tensor_10_pointer: SGPR PointerGlobal: Half x 1: s[36:37]
 // Extract 10 bit X coordinate
v_bfe_u32 v1, v0, 0, 10
 // Generate {Workitem Index X: v1:U32} into Workitem Index X: VGPR Value: UInt32 x 1: v1
 // reg expression
 // Extract 10 bit Y coordinate
v_bfe_u32 v2, v0, 10, 10
 // Generate {Workitem Index Y: v2:U32} into Workitem Index Y: VGPR Value: UInt32 x 1: v2
 // reg expression
 // Freeing Packed Workitem Index: 1 VGPR (Value: Raw32): v0

 // Freeing Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1

 // Marking Workgroup Index X: SGPR Value: UInt32 x 1: s16 as read-only
 // Marking Workgroup Index Y: SGPR Value: UInt32 x 1: s17 as read-only
 // Marking Workitem Index X: VGPR Value: UInt32 x 1: v1 as read-only
 // Marking Workitem Index Y: VGPR Value: UInt32 x 1: v2 as read-only
 // CodeGeneratorVisitor::generate() begin
 // UpdateParameters
 // IdentifyParallelDimensions
 // OrderMemory
 // AddLDS
 // LowerLinear
 // LowerTile
 // LowerTensorContraction
 // Simplify
 // AddLDSPadding
 // ConstantPropagation
 // FuseExpressions
 // ConnectWorkgroups
 // WorkgroupRemapXCC
 // UnrollLoops
 // FuseLoops
 // RemoveDuplicates
 // OrderEpilogueBlocks
 // Simplify
 // CleanLoops
 // SwizzleScale
 // AddPrefetch
 // AddF6LDSPadding
 // AddDirect2LDS
 // AddTDMToLDS
 // Simplify
 // AddPRNG
 // UpdateWavefrontParameters
 // AssignIndexExpressions
 // LoadPacked
 // AddConvert
 // NopExtraScopes
 // InlineInits
 // InlineIncrements
 // OrderMultiplyNodes
 // Simplify
 // Simplify
 // OrderExchangeNodes
 // AliasDataFlowTags
 // AddLDSBarriers
 // AddDeallocateDataFlow
 // CleanArguments
 // AddDeallocateArguments
 // MergeAdjacentDeallocates
 // Simplify
 // SortArguments
 // SetWorkitemCount
 // ModelAddresses
 // generate({1})
 // Kernel(1) BEGIN
 // (op 1) generate({})
 // (op 1) end: generate({})
 // (op 1) generate({787, 840, 893, 946, 1019, 1043, 1067, 1091})
 // (op 1) NOP(787) BEGIN
 // (op 787) generate({})
 // (op 787) end: generate({})
 // (op 1) NOP(787) END
 // (op 1) NOP(840) BEGIN
 // (op 840) generate({})
 // (op 840) end: generate({})
 // (op 1) NOP(840) END
 // (op 1) NOP(893) BEGIN
 // (op 893) generate({})
 // (op 893) end: generate({})
 // (op 1) NOP(893) END
 // (op 1) NOP(946) BEGIN
 // (op 946) generate({})
 // (op 946) end: generate({})
 // (op 1) NOP(946) END
 // (op 1) NOP(1019) BEGIN
 // (op 1019) generate({})
 // (op 1019) end: generate({})
 // (op 1) NOP(1019) END
 // (op 1) NOP(1043) BEGIN
 // (op 1043) generate({})
 // (op 1043) end: generate({})
 // (op 1) NOP(1043) END
 // (op 1) NOP(1067) BEGIN
 // (op 1067) generate({})
 // (op 1067) end: generate({})
 // (op 1) NOP(1067) END
 // (op 1) NOP(1091) BEGIN
 // (op 1091) generate({})
 // (op 1091) end: generate({})
 // (op 1) NOP(1091) END
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(791) BEGIN
 // (op 791) Assign dim(588) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 588: v**UNALLOCATED**
 // (op 791) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag588: VGPR Value: UInt32 x 1: (unallocated)
 // (op 791) reg expression
 // (op 791) BEGIN: Flatten[1]
 // (op 791) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(844) BEGIN
 // (op 844) Assign dim(604) = LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 604: v**UNALLOCATED**
 // (op 844) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag604: VGPR Value: UInt32 x 1: (unallocated)
 // (op 844) reg expression
 // (op 844) BEGIN: Flatten[1]
 // (op 844) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(897) BEGIN
 // (op 897) Assign dim(620) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 620: v**UNALLOCATED**
 // (op 897) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag620: VGPR Value: UInt32 x 1: (unallocated)
 // (op 897) reg expression
 // (op 897) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(950) BEGIN
 // (op 950) Assign dim(636) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 636: v**UNALLOCATED**
 // (op 950) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag636: VGPR Value: UInt32 x 1: (unallocated)
 // (op 950) reg expression
 // (op 950) BEGIN: Flatten[1]Tile
 // (op 950) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1023) BEGIN
 // (op 1023) Assign dim(658) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 658: v**UNALLOCATED**
 // (op 1023) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag658: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1023) reg expression
 // (op 1023) BEGIN: Flatten[1]
 // (op 1023) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1047) BEGIN
 // (op 1047) Assign dim(666) = LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 666: v**UNALLOCATED**
 // (op 1047) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag666: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1047) reg expression
 // (op 1047) BEGIN: Flatten[1]
 // (op 1047) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(1071) BEGIN
 // (op 1071) Assign dim(674) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 674: v**UNALLOCATED**
 // (op 1071) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag674: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1071) reg expression
 // (op 1071) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(1095) BEGIN
 // (op 1095) Assign dim(682) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 682: v**UNALLOCATED**
 // (op 1095) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag682: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1095) reg expression
 // (op 1095) BEGIN: Flatten[1]Tile
 // (op 1095) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 791): v0
v_and_b32 v0, 63, v1 // (op 791) 
 // (op 791) END: Flatten[1]
 // (op 791) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 791): v3
v_lshrrev_b32 v3, 4, v0 // (op 791) 
 // (op 791) BEGIN: Flatten[1]
 // (op 791) {Flatten[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 791): v4
v_and_b32 v4, 1, v3 // (op 791) 
 // (op 791) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 791): v3

 // (op 791) BEGIN: Flatten[1]
 // (op 791) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 791): v3
v_and_b32 v3, 15, v1 // (op 791) 
 // (op 791) END: Flatten[1]
 // (op 791) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 791): v5
v_lshl_add_u32 v5, v4, 4, v3 // (op 791) 
 // (op 791) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 791): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 791): v4

 // (op 791) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 791): v3
v_lshrrev_b32 v3, 6, v1 // (op 791) 
 // (op 791) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 791): v4
v_lshl_add_u32 v4, v3, 5, v5 // (op 791) 
 // (op 791) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 791): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 791): v3

 // (op 791) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 791): v3
v_lshrrev_b32 v3, 5, v0 // (op 791) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 791): v0

 // (op 791) BEGIN: TileTile
 // (op 791) {TileTile: ShiftL(v3:U32, 4:U32)U32}
 // (op 791) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 791)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 791)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 791): v0
v_lshlrev_b32 v0, 4, v3 // (op 791) 
 // (op 791) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 791): v3

 // (op 791) BEGIN: Tile
// Allocated DataFlowTag588: 1 VGPR (Value: UInt32) (op 791): v3
v_lshl_add_u32 v3, v4, 6, v0 // (op 791) 
 // (op 791) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 791): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 791): v4

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(791) END
// Allocated : 1 VGPR (Value: UInt32) (op 844): v0
v_and_b32 v0, 63, v1 // (op 844) 
 // (op 844) END: Flatten[1]
 // (op 844) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 844): v4
v_lshrrev_b32 v4, 4, v0 // (op 844) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 844): v0

 // (op 844) BEGIN: Flatten[1]
 // (op 844) {Flatten[1]: BitwiseAnd(v4:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 844): v0
v_and_b32 v0, 1, v4 // (op 844) 
 // (op 844) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 844): v4

 // (op 844) BEGIN: Flatten[1]
 // (op 844) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 844): v4
v_and_b32 v4, 15, v1 // (op 844) 
 // (op 844) END: Flatten[1]
 // (op 844) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 844): v5
v_lshl_add_u32 v5, v0, 4, v4 // (op 844) 
 // (op 844) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 844): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 844): v0

 // (op 844) BEGIN: TileTile
 // (op 844) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 844): v0
v_and_b32 v0, 32, v1 // (op 844) 
 // (op 844) END: TileTile
 // (op 844) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 844): v4
v_lshl_add_u32 v4, v5, 6, v0 // (op 844) 
 // (op 844) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 844): v0

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 844): v5

 // (op 844) Multiply({Tile: v4:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 844): v0
v_mul_lo_u32 v0, v4, 6 // (op 844) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 844): v4

 // (op 844) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag604: 1 VGPR (Value: UInt32) (op 844): v4
v_lshrrev_b32 v4, 3, v0 // (op 844) 
 // Freeing : 1 VGPR (Value: UInt32) (op 844): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(844) END
// Allocated : 1 VGPR (Value: UInt32) (op 897): v0
v_lshrrev_b32 v0, 6, v1 // (op 897) 
 // (op 897) BEGIN: Flatten[1]
 // (op 897) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 897): v5
v_and_b32 v5, 31, v1 // (op 897) 
 // (op 897) END: Flatten[1]
 // (op 897) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 897): v6
v_lshl_add_u32 v6, v0, 5, v5 // (op 897) 
 // (op 897) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 897): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 897): v0

 // (op 897) BEGIN: Flatten[1]
 // (op 897) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 897): v0
v_and_b32 v0, 63, v1 // (op 897) 
 // (op 897) END: Flatten[1]
 // (op 897) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 897): v5
v_lshrrev_b32 v5, 5, v0 // (op 897) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 897): v0

 // (op 897) BEGIN: Tile
// Allocated DataFlowTag620: 1 VGPR (Value: UInt32) (op 897): v0
v_lshl_add_u32 v0, v6, 1, v5 // (op 897) 
 // (op 897) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 897): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 897): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(897) END
// Allocated : 1 VGPR (Value: UInt32) (op 950): v5
v_and_b32 v5, 31, v1 // (op 950) 
 // (op 950) END: Flatten[1]Tile
 // (op 950) BEGIN: Flatten[1]
 // (op 950) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 950): v6
v_and_b32 v6, 63, v1 // (op 950) 
 // (op 950) END: Flatten[1]
 // (op 950) LogicalShiftR({Flatten[1]: v6:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 950): v7
v_lshrrev_b32 v7, 5, v6 // (op 950) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 950): v6

 // (op 950) BEGIN: Tile
// Allocated DataFlowTag636: 1 VGPR (Value: UInt32) (op 950): v6
v_lshl_add_u32 v6, v5, 1, v7 // (op 950) 
 // (op 950) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 950): v7

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 950): v5

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(950) END
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v5
v_and_b32 v5, 63, v1 // (op 1023) 
 // (op 1023) END: Flatten[1]
 // (op 1023) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v7
v_lshrrev_b32 v7, 4, v5 // (op 1023) 
 // (op 1023) BEGIN: Flatten[1]
 // (op 1023) {Flatten[1]: BitwiseAnd(v7:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v8
v_and_b32 v8, 1, v7 // (op 1023) 
 // (op 1023) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1023): v7

 // (op 1023) BEGIN: Flatten[1]
 // (op 1023) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v7
v_and_b32 v7, 15, v1 // (op 1023) 
 // (op 1023) END: Flatten[1]
 // (op 1023) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v9
v_lshl_add_u32 v9, v8, 4, v7 // (op 1023) 
 // (op 1023) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1023): v7

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1023): v8

 // (op 1023) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v7
v_lshrrev_b32 v7, 6, v1 // (op 1023) 
 // (op 1023) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v8
v_lshl_add_u32 v8, v7, 5, v9 // (op 1023) 
 // (op 1023) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1023): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1023): v7

 // (op 1023) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v7
v_lshrrev_b32 v7, 5, v5 // (op 1023) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1023): v5

 // (op 1023) BEGIN: TileTile
 // (op 1023) {TileTile: ShiftL(v7:U32, 4:U32)U32}
 // (op 1023) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1023)         value (VGPR Value: UInt32 x 1: v7) 
 // (op 1023)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1023): v5
v_lshlrev_b32 v5, 4, v7 // (op 1023) 
 // (op 1023) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 1023): v7

 // (op 1023) BEGIN: Tile
// Allocated DataFlowTag658: 1 VGPR (Value: UInt32) (op 1023): v7
v_lshl_add_u32 v7, v8, 6, v5 // (op 1023) 
 // (op 1023) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1023): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1023): v8

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1023) END
// Allocated : 1 VGPR (Value: UInt32) (op 1047): v5
v_and_b32 v5, 63, v1 // (op 1047) 
 // (op 1047) END: Flatten[1]
 // (op 1047) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
 // (op 1) Deallocate{}(1325) BEGIN
 // (op 1325) Deallocate 104
 // Deleting tag 104
 // (op 1) Deallocate{}(1325) END
// Allocated : 1 VGPR (Value: UInt32) (op 1047): v8
v_lshrrev_b32 v8, 4, v5 // (op 1047) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1047): v5

 // (op 1047) BEGIN: Flatten[1]
 // (op 1047) {Flatten[1]: BitwiseAnd(v8:U32, 1:U32)U32}
 // (op 1) Assign VGPR 64:U32(792) BEGIN
 // (op 792) Assign dim(589) = 64:U32
 // (op 792) Assign stride expression
 // (op 1) Assign VGPR 64:U32(792) END
 // (op 1) Assign VGPR 64:U32(1024) BEGIN
 // (op 1024) Assign dim(659) = 64:U32
 // (op 1024) Assign stride expression
 // (op 1) Assign VGPR 64:U32(1024) END
 // (op 1) NOP(788) BEGIN
 // (op 788) generate({})
 // (op 788) end: generate({})
 // (op 1) NOP(788) END
 // (op 1) NOP(1020) BEGIN
 // (op 1020) generate({})
 // (op 1020) end: generate({})
 // (op 1) NOP(1020) END
 // (op 1) Assign VGPR 1:U32(796) BEGIN
 // (op 796) Assign dim(591) = 1:U32
 // (op 796) Assign stride expression
 // (op 1) Assign VGPR 1:U32(796) END
 // (op 1) Assign VGPR 1:U32(1028) BEGIN
 // (op 1028) Assign dim(661) = 1:U32
 // (op 1028) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1028) END
// Allocated : 1 VGPR (Value: UInt32) (op 1047): v5
v_and_b32 v5, 1, v8 // (op 1047) 
 // (op 1047) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1047): v8

 // (op 1047) BEGIN: Flatten[1]
 // (op 1047) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1047): v8
v_and_b32 v8, 15, v1 // (op 1047) 
 // (op 1047) END: Flatten[1]
 // (op 1047) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 1047): v9
v_lshl_add_u32 v9, v5, 4, v8 // (op 1047) 
 // (op 1047) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1047): v8

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1047): v5

 // (op 1047) BEGIN: TileTile
 // (op 1047) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1047): v5
v_and_b32 v5, 32, v1 // (op 1047) 
 // (op 1047) END: TileTile
 // (op 1047) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1047): v8
v_lshl_add_u32 v8, v9, 6, v5 // (op 1047) 
 // (op 1047) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1047): v5

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1047): v9

 // (op 1047) Multiply({Tile: v8:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1047): v5
v_mul_lo_u32 v5, v8, 6 // (op 1047) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1047): v8

 // (op 1047) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag666: 1 VGPR (Value: UInt32) (op 1047): v8
v_lshrrev_b32 v8, 3, v5 // (op 1047) 
 // Freeing : 1 VGPR (Value: UInt32) (op 1047): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1047) END
// Allocated : 1 VGPR (Value: UInt32) (op 1071): v5
v_lshrrev_b32 v5, 6, v1 // (op 1071) 
 // (op 1071) BEGIN: Flatten[1]
 // (op 1071) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Deallocate{}(1328) BEGIN
 // (op 1328) Deallocate 170
 // Deleting tag 170
 // (op 1) Deallocate{}(1328) END
// Allocated : 1 VGPR (Value: UInt32) (op 1071): v9
v_and_b32 v9, 31, v1 // (op 1071) 
 // (op 1071) END: Flatten[1]
 // (op 1071) BEGIN: Tile
 // (op 1) Assign VGPR 48:U32(845) BEGIN
 // (op 845) Assign dim(605) = 48:U32
 // (op 845) Assign stride expression
 // (op 1) Assign VGPR 48:U32(845) END
 // (op 1) Assign VGPR 48:U32(1048) BEGIN
 // (op 1048) Assign dim(667) = 48:U32
 // (op 1048) Assign stride expression
 // (op 1) Assign VGPR 48:U32(1048) END
 // (op 1) NOP(841) BEGIN
 // (op 841) generate({})
 // (op 841) end: generate({})
 // (op 1) NOP(841) END
 // (op 1) NOP(1044) BEGIN
 // (op 1044) generate({})
 // (op 1044) end: generate({})
 // (op 1) NOP(1044) END
 // (op 1) Assign VGPR 0:U32(849) BEGIN
 // (op 849) Assign dim(607) = 0:U32
 // (op 849) Assign stride expression
 // (op 1) Assign VGPR 0:U32(849) END
 // (op 1) Assign VGPR 0:U32(1052) BEGIN
 // (op 1052) Assign dim(669) = 0:U32
 // (op 1052) Assign stride expression
 // (op 1) Assign VGPR 0:U32(1052) END
// Allocated : 1 VGPR (Value: UInt32) (op 1071): v10
v_lshl_add_u32 v10, v5, 5, v9 // (op 1071) 
 // (op 1071) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1071): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1071): v5

 // (op 1071) BEGIN: Flatten[1]
 // (op 1071) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1071): v5
v_and_b32 v5, 63, v1 // (op 1071) 
 // (op 1071) END: Flatten[1]
 // (op 1071) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1071): v9
v_lshrrev_b32 v9, 5, v5 // (op 1071) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1071): v5

 // (op 1071) BEGIN: Tile
// Allocated DataFlowTag674: 1 VGPR (Value: UInt32) (op 1071): v5
v_lshl_add_u32 v5, v10, 1, v9 // (op 1071) 
 // (op 1071) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1071): v9

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1071): v10

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(1071) END
// Allocated : 1 VGPR (Value: UInt32) (op 1095): v9
v_and_b32 v9, 31, v1 // (op 1095) 
 // (op 1095) END: Flatten[1]Tile
 // (op 1095) BEGIN: Flatten[1]
 // (op 1095) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Deallocate{}(1331) BEGIN
 // (op 1331) Deallocate 236
 // Deleting tag 236
 // (op 1) Deallocate{}(1331) END
// Allocated : 1 VGPR (Value: UInt32) (op 1095): v10
v_and_b32 v10, 63, v1 // (op 1095) 
 // (op 1095) END: Flatten[1]
 // (op 1095) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
 // (op 1) Assign VGPR 2:U32(898) BEGIN
 // (op 898) Assign dim(621) = 2:U32
 // (op 898) Assign stride expression
 // (op 1) Assign VGPR 2:U32(898) END
 // (op 1) Assign VGPR 2:U32(1072) BEGIN
 // (op 1072) Assign dim(675) = 2:U32
 // (op 1072) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1072) END
 // (op 1) NOP(894) BEGIN
 // (op 894) generate({})
 // (op 894) end: generate({})
 // (op 1) NOP(894) END
 // (op 1) NOP(1068) BEGIN
 // (op 1068) generate({})
 // (op 1068) end: generate({})
 // (op 1) NOP(1068) END
 // (op 1) Assign VGPR 1:U32(902) BEGIN
 // (op 902) Assign dim(623) = 1:U32
 // (op 902) Assign stride expression
 // (op 1) Assign VGPR 1:U32(902) END
 // (op 1) Assign VGPR 1:U32(1076) BEGIN
 // (op 1076) Assign dim(677) = 1:U32
 // (op 1076) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1076) END
// Allocated : 1 VGPR (Value: UInt32) (op 1095): v11
v_lshrrev_b32 v11, 5, v10 // (op 1095) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1095): v10

 // (op 1095) BEGIN: Tile
// Allocated DataFlowTag682: 1 VGPR (Value: UInt32) (op 1095): v10
v_lshl_add_u32 v10, v9, 1, v11 // (op 1095) 
 // (op 1095) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1095): v11

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 1095): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(1095) END
 // (op 1) Deallocate{}(1334) BEGIN
 // (op 1334) Deallocate 296
 // Deleting tag 296
 // (op 1) Deallocate{}(1334) END
 // (op 1) Assign VGPR 2:U32(951) BEGIN
 // (op 951) Assign dim(637) = 2:U32
 // (op 951) Assign stride expression
 // (op 1) Assign VGPR 2:U32(951) END
 // (op 1) Assign VGPR 2:U32(1096) BEGIN
 // (op 1096) Assign dim(683) = 2:U32
 // (op 1096) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1096) END
 // (op 1) NOP(947) BEGIN
 // (op 947) generate({})
 // (op 947) end: generate({})
 // (op 1) NOP(947) END
 // (op 1) NOP(1092) BEGIN
 // (op 1092) generate({})
 // (op 1092) end: generate({})
 // (op 1) NOP(1092) END
 // (op 1) Assign VGPR 1:U32(955) BEGIN
 // (op 955) Assign dim(639) = 1:U32
 // (op 955) Assign stride expression
 // (op 1) Assign VGPR 1:U32(955) END
 // (op 1) Assign VGPR 1:U32(1100) BEGIN
 // (op 1100) Assign dim(685) = 1:U32
 // (op 1100) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1100) END
 // (op 1) LoadVGPR Value: Float(2) BEGIN
 // tag 44: v**UNALLOCATED**
 // (op 2) GEN: LoadVGPR; scalar value
 // (op 1) LoadVGPR Value: Float(43) BEGIN
 // tag 53: v**UNALLOCATED**
 // (op 43) GEN: LoadVGPR; scalar value
// Allocated : 1 VGPR (Value: Float) (op 2): v9
v_mov_b32 v9, s14 // (op 2) Move value
 // (op 1) LoadVGPR Value: Float(2) END
// Allocated : 1 VGPR (Value: Float) (op 43): v11
v_mov_b32 v11, s15 // (op 43) Move value
 // (op 1) LoadVGPR Value: Float(43) END
 // (op 1) Deallocate{user_Float_Value_14}(1383) BEGIN
 // (op 1383) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1383) END
 // (op 1) Deallocate{user_Float_Value_12}(1385) BEGIN
 // (op 1385) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1385) END
 // (op 1) Scope(344) BEGIN
 // (op 344) Lock Scope 344
 // Lock Branch
 // (op 344) generate({349})
 // (op 344) NOP(349) BEGIN
 // (op 349) generate({})
 // (op 349) end: generate({})
 // (op 344) NOP(349) END
 // (op 344) Assign ACCVGPR 0.00000:S(65) BEGIN
 // (op 65) Assign dim(55) = 0.00000:S
 // tag 55: a**UNALLOCATED**
 // (op 65) Generate 0.00000:S into DataFlowTag55: ACCVGPR Value: Float x 16: (unallocated)
// Allocated DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 65): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 65) call()
v_accvgpr_write a5, 0.00000 // (op 65) call()
v_accvgpr_write a6, 0.00000 // (op 65) call()
v_accvgpr_write a7, 0.00000 // (op 65) call()
v_accvgpr_write a8, 0.00000 // (op 65) call()
v_accvgpr_write a9, 0.00000 // (op 65) call()
v_accvgpr_write a10, 0.00000 // (op 65) call()
v_accvgpr_write a11, 0.00000 // (op 65) call()
v_accvgpr_write a12, 0.00000 // (op 65) call()
v_accvgpr_write a13, 0.00000 // (op 65) call()
v_accvgpr_write a14, 0.00000 // (op 65) call()
v_accvgpr_write a15, 0.00000 // (op 65) call()
 // (op 344) Assign ACCVGPR 0.00000:S(65) END
 // (op 344) Scope(494) BEGIN
 // (op 494) Lock Scope 494
 // Lock Branch
 // (op 494) generate({489})
 // (op 494) NOP(489) BEGIN
 // (op 489) generate({})
 // (op 489) end: generate({})
 // (op 494) NOP(489) END
 // (op 494) Assign VGPR 64:U64(499) BEGIN
 // (op 499) Assign dim(502) = 64:U64
 // (op 499) Assign stride expression
 // (op 494) Assign VGPR 64:U64(499) END
 // (op 494) NOP(490) BEGIN
 // (op 490) generate({})
 // (op 490) end: generate({})
 // (op 494) NOP(490) END
 // (op 494) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(502) BEGIN
 // (op 502) Assign dim(503) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}
 // tag 503: v**UNALLOCATED**
 // (op 502) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64} into DataFlowTag503: VGPR Value: UInt64 x 1: (unallocated)
 // (op 502) reg expression
 // (op 502) reg expression
 // (op 502) reg expression
 // (op 502) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 502): v12
v_lshrrev_b32 v12, 2, v1 // (op 502) 
 // (op 502) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 502): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 502) 
 // (op 502) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 502): v12

 // (op 502) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 502): v12
v_and_b32 v12, -4, v1 // (op 502) 
 // (op 502) BEGIN: Flatten[1]
 // (op 502) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 502): v14
v_sub_u32 v14, v1, v12 // (op 502) 
 // (op 502) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 502): v12

 // (op 502) BEGIN: TileTile
 // (op 502) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 502) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 502)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 502)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 502): v12
v_lshlrev_b32 v12, 4, v14 // (op 502) 
 // (op 502) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 502): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 502): v14, v15
v_mov_b32 v15, 0 // (op 502) convert
v_mov_b32 v14, v12 // (op 502) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 502): v12

 // (op 502) BEGIN: Split
 // (op 502) Multiply: dest (DataFlowTag503: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 502)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 502)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag503: 2 VGPRs (Value: UInt64) (op 502): v16, v17
v_mul_lo_u32 v17, v13, s3 // (op 502) most significant: low of low * high
 // (op 502) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 502): v12
v_mul_hi_u32 v12, v13, s2 // (op 502) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 502) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 502) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 502): v12

 // (op 502) Add: dest (DataFlowTag503: VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 502)      lhs (DataFlowTag503: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 502)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 502): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 502) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 502) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 502): s0, s1

 // (op 502) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 502): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 502): v13

 // (op 494) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(502) END
 // (op 494) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(503) BEGIN
 // (op 503) Assign dim(504) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 503) Assign stride expression
 // (op 494) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(503) END
 // (op 494) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(504) BEGIN
 // (op 504) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 497: s**UNALLOCATED**
 // (op 504) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: (unallocated)
 // (op 504) reg expression
 // (op 504) reg expression
 // (op 504) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 504) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 504)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 504)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 504): s0, s1
s_mul_i32 s1, s2, s7 // (op 504) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 504): s14
s_mul_i32 s14, s3, s6 // (op 504) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 504) least significant
// Allocated : 1 SGPR (Value: Int32) (op 504): s15
s_mul_hi_u32 s15, s2, s6 // (op 504) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 504) most significant: sum
s_add_u32 s1, s1, s15 // (op 504) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 504): s15

 // Freeing : 1 SGPR (Value: Int32) (op 504): s14

 // (op 504) reg expression
 // (op 504) reg expression
// Allocated DataFlowTag497: 4 SGPRs (Buffer: None) (op 504): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 504) reg expression
s_mov_b32 s41, s5 // (op 504) reg expression
s_mov_b32 s42, s0 // (op 504) reg expression
s_mov_b32 s43, 131072 // (op 504) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 504): s0, s1

 // (op 494) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(504) END
 // (op 494) NOP(491) BEGIN
 // (op 491) generate({})
 // (op 491) end: generate({})
 // (op 494) NOP(491) END
 // (op 494) Assign VGPR 1:U64(509) BEGIN
 // (op 509) Assign dim(506) = 1:U64
 // (op 509) Assign stride expression
 // (op 494) Assign VGPR 1:U64(509) END
 // (op 494) NOP(512) BEGIN
 // (op 512) generate({})
 // (op 512) end: generate({})
 // (op 494) NOP(512) END
 // (op 494) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(521) BEGIN
 // (op 521) Assign dim(507) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}
 // tag 507: v**UNALLOCATED**
 // (op 521) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64} into DataFlowTag507: VGPR Value: UInt64 x 1: (unallocated)
 // (op 521) reg expression
 // (op 521) reg expression
 // (op 521) reg expression
 // (op 521) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 521): v12
v_lshrrev_b32 v12, 2, v1 // (op 521) 
 // (op 521) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 521): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 521) 
 // (op 521) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 521): v12

 // (op 521) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 521): v12
v_and_b32 v12, -4, v1 // (op 521) 
 // (op 521) BEGIN: Flatten[1]
 // (op 521) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 521): v14
v_sub_u32 v14, v1, v12 // (op 521) 
 // (op 521) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 521): v12

 // (op 521) BEGIN: TileTile
 // (op 521) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 521) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 521)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 521)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 521): v12
v_lshlrev_b32 v12, 4, v14 // (op 521) 
 // (op 521) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 521): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 521): v14, v15
v_mov_b32 v15, 0 // (op 521) convert
v_mov_b32 v14, v12 // (op 521) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 521): v12

 // (op 521) BEGIN: Split
 // (op 521) Multiply: dest (DataFlowTag507: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 521)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 521)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag507: 2 VGPRs (Value: UInt64) (op 521): v18, v19
v_mul_lo_u32 v19, v13, s3 // (op 521) most significant: low of low * high
 // (op 521) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 521): v12
v_mul_hi_u32 v12, v13, s2 // (op 521) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 521) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 521) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 521): v12

 // (op 521) Add: dest (DataFlowTag507: VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 521)      lhs (DataFlowTag507: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 521)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 521): s0, s1
v_add_co_u32 v18, s[0:1], v18, v14 // (op 521) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 521) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 521): s0, s1

 // (op 521) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 521): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 521): v13

 // (op 494) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(521) END
 // (op 494) Assign VGPR 64:U64(522) BEGIN
 // (op 522) Assign dim(508) = 64:U64
 // (op 522) Assign stride expression
 // (op 494) Assign VGPR 64:U64(522) END
 // (op 494) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(523) BEGIN
 // (op 523) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 523) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // (op 523) reg expression
 // (op 523) reg expression
 // (op 523) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 523) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 523)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 523)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 523): s0, s1
s_mul_i32 s1, s2, s7 // (op 523) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 523): s14
s_mul_i32 s14, s3, s6 // (op 523) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 523) least significant
// Allocated : 1 SGPR (Value: Int32) (op 523): s15
s_mul_hi_u32 s15, s2, s6 // (op 523) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 523) most significant: sum
s_add_u32 s1, s1, s15 // (op 523) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 523): s15

 // Freeing : 1 SGPR (Value: Int32) (op 523): s14

 // (op 523) reg expression
 // (op 523) reg expression
s_mov_b32 s40, s4 // (op 523) reg expression
s_mov_b32 s41, s5 // (op 523) reg expression
s_mov_b32 s42, s0 // (op 523) reg expression
s_mov_b32 s43, 131072 // (op 523) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 523): s0, s1

 // (op 494) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(523) END
 // (op 494) NOP(514) BEGIN
 // (op 514) generate({})
 // (op 514) end: generate({})
 // (op 494) NOP(514) END
 // (op 494) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(528) BEGIN
 // (op 528) Assign dim(510) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 528) Assign stride expression
 // (op 494) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(528) END
 // (op 494) NOP(515) BEGIN
 // (op 515) generate({})
 // (op 515) end: generate({})
 // (op 494) NOP(515) END
 // (op 494) Assign VGPR 1:U64(531) BEGIN
 // (op 531) Assign dim(512) = 1:U64
 // (op 531) Assign stride expression
 // (op 494) Assign VGPR 1:U64(531) END
 // (op 494) NOP(556) BEGIN
 // (op 556) generate({})
 // (op 556) end: generate({})
 // (op 494) NOP(556) END
 // (op 494) Assign VGPR 48:U64(563) BEGIN
 // (op 563) Assign dim(521) = 48:U64
 // (op 563) Assign stride expression
 // (op 494) Assign VGPR 48:U64(563) END
 // (op 494) NOP(557) BEGIN
 // (op 557) generate({})
 // (op 557) end: generate({})
 // (op 494) NOP(557) END
 // (op 494) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(566) BEGIN
 // (op 566) Assign dim(522) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 522: v**UNALLOCATED**
 // (op 566) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag522: VGPR Value: UInt32 x 1: (unallocated)
 // (op 566) reg expression
 // (op 566) reg expression
 // (op 566) reg expression
 // (op 566) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 566): v12
v_and_b32 v12, -2, v1 // (op 566) 
 // (op 566) BEGIN: Flatten[1]
 // (op 566) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 566): v13
v_sub_u32 v13, v1, v12 // (op 566) 
 // (op 566) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 566): v12

 // (op 566) BEGIN: TileTile
 // (op 566) {TileTile: ShiftL({Flatten[1]: v13:U32}, 5:U32)U32}
 // (op 566) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 566)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 566)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 566): v12
v_lshlrev_b32 v12, 5, v13 // (op 566) 
 // (op 566) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 566): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 566): v14, v15
v_mov_b32 v15, 0 // (op 566) convert
v_mov_b32 v14, v12 // (op 566) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 566): v12

 // (op 566) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 566): v12
v_lshrrev_b32 v12, 1, v1 // (op 566) 
 // (op 566) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 566): v13
v_lshl_add_u32 v13, s17, 5, v12 // (op 566) 
 // (op 566) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 566): v12

 // (op 566) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 566) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 566)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 566)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 566): v20, v21
v_mul_lo_u32 v21, v13, s9 // (op 566) most significant: low of low * high
 // (op 566) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 566): v12
v_mul_hi_u32 v12, v13, s8 // (op 566) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 566) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 566) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 566): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 566): v13

 // (op 566) BEGIN: Split
 // (op 566) {Split: Add({TileTile: v[14:15]:U64}, v[20:21]:I64)U64}
 // (op 566) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 566)      lhs (TileTile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 566)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated : 2 VGPRs (Value: UInt64) (op 566): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 566): s0, s1
v_add_co_u32 v12, s[0:1], v14, v20 // (op 566) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 566) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 566): s0, s1

 // (op 566) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 566): v20, v21

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 566): v14, v15

 // (op 566) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 566) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 566)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 566)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 566) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 566): v14, v15
v_mul_lo_u32 v15, v13, 6 // (op 566) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 566): v20
v_mul_hi_u32 v20, v12, 6 // (op 566) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 566) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 566) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 566): v20

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 566): v12, v13

 // (op 566) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 566): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 566) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 566): v14, v15

// Allocated DataFlowTag522: 1 VGPR (Value: UInt32) (op 566): v14
v_mov_b32 v14, v12 // (op 566) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 566): v12, v13

 // (op 494) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(566) END
 // (op 494) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(567) BEGIN
 // (op 567) Assign dim(523) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 567) Assign stride expression
 // (op 494) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(567) END
 // (op 494) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(568) BEGIN
 // (op 568) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 516: s**UNALLOCATED**
 // (op 568) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: (unallocated)
 // (op 568) reg expression
 // (op 568) reg expression
 // (op 568) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 568) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 568)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 568)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 568) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 568): s0, s1
s_mul_i32 s1, s13, 6 // (op 568) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 568) least significant
// Allocated : 1 SGPR (Value: Int32) (op 568): s14
s_mul_hi_u32 s14, s12, 6 // (op 568) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 568) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 568): s14

 // (op 568) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 568) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 568)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 568)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 568): s14, s15
s_mul_i32 s15, s8, s1 // (op 568) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 568): s18
s_mul_i32 s18, s9, s0 // (op 568) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 568) least significant
// Allocated : 1 SGPR (Value: Int32) (op 568): s19
s_mul_hi_u32 s19, s8, s0 // (op 568) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 568) most significant: sum
s_add_u32 s15, s15, s19 // (op 568) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 568): s19

 // Freeing : 1 SGPR (Value: Int32) (op 568): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 568): s0, s1

 // (op 568) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 568): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 568) 
 // Freeing : 2 SGPRs (Value: Int64) (op 568): s14, s15

 // (op 568) reg expression
 // (op 568) reg expression
// Allocated DataFlowTag516: 4 SGPRs (Buffer: None) (op 568): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 568) reg expression
s_mov_b32 s45, s11 // (op 568) reg expression
s_mov_b32 s46, s0 // (op 568) reg expression
s_mov_b32 s47, 131072 // (op 568) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 568): s0, s1

 // (op 494) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(568) END
 // (op 494) NOP(558) BEGIN
 // (op 558) generate({})
 // (op 558) end: generate({})
 // (op 494) NOP(558) END
 // (op 494) Assign VGPR 0:U64(573) BEGIN
 // (op 573) Assign dim(525) = 0:U64
 // (op 573) Assign stride expression
 // (op 494) Assign VGPR 0:U64(573) END
 // (op 494) NOP(576) BEGIN
 // (op 576) generate({})
 // (op 576) end: generate({})
 // (op 494) NOP(576) END
 // (op 494) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(585) BEGIN
 // (op 585) Assign dim(526) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 526: v**UNALLOCATED**
 // (op 585) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag526: VGPR Value: UInt32 x 1: (unallocated)
 // (op 585) reg expression
 // (op 585) reg expression
 // (op 585) reg expression
 // (op 585) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 585): v15
v_and_b32 v15, -2, v1 // (op 585) 
 // (op 585) BEGIN: Flatten[1]
 // (op 585) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 585): v12
v_sub_u32 v12, v1, v15 // (op 585) 
 // (op 585) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 585): v15

 // (op 585) BEGIN: TileTile
 // (op 585) {TileTile: ShiftL({Flatten[1]: v12:U32}, 5:U32)U32}
 // (op 585) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 585)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 585)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 585): v13
v_lshlrev_b32 v13, 5, v12 // (op 585) 
 // (op 585) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 585): v12

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 585): v20, v21
v_mov_b32 v21, 0 // (op 585) convert
v_mov_b32 v20, v13 // (op 585) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 585): v13

 // (op 585) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 585): v15
v_lshrrev_b32 v15, 1, v1 // (op 585) 
 // (op 585) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 585): v12
v_lshl_add_u32 v12, s17, 5, v15 // (op 585) 
 // (op 585) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 585): v15

 // (op 585) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 585) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 585)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 585)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 585): v22, v23
v_mul_lo_u32 v23, v12, s9 // (op 585) most significant: low of low * high
 // (op 585) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 585): v13
v_mul_hi_u32 v13, v12, s8 // (op 585) most significant: high of low * low
v_mul_lo_u32 v22, v12, s8 // (op 585) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 585) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 585): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 585): v12

 // (op 585) BEGIN: Split
 // (op 585) {Split: Add({TileTile: v[20:21]:U64}, v[22:23]:I64)U64}
 // (op 585) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 585)      lhs (TileTile: VGPR Value: UInt64 x 1: v[20:21]) 
 // (op 585)      rhs (VGPR Value: Int64 x 1: v[22:23])
// Allocated : 2 VGPRs (Value: UInt64) (op 585): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 585): s0, s1
v_add_co_u32 v12, s[0:1], v20, v22 // (op 585) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 585) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 585): s0, s1

 // (op 585) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 585): v22, v23

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 585): v20, v21

 // (op 585) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 585) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 585)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 585)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 585) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 585): v20, v21
v_mul_lo_u32 v21, v13, 6 // (op 585) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 585): v15
v_mul_hi_u32 v15, v12, 6 // (op 585) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 585) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 585) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 585): v15

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 585): v12, v13

 // (op 585) LogicalShiftR(v[20:21]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 585): v12, v13
v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 585) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 585): v20, v21

// Allocated DataFlowTag526: 1 VGPR (Value: UInt32) (op 585): v15
v_mov_b32 v15, v12 // (op 585) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 585): v12, v13

 // (op 494) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(585) END
 // (op 494) Assign VGPR 48:U64(586) BEGIN
 // (op 586) Assign dim(527) = 48:U64
 // (op 586) Assign stride expression
 // (op 494) Assign VGPR 48:U64(586) END
 // (op 494) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(587) BEGIN
 // (op 587) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 587) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // (op 587) reg expression
 // (op 587) reg expression
 // (op 587) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 587) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 587)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 587)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 587) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 587): s0, s1
s_mul_i32 s1, s13, 6 // (op 587) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 587) least significant
// Allocated : 1 SGPR (Value: Int32) (op 587): s14
s_mul_hi_u32 s14, s12, 6 // (op 587) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 587) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 587): s14

 // (op 587) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 587) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 587)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 587)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 587): s14, s15
s_mul_i32 s15, s8, s1 // (op 587) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 587): s18
s_mul_i32 s18, s9, s0 // (op 587) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 587) least significant
// Allocated : 1 SGPR (Value: Int32) (op 587): s19
s_mul_hi_u32 s19, s8, s0 // (op 587) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 587) most significant: sum
s_add_u32 s15, s15, s19 // (op 587) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 587): s19

 // Freeing : 1 SGPR (Value: Int32) (op 587): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 587): s0, s1

 // (op 587) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 587): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 587) 
 // Freeing : 2 SGPRs (Value: Int64) (op 587): s14, s15

 // (op 587) reg expression
 // (op 587) reg expression
s_mov_b32 s44, s10 // (op 587) reg expression
s_mov_b32 s45, s11 // (op 587) reg expression
s_mov_b32 s46, s0 // (op 587) reg expression
s_mov_b32 s47, 131072 // (op 587) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 587): s0, s1

 // (op 494) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(587) END
 // (op 494) NOP(578) BEGIN
 // (op 578) generate({})
 // (op 578) end: generate({})
 // (op 494) NOP(578) END
 // (op 494) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(592) BEGIN
 // (op 592) Assign dim(529) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 592) Assign stride expression
 // (op 494) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(592) END
 // (op 494) NOP(579) BEGIN
 // (op 579) generate({})
 // (op 579) end: generate({})
 // (op 494) NOP(579) END
 // (op 494) Assign VGPR 0:U64(595) BEGIN
 // (op 595) Assign dim(531) = 0:U64
 // (op 595) Assign stride expression
 // (op 494) Assign VGPR 0:U64(595) END
 // (op 494) NOP(620) BEGIN
 // (op 620) generate({})
 // (op 620) end: generate({})
 // (op 494) NOP(620) END
 // (op 494) Assign VGPR 2:U64(627) BEGIN
 // (op 627) Assign dim(540) = 2:U64
 // (op 627) Assign stride expression
 // (op 494) Assign VGPR 2:U64(627) END
 // (op 494) NOP(621) BEGIN
 // (op 621) generate({})
 // (op 621) end: generate({})
 // (op 494) NOP(621) END
 // (op 494) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(630) BEGIN
 // (op 630) Assign dim(541) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}
 // tag 541: v**UNALLOCATED**
 // (op 630) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32} into DataFlowTag541: VGPR Value: UInt32 x 1: (unallocated)
 // (op 630) reg expression
 // (op 630) reg expression
 // (op 630) reg expression
 // (op 630) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 630) convert
// Allocated : 1 VGPR (Value: UInt32) (op 630): v12
v_lshrrev_b32 v12, 1, v1 // (op 630) 
 // (op 630) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 630): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 630) 
 // (op 630) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 630): v12

 // (op 630) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 630): v12
v_and_b32 v12, -2, v1 // (op 630) 
 // (op 630) BEGIN: Flatten[1]TileTile
 // (op 630) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 630): v20
v_sub_u32 v20, v1, v12 // (op 630) 
 // (op 630) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 630): v12

 // (op 630) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag541: 1 VGPR (Value: UInt32) (op 630): v12
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v12, v13, s20 // (op 630) 
v_add_u32 v12, v12, v20 // (op 630) 
 // (op 630) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 630): v20

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 630): v13

 // (op 494) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(630) END
 // (op 494) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(631) BEGIN
 // (op 631) Assign dim(542) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 631) Assign stride expression
 // (op 494) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(631) END
 // (op 494) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(632) BEGIN
 // (op 632) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 535: s**UNALLOCATED**
 // (op 632) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: (unallocated)
 // (op 632) reg expression
 // (op 632) reg expression
 // (op 632) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 632) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 632)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 632)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 632): s0, s1
s_mul_i32 s1, s20, s7 // (op 632) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 632): s14
s_mul_i32 s14, s21, s6 // (op 632) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 632) least significant
// Allocated : 1 SGPR (Value: Int32) (op 632): s15
s_mul_hi_u32 s15, s20, s6 // (op 632) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 632) most significant: sum
s_add_u32 s1, s1, s15 // (op 632) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 632): s15

 // Freeing : 1 SGPR (Value: Int32) (op 632): s14

 // (op 632) reg expression
 // (op 632) reg expression
// Allocated DataFlowTag535: 4 SGPRs (Buffer: None) (op 632): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 632) reg expression
s_mov_b32 s49, s23 // (op 632) reg expression
s_mov_b32 s50, s0 // (op 632) reg expression
s_mov_b32 s51, 131072 // (op 632) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 632): s0, s1

 // (op 494) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(632) END
 // (op 494) NOP(622) BEGIN
 // (op 622) generate({})
 // (op 622) end: generate({})
 // (op 494) NOP(622) END
 // (op 494) Assign VGPR 1:U64(637) BEGIN
 // (op 637) Assign dim(544) = 1:U64
 // (op 637) Assign stride expression
 // (op 494) Assign VGPR 1:U64(637) END
 // (op 494) NOP(640) BEGIN
 // (op 640) generate({})
 // (op 640) end: generate({})
 // (op 494) NOP(640) END
 // (op 494) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(649) BEGIN
 // (op 649) Assign dim(545) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}
 // tag 545: v**UNALLOCATED**
 // (op 649) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32} into DataFlowTag545: VGPR Value: UInt32 x 1: (unallocated)
 // (op 649) reg expression
 // (op 649) reg expression
 // (op 649) reg expression
 // (op 649) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 649) convert
// Allocated : 1 VGPR (Value: UInt32) (op 649): v13
v_lshrrev_b32 v13, 1, v1 // (op 649) 
 // (op 649) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 649): v20
v_lshl_add_u32 v20, s16, 5, v13 // (op 649) 
 // (op 649) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 649): v13

 // (op 649) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 649): v13
v_and_b32 v13, -2, v1 // (op 649) 
 // (op 649) BEGIN: Flatten[1]TileTile
 // (op 649) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 649): v21
v_sub_u32 v21, v1, v13 // (op 649) 
 // (op 649) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 649): v13

 // (op 649) BEGIN: Split
// Allocated DataFlowTag545: 1 VGPR (Value: UInt32) (op 649): v13
v_mul_lo_u32 v13, v20, s20 // (op 649) 
v_add_u32 v13, v13, v21 // (op 649) 
 // (op 649) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 649): v21

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 649): v20

 // (op 494) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(649) END
 // (op 494) Assign VGPR 2:U64(650) BEGIN
 // (op 650) Assign dim(546) = 2:U64
 // (op 650) Assign stride expression
 // (op 494) Assign VGPR 2:U64(650) END
 // (op 494) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(651) BEGIN
 // (op 651) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 651) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // (op 651) reg expression
 // (op 651) reg expression
 // (op 651) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 651) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 651)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 651)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 651): s0, s1
s_mul_i32 s1, s20, s7 // (op 651) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 651): s14
s_mul_i32 s14, s21, s6 // (op 651) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 651) least significant
// Allocated : 1 SGPR (Value: Int32) (op 651): s15
s_mul_hi_u32 s15, s20, s6 // (op 651) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 651) most significant: sum
s_add_u32 s1, s1, s15 // (op 651) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 651): s15

 // Freeing : 1 SGPR (Value: Int32) (op 651): s14

 // (op 651) reg expression
 // (op 651) reg expression
s_mov_b32 s48, s22 // (op 651) reg expression
s_mov_b32 s49, s23 // (op 651) reg expression
s_mov_b32 s50, s0 // (op 651) reg expression
s_mov_b32 s51, 131072 // (op 651) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 651): s0, s1

 // (op 494) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(651) END
 // (op 494) NOP(642) BEGIN
 // (op 642) generate({})
 // (op 642) end: generate({})
 // (op 494) NOP(642) END
 // (op 494) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(656) BEGIN
 // (op 656) Assign dim(548) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 656) Assign stride expression
 // (op 494) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(656) END
 // (op 494) NOP(643) BEGIN
 // (op 643) generate({})
 // (op 643) end: generate({})
 // (op 494) NOP(643) END
 // (op 494) Assign VGPR 1:U64(659) BEGIN
 // (op 659) Assign dim(550) = 1:U64
 // (op 659) Assign stride expression
 // (op 494) Assign VGPR 1:U64(659) END
 // (op 494) NOP(684) BEGIN
 // (op 684) generate({})
 // (op 684) end: generate({})
 // (op 494) NOP(684) END
 // (op 494) Assign VGPR 2:U64(691) BEGIN
 // (op 691) Assign dim(559) = 2:U64
 // (op 691) Assign stride expression
 // (op 494) Assign VGPR 2:U64(691) END
 // (op 494) NOP(685) BEGIN
 // (op 685) generate({})
 // (op 685) end: generate({})
 // (op 494) NOP(685) END
 // (op 494) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(694) BEGIN
 // (op 694) Assign dim(560) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 560: v**UNALLOCATED**
 // (op 694) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 694) reg expression
 // (op 694) reg expression
 // (op 694) reg expression
 // (op 694) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 694) convert
// Allocated : 1 VGPR (Value: UInt32) (op 694): v20
v_and_b32 v20, -2, v1 // (op 694) 
 // (op 694) BEGIN: Flatten[1]TileTile
 // (op 694) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 694): v21
v_sub_u32 v21, v1, v20 // (op 694) 
 // (op 694) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 694): v20

 // (op 694) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 694): v20
v_lshrrev_b32 v20, 1, v1 // (op 694) 
 // (op 694) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 694): v22
v_lshl_add_u32 v22, s17, 5, v20 // (op 694) 
 // (op 694) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 694): v20

 // (op 694) Multiply({Tile: v22:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 694): v20
v_mul_lo_u32 v20, v22, s24 // (op 694) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 694): v22

 // (op 694) BEGIN: Split
 // (op 694) {Split: Add({Flatten[1]TileTile: v21:U32}, v20:U32)U32}
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 694): v22
v_add_u32 v22, v21, v20 // (op 694) 
 // (op 694) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 694): v20

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 694): v21

 // (op 494) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(694) END
 // (op 494) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(695) BEGIN
 // (op 695) Assign dim(561) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 695) Assign stride expression
 // (op 494) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(695) END
 // (op 494) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(696) BEGIN
 // (op 696) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 554: s**UNALLOCATED**
 // (op 696) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: (unallocated)
 // (op 696) reg expression
 // (op 696) reg expression
 // (op 696) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 696) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 696)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 696)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 696): s0, s1
s_mul_i32 s1, s24, s13 // (op 696) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 696): s14
s_mul_i32 s14, s25, s12 // (op 696) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 696) least significant
// Allocated : 1 SGPR (Value: Int32) (op 696): s15
s_mul_hi_u32 s15, s24, s12 // (op 696) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 696) most significant: sum
s_add_u32 s1, s1, s15 // (op 696) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 696): s15

 // Freeing : 1 SGPR (Value: Int32) (op 696): s14

 // (op 696) reg expression
 // (op 696) reg expression
// Allocated DataFlowTag554: 4 SGPRs (Buffer: None) (op 696): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 696) reg expression
s_mov_b32 s53, s27 // (op 696) reg expression
s_mov_b32 s54, s0 // (op 696) reg expression
s_mov_b32 s55, 131072 // (op 696) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 696): s0, s1

 // (op 494) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(696) END
 // (op 494) NOP(686) BEGIN
 // (op 686) generate({})
 // (op 686) end: generate({})
 // (op 494) NOP(686) END
 // (op 494) Assign VGPR 1:U64(701) BEGIN
 // (op 701) Assign dim(563) = 1:U64
 // (op 701) Assign stride expression
 // (op 494) Assign VGPR 1:U64(701) END
 // (op 494) NOP(704) BEGIN
 // (op 704) generate({})
 // (op 704) end: generate({})
 // (op 494) NOP(704) END
 // (op 494) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(713) BEGIN
 // (op 713) Assign dim(564) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 564: v**UNALLOCATED**
 // (op 713) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag564: VGPR Value: UInt32 x 1: (unallocated)
 // (op 713) reg expression
 // (op 713) reg expression
 // (op 713) reg expression
 // (op 713) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 713) convert
// Allocated : 1 VGPR (Value: UInt32) (op 713): v20
v_and_b32 v20, -2, v1 // (op 713) 
 // (op 713) BEGIN: Flatten[1]TileTile
 // (op 713) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 713): v21
v_sub_u32 v21, v1, v20 // (op 713) 
 // (op 713) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 713): v20

 // (op 713) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 713): v20
v_lshrrev_b32 v20, 1, v1 // (op 713) 
 // (op 713) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 713): v23
v_lshl_add_u32 v23, s17, 5, v20 // (op 713) 
 // (op 713) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 713): v20

 // (op 713) Multiply({Tile: v23:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 713): v20
v_mul_lo_u32 v20, v23, s24 // (op 713) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 713): v23

 // (op 713) BEGIN: Split
 // (op 713) {Split: Add({Flatten[1]TileTile: v21:U32}, v20:U32)U32}
// Allocated DataFlowTag564: 1 VGPR (Value: UInt32) (op 713): v23
v_add_u32 v23, v21, v20 // (op 713) 
 // (op 713) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 713): v20

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 713): v21

 // (op 494) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(713) END
 // (op 494) Assign VGPR 2:U64(714) BEGIN
 // (op 714) Assign dim(565) = 2:U64
 // (op 714) Assign stride expression
 // (op 494) Assign VGPR 2:U64(714) END
 // (op 494) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(715) BEGIN
 // (op 715) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 715) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // (op 715) reg expression
 // (op 715) reg expression
 // (op 715) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 715) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 715)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 715)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 715): s0, s1
s_mul_i32 s1, s24, s13 // (op 715) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 715): s14
s_mul_i32 s14, s25, s12 // (op 715) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 715) least significant
// Allocated : 1 SGPR (Value: Int32) (op 715): s15
s_mul_hi_u32 s15, s24, s12 // (op 715) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 715) most significant: sum
s_add_u32 s1, s1, s15 // (op 715) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 715): s15

 // Freeing : 1 SGPR (Value: Int32) (op 715): s14

 // (op 715) reg expression
 // (op 715) reg expression
s_mov_b32 s52, s26 // (op 715) reg expression
s_mov_b32 s53, s27 // (op 715) reg expression
s_mov_b32 s54, s0 // (op 715) reg expression
s_mov_b32 s55, 131072 // (op 715) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 715): s0, s1

 // (op 494) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(715) END
 // (op 494) NOP(706) BEGIN
 // (op 706) generate({})
 // (op 706) end: generate({})
 // (op 494) NOP(706) END
 // (op 494) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(720) BEGIN
 // (op 720) Assign dim(567) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 720) Assign stride expression
 // (op 494) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(720) END
 // (op 494) NOP(707) BEGIN
 // (op 707) generate({})
 // (op 707) end: generate({})
 // (op 494) NOP(707) END
 // (op 494) Assign VGPR 1:U64(723) BEGIN
 // (op 723) Assign dim(569) = 1:U64
 // (op 723) Assign stride expression
 // (op 494) Assign VGPR 1:U64(723) END
 // (op 494) NOP(364) BEGIN
 // (op 364) generate({})
 // (op 364) end: generate({})
 // (op 494) NOP(364) END
 // (op 494) SetCoordinate(369) BEGIN
 // (op 369) generate({})
 // (op 369) end: generate({})
 // (op 369) generate({370})
 // (op 369) LoadTiled Value: FP6x16(370) BEGIN
 // (op 370) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 370) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 370) Generate 0:U64 into nullptr
 // tag 9: v**UNALLOCATED**
 // (op 370) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 370) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag522: VGPR Value: UInt32 x 1: v14
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 370) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 370): v24, v25, v26, v27, v28, v29
buffer_load_dwordx4 v[24:27], v14, s[44:47], 0 offen // (op 370) Load value
// VMEM: Expected complete at 811 (current 331)
buffer_load_dwordx2 v[28:29], v14, s[44:47], 0 offen offset:16 // (op 370) Load value
// VMEM: Expected complete at 812 (current 332)
 // (op 369) LoadTiled Value: FP6x16(370) END
 // (op 369) end: generate({370})
 // (op 494) SetCoordinate(369) END
 // (op 494) SetCoordinate(372) BEGIN
 // (op 372) generate({})
 // (op 372) end: generate({})
 // (op 372) generate({373})
 // (op 372) LoadTiled Value: E8M0(373) BEGIN
 // (op 373) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 373) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 373) Generate 1:U64 into nullptr
 // tag 11: v**UNALLOCATED**
 // (op 373) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 373) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v22
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 373) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 373): v20
buffer_load_ubyte v20, v22, s[52:55], 0 offen offset: 0 // (op 373) Load value
// VMEM: Expected complete at 813 (current 333)
 // (op 372) LoadTiled Value: E8M0(373) END
 // (op 372) end: generate({373})
 // (op 494) SetCoordinate(372) END
 // (op 494) SetCoordinate(375) BEGIN
 // (op 375) generate({})
 // (op 375) end: generate({})
 // (op 375) generate({775})
 // (op 375) Scope(775) BEGIN
 // (op 775) Lock Scope 775
 // Lock Branch
 // (op 775) generate({772})
 // (op 775) NOP(772) BEGIN
 // (op 772) generate({})
 // (op 772) end: generate({})
 // (op 775) NOP(772) END
 // (op 775) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(779) BEGIN
 // (op 779) Assign dim(584) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 584: s**UNALLOCATED**
 // (op 779) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag584: SGPR Value: UInt32 x 1: (unallocated)
 // (op 779) reg expression
 // (op 779) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 779): v21
v_and_b32 v21, -4, v1 // (op 779) 
 // (op 779) BEGIN: Tile[1]
 // (op 779) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v21:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 779): v30
v_sub_u32 v30, v1, v21 // (op 779) 
 // (op 779) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 779): v21

 // (op 779) BEGIN: Flatten
 // (op 779) {Flatten: ShiftL({Tile[1]: v30:U32}, 4:U32)U32}
 // (op 779) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 779)         value (Tile[1]: VGPR Value: UInt32 x 1: v30) 
 // (op 779)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 779): v21
v_lshlrev_b32 v21, 4, v30 // (op 779) 
 // (op 779) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 779): v30

 // (op 779) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 779): v30
v_lshrrev_b32 v30, 2, v1 // (op 779) 
 // (op 779) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 779): v31
v_lshl_add_u32 v31, v30, 6, v21 // (op 779) 
 // (op 779) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 779): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 779): v30

// Allocated DataFlowTag584: 1 SGPR (Value: UInt32) (op 779): s0
s_nop 0
v_readfirstlane_b32 s0, v31 // (op 779) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 779): v31

 // (op 775) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(779) END
 // (op 775) Assign VGPR 1024:U32(780) BEGIN
 // (op 780) Assign dim(585) = 1024:U32
 // (op 780) Assign stride expression
 // (op 775) Assign VGPR 1024:U32(780) END
 // (op 775) NOP(773) BEGIN
 // (op 773) generate({})
 // (op 773) end: generate({})
 // (op 775) NOP(773) END
 // (op 775) Assign VGPR 1:U32(784) BEGIN
 // (op 784) Assign dim(587) = 1:U32
 // (op 784) Assign stride expression
 // (op 775) Assign VGPR 1:U32(784) END
 // (op 775) LoadTileDirect2LDS Value: BF8x4(458) BEGIN
 // (op 458) GEN: LoadTileDirect2LDS
 // (op 458) GEN: loadMacroTileDirect2LDS OP 458 LDS 17 MacroTile 8
 // (op 458) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 458) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride504: VGPR Value: UInt64 x 1: (unallocated)
 // (op 458) reg expression
 // (op 458) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 458) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 458)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 458)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 458): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 458) 
// Allocated Stride504: 2 VGPRs (Value: UInt64) (op 458): v30, v31
v_mov_b64 v[30:31], s[14:15] // (op 458) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 458): s14, s15

 // (op 458) Generate 1:U64 into nullptr
 // (op 458) Generate 1024:U32 into nullptr
 // (op 458) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride504: VGPR Value: UInt64 x 1: v[30:31]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 458): v32, v33
v_mov_b64 v[32:33], v[16:17] // (op 458) 
 // (op 458) Generate Add(0:U32, {DataFlowTag584Flatten: s0:U32})U32 into nullptr
 // (op 458)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride504: v[30:31]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 458) Generate {DataFlowTag584Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 458) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 458) Load value direct to lds
// VMEM: Expected stall of 467. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1292 (current 812)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 458) Generate Add(v[32:33]:U64, v30:R)U64 into VGPR Value: UInt64 x 1: v[32:33]
 // (op 458) reg expression
 // (op 458) reg expression
 // (op 458) Add(v[32:33]:U64, v30:R)U64
 // (op 458) Add: dest (VGPR Value: UInt64 x 1: v[32:33]) = 
 // (op 458)      lhs (VGPR Value: UInt64 x 1: v[32:33]) 
 // (op 458)      rhs (VGPR Value: Raw32 x 1: v30)
// Allocated : 2 SGPRs (Value: Bool64) (op 458): s14, s15
v_add_co_u32 v32, s[14:15], v32, v30 // (op 458) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 458) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 458): s14, s15

 // (op 458) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 458) reg expression
 // (op 458) Add(m0:U32, 1024:U32)U32
 // (op 458) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 458) 
 // (op 458) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 458) Load value direct to lds
// VMEM: Expected complete at 1296 (current 816)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing Stride504: 2 VGPRs (Value: UInt64) (op 458): v30, v31

 // Freeing : 2 VGPRs (Value: UInt64) (op 458): v32, v33

 // (op 775) LoadTileDirect2LDS Value: BF8x4(458) END
 // (op 775) Deallocate{}(1319) BEGIN
 // (op 1319) Deallocate 586
 // Deleting tag 586
 // (op 1319) Deallocate 584
 // Freeing DataFlowTag584: 1 SGPR (Value: UInt32) (op 779): s0

 // Deleting tag 584
 // (op 1319) Deallocate 585
 // Deleting tag 585
 // (op 1319) Deallocate 587
 // Deleting tag 587
 // (op 775) Deallocate{}(1319) END
 // (op 775) end: generate({772})
 // (op 775) Unlock Scope 775
 // Unlock None
 // (op 375) Scope(775) END
 // (op 375) end: generate({775})
 // (op 494) SetCoordinate(375) END
 // (op 494) SetCoordinate(378) BEGIN
 // (op 378) generate({})
 // (op 378) end: generate({})
 // (op 378) generate({379})
 // (op 378) LoadTiled Value: E8M0(379) BEGIN
 // (op 379) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 379) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 379) Generate 1:U64 into nullptr
 // tag 10: v**UNALLOCATED**
 // (op 379) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 379) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag541: VGPR Value: UInt32 x 1: v12
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 379) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 379): v21
buffer_load_ubyte v21, v12, s[48:51], 0 offen offset: 0 // (op 379) Load value
// VMEM: Expected complete at 1297 (current 817)
 // (op 378) LoadTiled Value: E8M0(379) END
 // (op 378) end: generate({379})
 // (op 494) SetCoordinate(378) END
 // (op 494) SetCoordinate(381) BEGIN
 // (op 381) generate({})
 // (op 381) end: generate({})
 // (op 381) generate({382})
 // (op 381) LoadTiled Value: FP6x16(382) BEGIN
 // (op 382) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 382) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 382) Generate 0:U64 into nullptr
 // tag 474: v**UNALLOCATED**
 // (op 382) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 382) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag522: VGPR Value: UInt32 x 1: v14
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 48
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 382) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 382): v30, v31, v32, v33, v34, v35
buffer_load_dwordx4 v[30:33], v14, s[44:47], 0 offen offset:48 // (op 382) Load value
// VMEM: Expected stall of 475. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1773 (current 1293)
buffer_load_dwordx2 v[34:35], v14, s[44:47], 0 offen offset:64 // (op 382) Load value
// VMEM: Expected stall of 3. CBNW: 4, Inc: 3
// VMEM: Expected complete at 1777 (current 1297)
 // (op 381) LoadTiled Value: FP6x16(382) END
 // (op 381) end: generate({382})
 // (op 494) SetCoordinate(381) END
 // (op 494) Deallocate{}(1299) BEGIN
 // (op 1299) Deallocate 522
 // Freeing DataFlowTag522: 1 VGPR (Value: UInt32) (op 566): v14

 // Deleting tag 522
 // (op 1299) Deallocate 521
 // Deleting tag 521
 // (op 1299) Deallocate 523
 // Deleting tag 523
 // (op 1299) Deallocate 525
 // Deleting tag 525
 // (op 1299) Deallocate 524
 // Deleting tag 524
 // (op 494) Deallocate{}(1299) END
 // (op 494) SetCoordinate(384) BEGIN
 // (op 384) generate({})
 // (op 384) end: generate({})
 // (op 384) generate({385})
 // (op 384) LoadTiled Value: E8M0(385) BEGIN
 // (op 385) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 385) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 385) Generate 1:U64 into nullptr
 // tag 482: v**UNALLOCATED**
 // (op 385) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 385) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v22
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 385) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 385): v14
buffer_load_ubyte v14, v22, s[52:55], 0 offen offset:2 // (op 385) Load value
// VMEM: Expected complete at 1778 (current 1298)
 // (op 384) LoadTiled Value: E8M0(385) END
 // (op 384) end: generate({385})
 // (op 494) SetCoordinate(384) END
 // (op 494) Deallocate{}(1301) BEGIN
 // (op 1301) Deallocate 562
 // Deleting tag 562
 // (op 1301) Deallocate 563
 // Deleting tag 563
 // (op 1301) Deallocate 561
 // Deleting tag 561
 // (op 1301) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 694): v22

 // Deleting tag 560
 // (op 1301) Deallocate 559
 // Deleting tag 559
 // (op 494) Deallocate{}(1301) END
 // (op 494) SetCoordinate(387) BEGIN
 // (op 387) generate({})
 // (op 387) end: generate({})
 // (op 387) generate({1007})
 // (op 387) Scope(1007) BEGIN
 // (op 1007) Lock Scope 1007
 // Lock Branch
 // (op 1007) generate({1004})
 // (op 1007) NOP(1004) BEGIN
 // (op 1004) generate({})
 // (op 1004) end: generate({})
 // (op 1007) NOP(1004) END
 // (op 1007) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1011) BEGIN
 // (op 1011) Assign dim(654) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 654: s**UNALLOCATED**
 // (op 1011) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag654: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1011) reg expression
 // (op 1011) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1011): v22
v_and_b32 v22, -4, v1 // (op 1011) 
 // (op 1011) BEGIN: Tile[1]
 // (op 1011) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v22:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1011): v36
v_sub_u32 v36, v1, v22 // (op 1011) 
 // (op 1011) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1011): v22

 // (op 1011) BEGIN: Flatten
 // (op 1011) {Flatten: ShiftL({Tile[1]: v36:U32}, 4:U32)U32}
 // (op 1011) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1011)         value (Tile[1]: VGPR Value: UInt32 x 1: v36) 
 // (op 1011)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1011): v22
v_lshlrev_b32 v22, 4, v36 // (op 1011) 
 // (op 1011) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1011): v36

 // (op 1011) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1011): v36
v_lshrrev_b32 v36, 2, v1 // (op 1011) 
 // (op 1011) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1011): v37
v_lshl_add_u32 v37, v36, 6, v22 // (op 1011) 
 // (op 1011) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1011): v22

 // Freeing : 1 VGPR (Value: UInt32) (op 1011): v36

// Allocated DataFlowTag654: 1 SGPR (Value: UInt32) (op 1011): s0
s_nop 0
v_readfirstlane_b32 s0, v37 // (op 1011) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1011): v37

 // (op 1007) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1011) END
 // (op 1007) Assign VGPR 1024:U32(1012) BEGIN
 // (op 1012) Assign dim(655) = 1024:U32
 // (op 1012) Assign stride expression
 // (op 1007) Assign VGPR 1024:U32(1012) END
 // (op 1007) NOP(1005) BEGIN
 // (op 1005) generate({})
 // (op 1005) end: generate({})
 // (op 1007) NOP(1005) END
 // (op 1007) Assign VGPR 1:U32(1016) BEGIN
 // (op 1016) Assign dim(657) = 1:U32
 // (op 1016) Assign stride expression
 // (op 1007) Assign VGPR 1:U32(1016) END
 // (op 1007) LoadTileDirect2LDS Value: BF8x4(463) BEGIN
 // (op 463) GEN: LoadTileDirect2LDS
 // (op 463) GEN: loadMacroTileDirect2LDS OP 463 LDS 472 MacroTile 470
 // (op 463) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 463) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride504: VGPR Value: UInt64 x 1: (unallocated)
 // (op 463) reg expression
 // (op 463) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 463) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 463)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 463)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 463): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 463) 
// Allocated Stride504: 2 VGPRs (Value: UInt64) (op 463): v36, v37
v_mov_b64 v[36:37], s[14:15] // (op 463) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 463): s14, s15

 // (op 463) Generate 1:U64 into nullptr
 // (op 463) Generate 1024:U32 into nullptr
 // (op 463) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride504: VGPR Value: UInt64 x 1: v[36:37]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 463): v38, v39
v_mov_b64 v[38:39], v[16:17] // (op 463) 
 // (op 463) Generate Add(2048:U32, {DataFlowTag654Flatten: s0:U32})U32 into nullptr
 // (op 463) reg expression
 // (op 463) Add(2048:U32, {DataFlowTag654Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 463): s1
s_add_u32 s1, 2048, s0 // (op 463) 
 // (op 463)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride504: v[36:37]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 463) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 463) reg expression
buffer_load_dwordx4 v38, s[40:43], 64 offen offset: 0 lds // (op 463) Load value direct to lds
// VMEM: Expected stall of 463. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2254 (current 1774)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 463) Generate Add(v[38:39]:U64, v36:R)U64 into VGPR Value: UInt64 x 1: v[38:39]
 // (op 463) reg expression
 // (op 463) reg expression
 // (op 463) Add(v[38:39]:U64, v36:R)U64
 // (op 463) Add: dest (VGPR Value: UInt64 x 1: v[38:39]) = 
 // (op 463)      lhs (VGPR Value: UInt64 x 1: v[38:39]) 
 // (op 463)      rhs (VGPR Value: Raw32 x 1: v36)
// Allocated : 2 SGPRs (Value: Bool64) (op 463): s14, s15
v_add_co_u32 v38, s[14:15], v38, v36 // (op 463) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 463) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 463): s14, s15

 // (op 463) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 463) reg expression
 // (op 463) Add(m0:U32, 1024:U32)U32
 // (op 463) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 463) 
 // (op 463) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v38, s[40:43], 64 offen offset: 0 lds // (op 463) Load value direct to lds
// VMEM: Expected complete at 2258 (current 1778)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing Stride504: 2 VGPRs (Value: UInt64) (op 463): v36, v37

 // Freeing : 2 VGPRs (Value: UInt64) (op 463): v38, v39

 // Freeing : 1 SGPR (Value: UInt32) (op 463): s1

 // (op 1007) LoadTileDirect2LDS Value: BF8x4(463) END
 // (op 1007) Deallocate{}(1321) BEGIN
 // (op 1321) Deallocate 654
 // Freeing DataFlowTag654: 1 SGPR (Value: UInt32) (op 1011): s0

 // Deleting tag 654
 // (op 1321) Deallocate 656
 // Deleting tag 656
 // (op 1321) Deallocate 655
 // Deleting tag 655
 // (op 1321) Deallocate 657
 // Deleting tag 657
 // (op 1007) Deallocate{}(1321) END
 // (op 1007) end: generate({1004})
 // (op 1007) Unlock Scope 1007
 // Unlock None
 // (op 387) Scope(1007) END
 // (op 387) end: generate({1007})
 // (op 494) SetCoordinate(387) END
 // (op 494) Deallocate{}(1303) BEGIN
 // (op 1303) Deallocate 504
 // Deleting tag 504
 // (op 1303) Deallocate 502
 // Deleting tag 502
 // (op 1303) Deallocate 503
 // Freeing DataFlowTag503: 2 VGPRs (Value: UInt64) (op 502): v16, v17

 // Deleting tag 503
 // (op 1303) Deallocate 506
 // Deleting tag 506
 // (op 1303) Deallocate 505
 // Deleting tag 505
 // (op 494) Deallocate{}(1303) END
 // (op 494) SetCoordinate(390) BEGIN
 // (op 390) generate({})
 // (op 390) end: generate({})
 // (op 390) generate({391})
 // (op 390) LoadTiled Value: E8M0(391) BEGIN
 // (op 391) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 391) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 391) Generate 1:U64 into nullptr
 // tag 478: v**UNALLOCATED**
 // (op 391) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 391) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag541: VGPR Value: UInt32 x 1: v12
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 391) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 391): v22
buffer_load_ubyte v22, v12, s[48:51], 0 offen offset:2 // (op 391) Load value
// VMEM: Expected complete at 2259 (current 1779)
 // (op 390) LoadTiled Value: E8M0(391) END
 // (op 390) end: generate({391})
 // (op 494) SetCoordinate(390) END
 // (op 494) Deallocate{}(1305) BEGIN
 // (op 1305) Deallocate 540
 // Deleting tag 540
 // (op 1305) Deallocate 542
 // Deleting tag 542
 // (op 1305) Deallocate 541
 // Freeing DataFlowTag541: 1 VGPR (Value: UInt32) (op 630): v12

 // Deleting tag 541
 // (op 1305) Deallocate 544
 // Deleting tag 544
 // (op 1305) Deallocate 543
 // Deleting tag 543
 // (op 494) Deallocate{}(1305) END
 // (op 494) Scope(827) BEGIN
 // (op 827) Lock Scope 827
 // Lock Branch
 // (op 827) generate({824})
 // (op 827) NOP(824) BEGIN
 // (op 824) generate({})
 // (op 824) end: generate({})
 // (op 827) NOP(824) END
 // (op 827) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(832) BEGIN
 // (op 832) Assign dim(600) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 600: v**UNALLOCATED**
 // (op 832) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag600: VGPR Value: UInt32 x 1: (unallocated)
 // (op 832) reg expression
 // (op 832) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 832): v12
v_and_b32 v12, -2, v1 // (op 832) 
 // (op 832) BEGIN: Tile[1]
 // (op 832) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 832): v16
v_sub_u32 v16, v1, v12 // (op 832) 
 // (op 832) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 832): v12

 // (op 832) BEGIN: Flatten
 // (op 832) {Flatten: ShiftL({Tile[1]: v16:U32}, 5:U32)U32}
 // (op 832) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 832)         value (Tile[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 832)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 832): v12
v_lshlrev_b32 v12, 5, v16 // (op 832) 
 // (op 832) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 832): v16

 // (op 832) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 832): v16
v_lshrrev_b32 v16, 1, v1 // (op 832) 
 // (op 832) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 832): v17
v_lshl_add_u32 v17, v16, 6, v12 // (op 832) 
 // (op 832) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 832): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 832): v16

 // (op 832) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 832): v12
v_mul_lo_u32 v12, v17, 6 // (op 832) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 832): v17

 // (op 832) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag600: 1 VGPR (Value: UInt32) (op 832): v16
v_lshrrev_b32 v16, 3, v12 // (op 832) 
 // Freeing : 1 VGPR (Value: UInt32) (op 832): v12

 // (op 827) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(832) END
 // (op 827) Assign VGPR 48:U32(833) BEGIN
 // (op 833) Assign dim(601) = 48:U32
 // (op 833) Assign stride expression
 // (op 827) Assign VGPR 48:U32(833) END
 // (op 827) NOP(825) BEGIN
 // (op 825) generate({})
 // (op 825) end: generate({})
 // (op 827) NOP(825) END
 // (op 827) Assign VGPR 0:U32(837) BEGIN
 // (op 837) Assign dim(603) = 0:U32
 // (op 837) Assign stride expression
 // (op 827) Assign VGPR 0:U32(837) END
 // (op 827) StoreLDSTile Value: FP6x16(393) BEGIN
 // (op 393) GEN: StoreLDSTile
 // (op 393) GEN: storeMacroTileLDS OP 393 LDS 25 MacroTile 9
 // (op 393) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 393) Generate 0:U32 into nullptr
 // (op 393) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 393) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag600: VGPR Value: UInt32 x 1: v16
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 393) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] offset:4096 // (op 393) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 1883 (current 1788)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:4112 // (op 393) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 1885 (current 1790)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 827) StoreLDSTile Value: FP6x16(393) END
 // (op 827) Deallocate{}(1307) BEGIN
 // (op 1307) Deallocate 601
 // Deleting tag 601
 // (op 1307) Deallocate 600
 // Freeing DataFlowTag600: 1 VGPR (Value: UInt32) (op 832): v16

 // Deleting tag 600
 // (op 1307) Deallocate 603
 // Deleting tag 603
 // (op 1307) Deallocate 602
 // Deleting tag 602
 // (op 827) Deallocate{}(1307) END
 // (op 827) end: generate({824})
 // (op 827) Unlock Scope 827
 // Unlock None
 // (op 494) Scope(827) END
 // (op 494) Scope(933) BEGIN
 // (op 933) Lock Scope 933
 // Lock Branch
 // (op 933) generate({930})
 // (op 933) NOP(930) BEGIN
 // (op 930) generate({})
 // (op 930) end: generate({})
 // (op 933) NOP(930) END
 // (op 933) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(938) BEGIN
 // (op 938) Assign dim(632) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 632: v**UNALLOCATED**
 // (op 938) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag632: VGPR Value: UInt32 x 1: (unallocated)
 // (op 938) reg expression
 // (op 938) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 938): v12
v_and_b32 v12, -2, v1 // (op 938) 
 // (op 938) BEGIN: Tile[1]Flatten
 // (op 938) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 938): v16
v_sub_u32 v16, v1, v12 // (op 938) 
 // (op 938) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 938): v12

 // (op 938) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 938): v12
v_and_b32 v12, 4294967294, v1 // (op 938) 
 // (op 938) BEGIN: Flatten
 // (op 938) {Flatten: Add(v12:U32, {Tile[1]Flatten: v16:U32})U32}
// Allocated DataFlowTag632: 1 VGPR (Value: UInt32) (op 938): v17
v_add_u32 v17, v12, v16 // (op 938) 
 // (op 938) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 938): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 938): v12

 // (op 933) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(938) END
 // (op 933) Assign VGPR 2:U32(939) BEGIN
 // (op 939) Assign dim(633) = 2:U32
 // (op 939) Assign stride expression
 // (op 933) Assign VGPR 2:U32(939) END
 // (op 933) NOP(931) BEGIN
 // (op 931) generate({})
 // (op 931) end: generate({})
 // (op 933) NOP(931) END
 // (op 933) Assign VGPR 1:U32(943) BEGIN
 // (op 943) Assign dim(635) = 1:U32
 // (op 943) Assign stride expression
 // (op 933) Assign VGPR 1:U32(943) END
 // (op 933) StoreLDSTile Value: E8M0(394) BEGIN
 // (op 394) GEN: StoreLDSTile
 // (op 394) GEN: storeMacroTileLDS OP 394 LDS 41 MacroTile 11
 // (op 394) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 394) Generate 1:U32 into nullptr
 // (op 394) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 394) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = DataFlowTag632: VGPR Value: UInt32 x 1: v17
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 394) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v17, v20 offset:5632 // (op 394) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1891 (current 1796)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 933) StoreLDSTile Value: E8M0(394) END
 // (op 933) Deallocate{}(1309) BEGIN
 // (op 1309) Deallocate 634
 // Deleting tag 634
 // (op 1309) Deallocate 633
 // Deleting tag 633
 // (op 1309) Deallocate 635
 // Deleting tag 635
 // (op 1309) Deallocate 632
 // Freeing DataFlowTag632: 1 VGPR (Value: UInt32) (op 938): v17

 // Deleting tag 632
 // (op 933) Deallocate{}(1309) END
 // (op 933) end: generate({930})
 // (op 933) Unlock Scope 933
 // Unlock None
 // (op 494) Scope(933) END
 // (op 494) NOP(460) BEGIN
 // (op 460) generate({})
 // (op 460) end: generate({})
 // (op 494) NOP(460) END
 // (op 494) Scope(880) BEGIN
 // (op 880) Lock Scope 880
 // Lock Branch
 // (op 880) generate({877})
 // (op 880) NOP(877) BEGIN
 // (op 877) generate({})
 // (op 877) end: generate({})
 // (op 880) NOP(877) END
 // (op 880) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(885) BEGIN
 // (op 885) Assign dim(616) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 616: v**UNALLOCATED**
 // (op 885) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag616: VGPR Value: UInt32 x 1: (unallocated)
 // (op 885) reg expression
 // (op 885) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 885): v12
v_and_b32 v12, -2, v1 // (op 885) 
 // (op 885) BEGIN: Tile[1]Flatten
 // (op 885) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 885): v16
v_sub_u32 v16, v1, v12 // (op 885) 
 // (op 885) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 885): v12

 // (op 885) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 885): v12
v_and_b32 v12, 4294967294, v1 // (op 885) 
 // (op 885) BEGIN: Flatten
 // (op 885) {Flatten: Add(v12:U32, {Tile[1]Flatten: v16:U32})U32}
// Allocated DataFlowTag616: 1 VGPR (Value: UInt32) (op 885): v17
v_add_u32 v17, v12, v16 // (op 885) 
 // (op 885) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 885): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 885): v12

 // (op 880) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(885) END
 // (op 880) Assign VGPR 2:U32(886) BEGIN
 // (op 886) Assign dim(617) = 2:U32
 // (op 886) Assign stride expression
 // (op 880) Assign VGPR 2:U32(886) END
 // (op 880) NOP(878) BEGIN
 // (op 878) generate({})
 // (op 878) end: generate({})
 // (op 880) NOP(878) END
 // (op 880) Assign VGPR 1:U32(890) BEGIN
 // (op 890) Assign dim(619) = 1:U32
 // (op 890) Assign stride expression
 // (op 880) Assign VGPR 1:U32(890) END
 // (op 880) StoreLDSTile Value: E8M0(396) BEGIN
 // (op 396) GEN: StoreLDSTile
 // (op 396) GEN: storeMacroTileLDS OP 396 LDS 33 MacroTile 10
 // (op 396) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 396) Generate 1:U32 into nullptr
 // (op 396) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 396) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = DataFlowTag616: VGPR Value: UInt32 x 1: v17
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 396) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v17, v21 offset:5696 // (op 396) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 83. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1980 (current 1885)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 880) StoreLDSTile Value: E8M0(396) END
 // (op 880) Deallocate{}(1311) BEGIN
 // (op 1311) Deallocate 617
 // Deleting tag 617
 // (op 1311) Deallocate 616
 // Freeing DataFlowTag616: 1 VGPR (Value: UInt32) (op 885): v17

 // Deleting tag 616
 // (op 1311) Deallocate 619
 // Deleting tag 619
 // (op 1311) Deallocate 618
 // Deleting tag 618
 // (op 880) Deallocate{}(1311) END
 // (op 880) end: generate({877})
 // (op 880) Unlock Scope 880
 // Unlock None
 // (op 494) Scope(880) END
 // (op 494) NOP(367) BEGIN
 // (op 367) generate({})
 // (op 367) end: generate({})
 // (op 494) NOP(367) END
 // (op 494) NOP(747) BEGIN
 // (op 747) generate({})
 // (op 747) end: generate({})
 // (op 494) NOP(747) END
 // (op 494) NOP(744) BEGIN
 // (op 744) generate({})
 // (op 744) end: generate({})
 // (op 494) NOP(744) END
 // (op 494) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(751) BEGIN
 // (op 751) Assign dim(576) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 576: s**UNALLOCATED**
 // (op 751) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag576: SGPR Value: UInt32 x 1: (unallocated)
 // (op 751) reg expression
 // (op 751) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 751): v12
v_and_b32 v12, -4, v1 // (op 751) 
 // (op 751) BEGIN: Tile[1]
 // (op 751) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 751): v16
v_sub_u32 v16, v1, v12 // (op 751) 
 // (op 751) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 751): v12

 // (op 751) BEGIN: Flatten
 // (op 751) {Flatten: ShiftL({Tile[1]: v16:U32}, 4:U32)U32}
 // (op 751) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 751)         value (Tile[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 751)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 751): v12
v_lshlrev_b32 v12, 4, v16 // (op 751) 
 // (op 751) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 751): v16

 // (op 751) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 751): v16
v_lshrrev_b32 v16, 2, v1 // (op 751) 
 // (op 751) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 751): v17
v_lshl_add_u32 v17, v16, 6, v12 // (op 751) 
 // (op 751) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 751): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 751): v16

// Allocated DataFlowTag576: 1 SGPR (Value: UInt32) (op 751): s0
s_nop 0
v_readfirstlane_b32 s0, v17 // (op 751) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 751): v17

 // (op 494) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(751) END
 // (op 494) Assign VGPR 1024:U32(752) BEGIN
 // (op 752) Assign dim(577) = 1024:U32
 // (op 752) Assign stride expression
 // (op 494) Assign VGPR 1024:U32(752) END
 // (op 494) NOP(745) BEGIN
 // (op 745) generate({})
 // (op 745) end: generate({})
 // (op 494) NOP(745) END
 // (op 494) Assign VGPR 1:U32(756) BEGIN
 // (op 756) Assign dim(579) = 1:U32
 // (op 756) Assign stride expression
 // (op 494) Assign VGPR 1:U32(756) END
 // (op 494) NOP(798) BEGIN
 // (op 798) generate({})
 // (op 798) end: generate({})
 // (op 494) NOP(798) END
 // (op 494) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(803) BEGIN
 // (op 803) Assign dim(592) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 592: v**UNALLOCATED**
 // (op 803) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag592: VGPR Value: UInt32 x 1: (unallocated)
 // (op 803) reg expression
 // (op 803) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 803): v12
v_and_b32 v12, -2, v1 // (op 803) 
 // (op 803) BEGIN: Tile[1]
 // (op 803) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 803): v16
v_sub_u32 v16, v1, v12 // (op 803) 
 // (op 803) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 803): v12

 // (op 803) BEGIN: Flatten
 // (op 803) {Flatten: ShiftL({Tile[1]: v16:U32}, 5:U32)U32}
 // (op 803) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 803)         value (Tile[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 803)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 803): v12
v_lshlrev_b32 v12, 5, v16 // (op 803) 
 // (op 803) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 803): v16

 // (op 803) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 803): v16
v_lshrrev_b32 v16, 1, v1 // (op 803) 
 // (op 803) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 803): v17
v_lshl_add_u32 v17, v16, 6, v12 // (op 803) 
 // (op 803) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 803): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 803): v16

 // (op 803) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 803): v12
v_mul_lo_u32 v12, v17, 6 // (op 803) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 803): v17

 // (op 803) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag592: 1 VGPR (Value: UInt32) (op 803): v16
v_lshrrev_b32 v16, 3, v12 // (op 803) 
 // Freeing : 1 VGPR (Value: UInt32) (op 803): v12

 // (op 494) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(803) END
 // (op 494) Assign VGPR 48:U32(804) BEGIN
 // (op 804) Assign dim(593) = 48:U32
 // (op 804) Assign stride expression
 // (op 494) Assign VGPR 48:U32(804) END
 // (op 494) NOP(799) BEGIN
 // (op 799) generate({})
 // (op 799) end: generate({})
 // (op 494) NOP(799) END
 // (op 494) Assign VGPR 0:U32(808) BEGIN
 // (op 808) Assign dim(595) = 0:U32
 // (op 808) Assign stride expression
 // (op 494) Assign VGPR 0:U32(808) END
 // (op 494) NOP(851) BEGIN
 // (op 851) generate({})
 // (op 851) end: generate({})
 // (op 494) NOP(851) END
 // (op 494) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(856) BEGIN
 // (op 856) Assign dim(608) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 608: v**UNALLOCATED**
 // (op 856) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag608: VGPR Value: UInt32 x 1: (unallocated)
 // (op 856) reg expression
 // (op 856) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 856): v12
v_and_b32 v12, -2, v1 // (op 856) 
 // (op 856) BEGIN: Tile[1]Flatten
 // (op 856) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 856): v17
v_sub_u32 v17, v1, v12 // (op 856) 
 // (op 856) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 856): v12

 // (op 856) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 856): v12
v_and_b32 v12, 4294967294, v1 // (op 856) 
 // (op 856) BEGIN: Flatten
 // (op 856) {Flatten: Add(v12:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag608: 1 VGPR (Value: UInt32) (op 856): v36
v_add_u32 v36, v12, v17 // (op 856) 
 // (op 856) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 856): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 856): v12

 // (op 494) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(856) END
 // (op 494) Assign VGPR 2:U32(857) BEGIN
 // (op 857) Assign dim(609) = 2:U32
 // (op 857) Assign stride expression
 // (op 494) Assign VGPR 2:U32(857) END
 // (op 494) NOP(852) BEGIN
 // (op 852) generate({})
 // (op 852) end: generate({})
 // (op 494) NOP(852) END
 // (op 494) Assign VGPR 1:U32(861) BEGIN
 // (op 861) Assign dim(611) = 1:U32
 // (op 861) Assign stride expression
 // (op 494) Assign VGPR 1:U32(861) END
 // (op 494) NOP(904) BEGIN
 // (op 904) generate({})
 // (op 904) end: generate({})
 // (op 494) NOP(904) END
 // (op 494) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(909) BEGIN
 // (op 909) Assign dim(624) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 624: v**UNALLOCATED**
 // (op 909) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag624: VGPR Value: UInt32 x 1: (unallocated)
 // (op 909) reg expression
 // (op 909) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 909): v12
v_and_b32 v12, -2, v1 // (op 909) 
 // (op 909) BEGIN: Tile[1]Flatten
 // (op 909) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 909): v17
v_sub_u32 v17, v1, v12 // (op 909) 
 // (op 909) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 909): v12

 // (op 909) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 909): v12
v_and_b32 v12, 4294967294, v1 // (op 909) 
 // (op 909) BEGIN: Flatten
 // (op 909) {Flatten: Add(v12:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag624: 1 VGPR (Value: UInt32) (op 909): v37
v_add_u32 v37, v12, v17 // (op 909) 
 // (op 909) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 909): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 909): v12

 // (op 494) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(909) END
 // (op 494) Assign VGPR 2:U32(910) BEGIN
 // (op 910) Assign dim(625) = 2:U32
 // (op 910) Assign stride expression
 // (op 494) Assign VGPR 2:U32(910) END
 // (op 494) NOP(905) BEGIN
 // (op 905) generate({})
 // (op 905) end: generate({})
 // (op 494) NOP(905) END
 // (op 494) Assign VGPR 1:U32(914) BEGIN
 // (op 914) Assign dim(627) = 1:U32
 // (op 914) Assign stride expression
 // (op 494) Assign VGPR 1:U32(914) END
 // (op 494) NOP(991) BEGIN
 // (op 991) generate({})
 // (op 991) end: generate({})
 // (op 494) NOP(991) END
 // (op 494) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(996) BEGIN
 // (op 996) Assign dim(650) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 650: s**UNALLOCATED**
 // (op 996) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag650: SGPR Value: UInt32 x 1: (unallocated)
 // (op 996) reg expression
 // (op 996) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 996): v12
v_and_b32 v12, -4, v1 // (op 996) 
 // (op 996) BEGIN: Tile[1]
 // (op 996) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 996): v17
v_sub_u32 v17, v1, v12 // (op 996) 
 // (op 996) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 996): v12

 // (op 996) BEGIN: Flatten
 // (op 996) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 996) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 996)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 996)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 996): v12
v_lshlrev_b32 v12, 4, v17 // (op 996) 
 // (op 996) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 996): v17

 // (op 996) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 996): v17
v_lshrrev_b32 v17, 2, v1 // (op 996) 
 // (op 996) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 996): v38
v_lshl_add_u32 v38, v17, 6, v12 // (op 996) 
 // (op 996) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 996): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 996): v17

// Allocated DataFlowTag650: 1 SGPR (Value: UInt32) (op 996): s1
s_nop 0
v_readfirstlane_b32 s1, v38 // (op 996) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 996): v38

 // (op 494) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(996) END
 // (op 494) Assign VGPR 1024:U32(997) BEGIN
 // (op 997) Assign dim(651) = 1024:U32
 // (op 997) Assign stride expression
 // (op 494) Assign VGPR 1024:U32(997) END
 // (op 494) NOP(992) BEGIN
 // (op 992) generate({})
 // (op 992) end: generate({})
 // (op 494) NOP(992) END
 // (op 494) Assign VGPR 1:U32(1001) BEGIN
 // (op 1001) Assign dim(653) = 1:U32
 // (op 1001) Assign stride expression
 // (op 494) Assign VGPR 1:U32(1001) END
 // (op 494) NOP(1030) BEGIN
 // (op 1030) generate({})
 // (op 1030) end: generate({})
 // (op 494) NOP(1030) END
 // (op 494) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1035) BEGIN
 // (op 1035) Assign dim(662) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 662: v**UNALLOCATED**
 // (op 1035) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag662: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1035) reg expression
 // (op 1035) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v12
v_and_b32 v12, -2, v1 // (op 1035) 
 // (op 1035) BEGIN: Tile[1]
 // (op 1035) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v17
v_sub_u32 v17, v1, v12 // (op 1035) 
 // (op 1035) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1035): v12

 // (op 1035) BEGIN: Flatten
 // (op 1035) {Flatten: ShiftL({Tile[1]: v17:U32}, 5:U32)U32}
 // (op 1035) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1035)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 1035)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v12
v_lshlrev_b32 v12, 5, v17 // (op 1035) 
 // (op 1035) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1035): v17

 // (op 1035) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v17
v_lshrrev_b32 v17, 1, v1 // (op 1035) 
 // (op 1035) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v38
v_lshl_add_u32 v38, v17, 6, v12 // (op 1035) 
 // (op 1035) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1035): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 1035): v17

 // (op 1035) Multiply({Flatten: v38:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v12
v_mul_lo_u32 v12, v38, 6 // (op 1035) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1035): v38

 // (op 1035) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag662: 1 VGPR (Value: UInt32) (op 1035): v17
v_lshrrev_b32 v17, 3, v12 // (op 1035) 
 // Freeing : 1 VGPR (Value: UInt32) (op 1035): v12

 // (op 494) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1035) END
 // (op 494) Assign VGPR 48:U32(1036) BEGIN
 // (op 1036) Assign dim(663) = 48:U32
 // (op 1036) Assign stride expression
 // (op 494) Assign VGPR 48:U32(1036) END
 // (op 494) NOP(1031) BEGIN
 // (op 1031) generate({})
 // (op 1031) end: generate({})
 // (op 494) NOP(1031) END
 // (op 494) Assign VGPR 0:U32(1040) BEGIN
 // (op 1040) Assign dim(665) = 0:U32
 // (op 1040) Assign stride expression
 // (op 494) Assign VGPR 0:U32(1040) END
 // (op 494) NOP(1054) BEGIN
 // (op 1054) generate({})
 // (op 1054) end: generate({})
 // (op 494) NOP(1054) END
 // (op 494) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(1059) BEGIN
 // (op 1059) Assign dim(670) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 670: v**UNALLOCATED**
 // (op 1059) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag670: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1059) reg expression
 // (op 1059) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1059): v12
v_and_b32 v12, -2, v1 // (op 1059) 
 // (op 1059) BEGIN: Tile[1]Flatten
 // (op 1059) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1059): v38
v_sub_u32 v38, v1, v12 // (op 1059) 
 // (op 1059) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1059): v12

 // (op 1059) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1059): v12
v_and_b32 v12, 4294967294, v1 // (op 1059) 
 // (op 1059) BEGIN: Flatten
 // (op 1059) {Flatten: Add(v12:U32, {Tile[1]Flatten: v38:U32})U32}
// Allocated DataFlowTag670: 1 VGPR (Value: UInt32) (op 1059): v39
v_add_u32 v39, v12, v38 // (op 1059) 
 // (op 1059) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1059): v38

 // Freeing : 1 VGPR (Value: UInt32) (op 1059): v12

 // (op 494) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(1059) END
 // (op 494) Assign VGPR 2:U32(1060) BEGIN
 // (op 1060) Assign dim(671) = 2:U32
 // (op 1060) Assign stride expression
 // (op 494) Assign VGPR 2:U32(1060) END
 // (op 494) NOP(1055) BEGIN
 // (op 1055) generate({})
 // (op 1055) end: generate({})
 // (op 494) NOP(1055) END
 // (op 494) Assign VGPR 1:U32(1064) BEGIN
 // (op 1064) Assign dim(673) = 1:U32
 // (op 1064) Assign stride expression
 // (op 494) Assign VGPR 1:U32(1064) END
 // (op 494) NOP(1078) BEGIN
 // (op 1078) generate({})
 // (op 1078) end: generate({})
 // (op 494) NOP(1078) END
 // (op 494) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(1083) BEGIN
 // (op 1083) Assign dim(678) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 678: v**UNALLOCATED**
 // (op 1083) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag678: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1083) reg expression
 // (op 1083) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1083): v12
v_and_b32 v12, -2, v1 // (op 1083) 
 // (op 1083) BEGIN: Tile[1]Flatten
 // (op 1083) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1083): v38
v_sub_u32 v38, v1, v12 // (op 1083) 
 // (op 1083) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1083): v12

 // (op 1083) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1083): v12
v_and_b32 v12, 4294967294, v1 // (op 1083) 
 // (op 1083) BEGIN: Flatten
 // (op 1083) {Flatten: Add(v12:U32, {Tile[1]Flatten: v38:U32})U32}
// Allocated DataFlowTag678: 1 VGPR (Value: UInt32) (op 1083): v40
v_add_u32 v40, v12, v38 // (op 1083) 
 // (op 1083) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1083): v38

 // Freeing : 1 VGPR (Value: UInt32) (op 1083): v12

 // (op 494) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(1083) END
 // (op 494) Assign VGPR 2:U32(1084) BEGIN
 // (op 1084) Assign dim(679) = 2:U32
 // (op 1084) Assign stride expression
 // (op 494) Assign VGPR 2:U32(1084) END
 // (op 494) NOP(1079) BEGIN
 // (op 1079) generate({})
 // (op 1079) end: generate({})
 // (op 494) NOP(1079) END
 // (op 494) Assign VGPR 1:U32(1088) BEGIN
 // (op 1088) Assign dim(681) = 1:U32
 // (op 1088) Assign stride expression
 // (op 494) Assign VGPR 1:U32(1088) END
 // (op 494) ForLoopOp KLoop: LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(60) BEGIN
 // (op 60) Initialize For Loop
 // (op 60) generate({61})
 // (op 60) Assign SGPR 0:U32(61) BEGIN
 // (op 61) Assign dim(397) = 0:U32
 // tag 397: s**UNALLOCATED**
 // (op 61) Generate 0:U32 into DataFlowTag397: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag397: 1 SGPR (Value: UInt32) (op 61): s14
s_mov_b32 s14, 0 // (op 61) call()
 // (op 60) Assign SGPR 0:U32(61) END
 // (op 60) end: generate({61})
 // (op 60) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 60) 
 // (op 60) Generate LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 60) LessThan({DataFlowTag397: s14:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 60) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 60) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 60): s18, s19
s_ashr_i64 s[18:19], s[28:29], 6 // (op 60) 
// Allocated : 1 SGPR (Value: UInt32) (op 60): s15
s_mov_b32 s15, s18 // (op 60) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 60): s18, s19

// Allocated : 1 SGPR (Value: UInt32) (op 60): s18
s_and_b32 s18, s15, 4294967294 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s15

s_cmp_lt_u32 s14, s18 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 60) 
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60:
 // (op 60) 
 // (op 60) generate({62, 292, 1132})
 // (op 60) Assign SGPR Add(DataFlowTag(397)U32, 2:U32)U32(62) BEGIN
 // (op 62) Assign dim(397) = Add(DataFlowTag(397)U32, 2:U32)U32
 // (op 62) Generate Add(DataFlowTag(397)U32, 2:U32)U32 into DataFlowTag397: SGPR Value: UInt32 x 1: s14
 // (op 62) reg expression
 // (op 62) Add({DataFlowTag397: s14:U32}, 2:U32)U32
 // (op 60) SetCoordinate(292) BEGIN
 // (op 292) generate({})
 // (op 292) end: generate({})
 // (op 292) generate({10})
 // (op 292) LoadTiled Value: FP6x16(10) BEGIN
 // (op 10) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 10) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 10) Generate 0:U64 into nullptr
 // (op 10) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 528: v**UNALLOCATED**
 // (op 60) Barrier(1132) BEGIN
s_barrier  // (op 1132) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 60) Barrier(1132) END
s_add_u32 s14, s14, 2 // (op 62) 
 // (op 60) Assign SGPR Add(DataFlowTag(397)U32, 2:U32)U32(62) END
 // (op 60) SetCoordinate(70) BEGIN
 // (op 70) generate({})
 // (op 70) end: generate({})
 // (op 70) generate({72})
 // (op 70) LoadLDSTile{Value: BF8x4}(72) BEGIN
 // (op 72) GEN: loadMacroTileWAVELDS OP 72 LDS 17 WaveTile 105
 // (op 72) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 72) Generate 1:U32 into nullptr
 // tag 15: v**UNALLOCATED**
 // (op 72) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 72) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 72) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated Offset528: 1 VGPR (Value: UInt32) (op 10): v12
v_mov_b32 v12, v15 // (op 10) 
 // (op 10) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = Offset528: VGPR Value: UInt32 x 1: v12
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 96
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 10) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 8 VGPRs (Value: BF8x4) (op 72): v42, v43, v44, v45, v46, v47, v48, v49
ds_read_b128 v[42:45], v3 // (op 72) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2035 (current 1940)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
ds_read_b128 v[46:49], v3 offset:32 // (op 72) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2036 (current 1941)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 70) LoadLDSTile{Value: BF8x4}(72) END
 // (op 70) end: generate({72})
 // (op 60) SetCoordinate(70) END
buffer_load_dwordx4 v[24:27], v12, s[44:47], 0 offen offset:96 // (op 10) Load value
// VMEM: Expected stall of 313. CBNW: 3, Inc: 3
// VMEM: Expected complete at 2735 (current 2255)
 // (op 60) Barrier(1145) BEGIN
buffer_load_dwordx2 v[28:29], v12, s[44:47], 0 offen offset:112 // (op 10) Load value
// VMEM: Expected stall of 3. CBNW: 4, Inc: 3
// VMEM: Expected complete at 2739 (current 2259)
 // (op 292) LoadTiled Value: FP6x16(10) END
 // (op 292) end: generate({10})
 // (op 60) SetCoordinate(292) END
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1145) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 60) Barrier(1145) END
 // (op 60) SetCoordinate(302) BEGIN
 // (op 302) generate({})
 // (op 302) end: generate({})
 // (op 302) generate({25})
 // (op 302) LoadTiled Value: E8M0(25) BEGIN
 // (op 25) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 25) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 25) Generate 1:U64 into nullptr
 // (op 25) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 566: v**UNALLOCATED**
 // (op 60) SetCoordinate(75) BEGIN
 // (op 75) generate({})
 // (op 75) end: generate({})
 // (op 75) generate({77})
 // (op 75) LoadLDSTile{Value: FP6x16}(77) BEGIN
 // (op 77) GEN: loadMacroTileWAVELDS OP 77 LDS 25 WaveTile 171
 // (op 77) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 77) Generate 0:U32 into nullptr
 // tag 23: v**UNALLOCATED**
 // (op 77) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 77) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 77) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated Offset566: 1 VGPR (Value: UInt32) (op 25): v38
v_mov_b32 v38, v23 // (op 25) 
 // (op 25) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = Offset566: VGPR Value: UInt32 x 1: v38
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 25) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v20, v38, s[52:55], 0 offen offset:4 // (op 25) Load value
// VMEM: Expected complete at 2743 (current 2263)
 // (op 302) LoadTiled Value: E8M0(25) END
 // (op 302) end: generate({25})
 // (op 60) SetCoordinate(302) END
// Allocated : 6 VGPRs (Value: FP6x16) (op 77): v50, v51, v52, v53, v54, v55
ds_read_b128 v[50:53], v4 offset:4096 // (op 77) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 2359 (current 2264)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 60) Barrier(1135) BEGIN
ds_read_b64 v[54:55], v4 offset:4112 // (op 77) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 2360 (current 2265)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 75) LoadLDSTile{Value: FP6x16}(77) END
 // (op 75) end: generate({77})
 // (op 60) SetCoordinate(75) END
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1135) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 60) Barrier(1135) END
 // (op 60) Barrier(1159) BEGIN
 // (op 60) SetCoordinate(287) BEGIN
 // (op 287) generate({})
 // (op 287) end: generate({})
 // (op 287) generate({442})
 // (op 287) LoadTileDirect2LDS Value: BF8x4(442) BEGIN
 // (op 442) GEN: LoadTileDirect2LDS
 // (op 442) GEN: loadMacroTileDirect2LDS OP 442 LDS 17 MacroTile 8
 // (op 442) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 442) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride510: VGPR Value: UInt64 x 1: (unallocated)
 // (op 442) reg expression
 // (op 442) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 442) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 442)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 442)         shiftAmount (Literal Value: UInt32 x 0: 4)
s_barrier  // (op 1159) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 60) Barrier(1159) END
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 442): s18, s19
s_lshl_b64 s[18:19], s[2:3], 4 // (op 442) 
 // (op 60) SetCoordinate(80) BEGIN
 // (op 80) generate({})
 // (op 80) end: generate({})
 // (op 80) generate({82})
 // (op 80) LoadLDSTile{Value: E8M0}(82) BEGIN
 // (op 82) GEN: loadMacroTileWAVELDS OP 82 LDS 33 WaveTile 237
 // (op 82) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 82) Generate 1:U32 into nullptr
 // tag 31: v**UNALLOCATED**
 // (op 82) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 82) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag620: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 82) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 82): v41
ds_read_u8 v41, v0 offset:5696 // (op 82) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 2365 (current 2270)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 80) LoadLDSTile{Value: E8M0}(82) END
 // (op 80) end: generate({82})
 // (op 60) SetCoordinate(80) END
// Allocated Stride510: 2 VGPRs (Value: UInt64) (op 442): v56, v57
v_mov_b64 v[56:57], s[18:19] // (op 442) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 442): s18, s19

 // (op 442) Generate 1:U64 into nullptr
 // (op 442) Generate 1024:U32 into nullptr
 // (op 442) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride510: VGPR Value: UInt64 x 1: v[56:57]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 509: v**UNALLOCATED**
 // (op 60) Barrier(1173) BEGIN
s_barrier  // (op 1173) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 60) Barrier(1173) END
// Allocated Offset509: 2 VGPRs (Value: UInt64) (op 442): v58, v59
v_mov_b64 v[58:59], v[18:19] // (op 442) 
 // (op 60) SetCoordinate(85) BEGIN
 // (op 85) generate({})
 // (op 85) end: generate({})
 // (op 85) generate({87})
 // (op 85) LoadLDSTile{Value: E8M0}(87) BEGIN
 // (op 87) GEN: loadMacroTileWAVELDS OP 87 LDS 41 WaveTile 297
 // (op 87) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 87) Generate 1:U32 into nullptr
 // tag 39: v**UNALLOCATED**
 // (op 87) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 87) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag636: VGPR Value: UInt32 x 1: v6
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 87) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 2 VGPRs (Value: UInt64) (op 442): v60, v61
v_mov_b64 v[60:61], v[58:59] // (op 442) 
 // (op 442) Generate Add(0:U32, {DataFlowTag576Flatten: s0:U32})U32 into nullptr
 // (op 442)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride510: v[56:57]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 442) Generate {DataFlowTag576Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 442) reg expression
 // (op 442) Generate 128:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 128 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 442): s15
s_mov_b32 s15, 128 // (op 442) call()
// 128 is not a supported value!
// Allocated : 1 VGPR (Value: E8M0) (op 87): v62
ds_read_u8 v62, v6 offset:5632 // (op 87) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected stall of 83. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2455 (current 2360)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 85) LoadLDSTile{Value: E8M0}(87) END
 // (op 85) end: generate({87})
 // (op 60) SetCoordinate(85) END
buffer_load_dwordx4 v60, s[40:43], s15 offen offset: 0 lds // (op 442) Load value direct to lds
// VMEM: Expected stall of 375. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3216 (current 2736)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 442) Generate Add(v[60:61]:U64, v56:R)U64 into VGPR Value: UInt64 x 1: v[60:61]
 // (op 442) reg expression
 // (op 442) reg expression
 // (op 442) Add(v[60:61]:U64, v56:R)U64
 // (op 442) Add: dest (VGPR Value: UInt64 x 1: v[60:61]) = 
 // (op 442)      lhs (VGPR Value: UInt64 x 1: v[60:61]) 
 // (op 442)      rhs (VGPR Value: Raw32 x 1: v56)
 // (op 60) Multiply(90) BEGIN
 // (op 90) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile16xFP6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
// Allocated : 2 SGPRs (Value: Bool64) (op 442): s18, s19
v_add_co_u32 v60, s[18:19], v60, v56 // (op 442) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 442) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 442): s18, s19

 // (op 442) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 442) reg expression
 // (op 442) Add(m0:U32, 1024:U32)U32
 // (op 442) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 442) 
 // (op 442) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v60, s[40:43], s15 offen offset: 0 lds // (op 442) Load value direct to lds
// VMEM: Expected complete at 3220 (current 2740)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 442): s15

 // Freeing Stride510: 2 VGPRs (Value: UInt64) (op 442): v56, v57

 // Freeing : 2 VGPRs (Value: UInt64) (op 442): v60, v61

 // (op 287) LoadTileDirect2LDS Value: BF8x4(442) END
 // (op 287) Deallocate{}(1313) BEGIN
 // (op 1313) Deallocate 578
 // Deleting tag 578
 // (op 287) Deallocate{}(1313) END
 // (op 287) end: generate({442})
 // (op 60) SetCoordinate(287) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[42:49], v[50:55], a[0:15], v41, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 90) 
 // (op 60) Multiply(90) END
 // (op 60) SetCoordinate(297) BEGIN
 // (op 297) generate({})
 // (op 297) end: generate({})
 // (op 297) generate({17})
 // (op 297) LoadTiled Value: E8M0(17) BEGIN
 // (op 17) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 17) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 17) Generate 1:U64 into nullptr
 // (op 17) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 547: v**UNALLOCATED**
 // (op 60) Deallocate{}(1225) BEGIN
 // (op 1225) Deallocate 23
 // Freeing : 6 VGPRs (Value: FP6x16) (op 77): v50, v51, v52, v53, v54, v55

 // Deleting tag 23
 // (op 1225) Deallocate 39
 // Freeing : 1 VGPR (Value: E8M0) (op 87): v62

 // Deleting tag 39
 // (op 1225) Deallocate 15
 // Freeing : 8 VGPRs (Value: BF8x4) (op 72): v42, v43, v44, v45, v46, v47, v48, v49

 // Deleting tag 15
 // (op 1225) Deallocate 31
 // Freeing : 1 VGPR (Value: E8M0) (op 82): v41

 // Deleting tag 31
 // (op 60) Deallocate{}(1225) END
// Allocated Offset547: 1 VGPR (Value: UInt32) (op 17): v41
v_mov_b32 v41, v13 // (op 17) 
 // (op 17) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = Offset547: VGPR Value: UInt32 x 1: v41
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 17) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v21, v41, s[48:51], 0 offen offset:4 // (op 17) Load value
// VMEM: Expected complete at 3224 (current 2744)
 // (op 297) LoadTiled Value: E8M0(17) END
 // (op 297) end: generate({17})
 // (op 60) SetCoordinate(297) END
 // (op 60) Barrier(1196) BEGIN
s_barrier  // (op 1196) 
 // (op 60) Barrier(1196) END
 // (op 60) StoreLDSTile Value: FP6x16(257) BEGIN
 // (op 257) GEN: StoreLDSTile
 // (op 257) GEN: storeMacroTileLDS OP 257 LDS 476 MacroTile 474
 // (op 257) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 257) Generate 0:U32 into nullptr
 // (op 257) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[30:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5760
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 257) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[30:35]
 // 	info.rowOffsetReg = DataFlowTag662: VGPR Value: UInt32 x 1: v17
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5760
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 257) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(11)
ds_write_b128 v17, v[30:33] offset:5760 // (op 257) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 2842 (current 2747)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 5760, Size: 1536)
s_waitcnt vmcnt(10)
ds_write_b64 v17, v[34:35] offset:5776 // (op 257) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 2844 (current 2749)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 5760, Size: 1536)
 // (op 60) StoreLDSTile Value: FP6x16(257) END
 // (op 60) Deallocate{}(1281) BEGIN
 // (op 1281) Deallocate 664
 // Deleting tag 664
 // (op 60) Deallocate{}(1281) END
 // (op 60) Barrier(1208) BEGIN
s_barrier  // (op 1208) 
 // (op 60) Barrier(1208) END
 // (op 60) StoreLDSTile Value: E8M0(259) BEGIN
 // (op 259) GEN: StoreLDSTile
 // (op 259) GEN: storeMacroTileLDS OP 259 LDS 484 MacroTile 482
 // (op 259) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 259) Generate 1:U32 into nullptr
 // (op 259) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7296
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 259) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = DataFlowTag678: VGPR Value: UInt32 x 1: v40
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7296
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 259) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v40, v14 offset:7296 // (op 259) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2847 (current 2752)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 7296, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(259) END
 // (op 60) Deallocate{}(1285) BEGIN
 // (op 1285) Deallocate 680
 // Deleting tag 680
 // (op 60) Deallocate{}(1285) END
 // (op 60) NOP(455) BEGIN
 // (op 455) generate({})
 // (op 455) end: generate({})
 // (op 60) NOP(455) END
 // (op 60) Barrier(1202) BEGIN
s_barrier  // (op 1202) 
 // (op 60) Barrier(1202) END
 // (op 60) StoreLDSTile Value: E8M0(258) BEGIN
 // (op 258) GEN: StoreLDSTile
 // (op 258) GEN: storeMacroTileLDS OP 258 LDS 480 MacroTile 478
 // (op 258) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 258) Generate 1:U32 into nullptr
 // (op 258) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 258) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = DataFlowTag670: VGPR Value: UInt32 x 1: v39
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 258) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v39, v22 offset:7360 // (op 258) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2939 (current 2844)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 7360, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(258) END
 // (op 60) Deallocate{}(1283) BEGIN
 // (op 1283) Deallocate 672
 // Deleting tag 672
 // (op 60) Deallocate{}(1283) END
 // (op 60) NOP(410) BEGIN
 // (op 410) generate({})
 // (op 410) end: generate({})
 // (op 60) NOP(410) END
 // (op 60) SetCoordinate(313) BEGIN
 // (op 313) generate({})
 // (op 313) end: generate({})
 // (op 313) generate({260})
 // (op 313) LoadTiled Value: FP6x16(260) BEGIN
 // (op 260) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 260) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 260) Generate 0:U64 into nullptr
 // (op 260) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[30:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 260) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[30:35]
 // 	info.rowOffsetReg = Offset528: VGPR Value: UInt32 x 1: v12
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 144
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 260) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Barrier(1187) BEGIN
s_barrier  // (op 1187) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 60) Barrier(1187) END
buffer_load_dwordx4 v[30:33], v12, s[44:47], 0 offen offset:144 // (op 260) Load value
// VMEM: Expected stall of 371. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3697 (current 3217)
 // (op 60) SetCoordinate(248) BEGIN
 // (op 248) generate({})
 // (op 248) end: generate({})
 // (op 248) generate({249})
 // (op 248) LoadLDSTile{Value: BF8x4}(249) BEGIN
 // (op 249) GEN: loadMacroTileWAVELDS OP 249 LDS 472 WaveTile 105
 // (op 249) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 249) Generate 1:U32 into nullptr
 // tag 486: v**UNALLOCATED**
 // (op 249) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2048
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 249) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag658: VGPR Value: UInt32 x 1: v7
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2048
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 249) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated : 8 VGPRs (Value: BF8x4) (op 249): v42, v43, v44, v45, v46, v47, v48, v49
ds_read_b128 v[42:45], v7 offset:2048 // (op 249) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 3313 (current 3218)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
ds_read_b128 v[46:49], v7 offset:2080 // (op 249) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 3314 (current 3219)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 248) LoadLDSTile{Value: BF8x4}(249) END
 // (op 248) Deallocate{}(1259) BEGIN
 // (op 1259) Deallocate 660
 // Deleting tag 660
 // (op 248) Deallocate{}(1259) END
 // (op 248) end: generate({249})
 // (op 60) SetCoordinate(248) END
buffer_load_dwordx2 v[34:35], v12, s[44:47], 0 offen offset:160 // (op 260) Load value
// VMEM: Expected stall of 1. CBNW: 4, Inc: 3
// VMEM: Expected complete at 3701 (current 3221)
 // (op 313) LoadTiled Value: FP6x16(260) END
 // (op 313) end: generate({260})
 // (op 60) SetCoordinate(313) END
 // (op 60) Barrier(1193) BEGIN
 // (op 60) Deallocate{}(1289) BEGIN
 // (op 1289) Deallocate 528
 // Freeing Offset528: 1 VGPR (Value: UInt32) (op 10): v12

 // Deleting tag 528
 // (op 1289) Deallocate 530
 // Deleting tag 530
 // (op 60) Deallocate{}(1289) END
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1193) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 5760, Size: 1536)
 // (op 60) Barrier(1193) END
 // (op 60) SetCoordinate(323) BEGIN
 // (op 323) generate({})
 // (op 323) end: generate({})
 // (op 323) generate({262})
 // (op 323) LoadTiled Value: E8M0(262) BEGIN
 // (op 262) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 262) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 262) Generate 1:U64 into nullptr
 // (op 262) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 262) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = Offset566: VGPR Value: UInt32 x 1: v38
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 6
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 262) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Assign VGPR Add(DataFlowTag(526)U32, ShiftL(Convert(DataFlowTag(527)U64)U32, 1:U32)U32)U32(577) BEGIN
 // (op 577) Assign dim(526) = Add(DataFlowTag(526)U32, ShiftL(Convert(DataFlowTag(527)U64)U32, 1:U32)U32)U32
 // (op 577) Generate Add(DataFlowTag(526)U32, ShiftL(Convert(DataFlowTag(527)U64)U32, 1:U32)U32)U32 into DataFlowTag526: VGPR Value: UInt32 x 1: v15
 // (op 577) reg expression
 // (op 577) Add({DataFlowTag526: v15:U32}, 96:U32)U32
 // (op 60) SetCoordinate(250) BEGIN
 // (op 250) generate({})
 // (op 250) end: generate({})
 // (op 250) generate({251})
 // (op 250) LoadLDSTile{Value: FP6x16}(251) BEGIN
 // (op 251) GEN: loadMacroTileWAVELDS OP 251 LDS 476 WaveTile 171
 // (op 251) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 251) Generate 0:U32 into nullptr
 // tag 488: v**UNALLOCATED**
 // (op 251) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5760
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 251) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag666: VGPR Value: UInt32 x 1: v8
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5760
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 251) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

v_add_u32 v15, 96, v15 // (op 577) 
 // (op 60) Assign VGPR Add(DataFlowTag(526)U32, ShiftL(Convert(DataFlowTag(527)U64)U32, 1:U32)U32)U32(577) END
buffer_load_ubyte v14, v38, s[52:55], 0 offen offset:6 // (op 262) Load value
// VMEM: Expected complete at 3705 (current 3225)
 // (op 323) LoadTiled Value: E8M0(262) END
 // (op 323) end: generate({262})
 // (op 60) SetCoordinate(323) END
// Allocated : 6 VGPRs (Value: FP6x16) (op 251): v50, v51, v52, v53, v54, v55
ds_read_b128 v[50:53], v8 offset:5760 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 3321 (current 3226)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 5760, Size: 1536)
 // (op 60) Deallocate{}(1293) BEGIN
 // (op 1293) Deallocate 566
 // Freeing Offset566: 1 VGPR (Value: UInt32) (op 25): v38

 // Deleting tag 566
 // (op 1293) Deallocate 568
 // Deleting tag 568
 // (op 60) Deallocate{}(1293) END
ds_read_b64 v[54:55], v8 offset:5776 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected stall of 87. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 3409 (current 3314)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 5760, Size: 1536)
 // (op 250) LoadLDSTile{Value: FP6x16}(251) END
 // (op 250) Deallocate{}(1265) BEGIN
 // (op 1265) Deallocate 668
 // Deleting tag 668
 // (op 250) Deallocate{}(1265) END
 // (op 250) end: generate({251})
 // (op 60) SetCoordinate(250) END
 // (op 60) Assign VGPR Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32(705) BEGIN
 // (op 705) Assign dim(564) = Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32
 // (op 705) Generate Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32 into DataFlowTag564: VGPR Value: UInt32 x 1: v23
 // (op 705) reg expression
 // (op 705) Add({DataFlowTag564: v23:U32}, 4:U32)U32
 // (op 60) Barrier(1190) BEGIN
 // (op 60) Deallocate{}(1261) BEGIN
 // (op 1261) Deallocate 476
 // Deleting tag 476
 // (op 60) Deallocate{}(1261) END
v_add_u32 v23, 4, v23 // (op 705) 
 // (op 60) Assign VGPR Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32(705) END
 // (op 60) Barrier(1199) BEGIN
 // (op 60) Barrier(1263) BEGIN
s_barrier  // (op 1263) 
 // (op 60) Barrier(1263) END
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1199) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7360, Size: 64)
 // (op 60) Barrier(1199) END
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1190) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 60) Barrier(1190) END
 // (op 60) SetCoordinate(252) BEGIN
 // (op 252) generate({})
 // (op 252) end: generate({})
 // (op 252) generate({253})
 // (op 252) LoadLDSTile{Value: E8M0}(253) BEGIN
 // (op 253) GEN: loadMacroTileWAVELDS OP 253 LDS 480 WaveTile 237
 // (op 253) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 253) Generate 1:U32 into nullptr
 // tag 490: v**UNALLOCATED**
 // (op 253) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 253) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag674: VGPR Value: UInt32 x 1: v5
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 253) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) SetCoordinate(308) BEGIN
 // (op 308) generate({})
 // (op 308) end: generate({})
 // (op 308) generate({453})
 // (op 308) LoadTileDirect2LDS Value: BF8x4(453) BEGIN
 // (op 453) GEN: LoadTileDirect2LDS
 // (op 453) GEN: loadMacroTileDirect2LDS OP 453 LDS 472 MacroTile 470
 // (op 453) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 453) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride510: VGPR Value: UInt64 x 1: (unallocated)
 // (op 453) reg expression
 // (op 453) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 453) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 453)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 453)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: E8M0) (op 253): v12
ds_read_u8 v12, v5 offset:7360 // (op 253) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3416 (current 3321)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7360, Size: 64)
 // (op 252) LoadLDSTile{Value: E8M0}(253) END
 // (op 252) Deallocate{}(1271) BEGIN
 // (op 1271) Deallocate 676
 // Deleting tag 676
 // (op 252) Deallocate{}(1271) END
 // (op 252) end: generate({253})
 // (op 60) SetCoordinate(252) END
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 453): s18, s19
s_lshl_b64 s[18:19], s[2:3], 4 // (op 453) 
 // (op 60) Deallocate{}(1267) BEGIN
 // (op 1267) Deallocate 480
 // Deleting tag 480
 // (op 60) Deallocate{}(1267) END
// Allocated Stride510: 2 VGPRs (Value: UInt64) (op 453): v56, v57
v_mov_b64 v[56:57], s[18:19] // (op 453) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 453): s18, s19

 // (op 453) Generate 1:U64 into nullptr
 // (op 453) Generate 1024:U32 into nullptr
 // (op 453) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride510: VGPR Value: UInt64 x 1: v[56:57]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // (op 60) Barrier(1205) BEGIN
 // (op 60) Barrier(1269) BEGIN
s_barrier  // (op 1205) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7296, Size: 64)
 // (op 60) Barrier(1205) END
s_barrier  // (op 1269) 
 // (op 60) Barrier(1269) END
 // (op 60) SetCoordinate(254) BEGIN
 // (op 254) generate({})
 // (op 254) end: generate({})
 // (op 254) generate({255})
 // (op 254) LoadLDSTile{Value: E8M0}(255) BEGIN
 // (op 255) GEN: loadMacroTileWAVELDS OP 255 LDS 484 WaveTile 297
 // (op 255) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 255) Generate 1:U32 into nullptr
 // tag 492: v**UNALLOCATED**
 // (op 255) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7296
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 255) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag682: VGPR Value: UInt32 x 1: v10
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7296
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 255) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 255): v38
ds_read_u8 v38, v10 offset:7296 // (op 255) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3421 (current 3326)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7296, Size: 64)
 // (op 254) LoadLDSTile{Value: E8M0}(255) END
 // (op 254) Deallocate{}(1277) BEGIN
 // (op 1277) Deallocate 684
 // Deleting tag 684
 // (op 254) Deallocate{}(1277) END
 // (op 254) end: generate({255})
 // (op 60) SetCoordinate(254) END
// Allocated : 2 VGPRs (Value: UInt64) (op 453): v60, v61
v_mov_b64 v[60:61], v[58:59] // (op 453) 
 // (op 453) Generate Add(2048:U32, {DataFlowTag650Flatten: s1:U32})U32 into nullptr
 // (op 453) reg expression
 // (op 453) Add(2048:U32, {DataFlowTag650Flatten: s1:U32})U32
 // (op 60) Deallocate{}(1273) BEGIN
 // (op 1273) Deallocate 484
 // Deleting tag 484
 // (op 60) Deallocate{}(1273) END
// Allocated : 1 SGPR (Value: UInt32) (op 453): s15
s_add_u32 s15, 2048, s1 // (op 453) 
 // (op 453)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride510: v[56:57]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 453) Generate s15:U32 into M0 Value: UInt32 x 1: m0
 // (op 60) Multiply(256) BEGIN
 // (op 256) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile16xFP6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 60) Barrier(1275) BEGIN
s_barrier  // (op 1275) 
 // (op 60) Barrier(1275) END
s_mov_b32 m0, s15 // (op 453) reg expression
 // (op 453) Generate 192:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 192 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 453): s18
s_mov_b32 s18, 192 // (op 453) call()
// 192 is not a supported value!
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[42:49], v[50:55], a[0:15], v12, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 256) 
 // (op 60) Multiply(256) END
buffer_load_dwordx4 v60, s[40:43], s18 offen offset: 0 lds // (op 453) Load value direct to lds
// VMEM: Expected stall of 364. CBNW: 6, Inc: 3
// VMEM: Expected complete at 4178 (current 3698)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 453) Generate Add(v[60:61]:U64, v56:R)U64 into VGPR Value: UInt64 x 1: v[60:61]
 // (op 453) reg expression
 // (op 453) reg expression
 // (op 453) Add(v[60:61]:U64, v56:R)U64
 // (op 453) Add: dest (VGPR Value: UInt64 x 1: v[60:61]) = 
 // (op 453)      lhs (VGPR Value: UInt64 x 1: v[60:61]) 
 // (op 453)      rhs (VGPR Value: Raw32 x 1: v56)
 // (op 60) Deallocate{}(1279) BEGIN
 // (op 1279) Deallocate 490
 // Freeing : 1 VGPR (Value: E8M0) (op 253): v12

 // Deleting tag 490
 // (op 1279) Deallocate 492
 // Freeing : 1 VGPR (Value: E8M0) (op 255): v38

 // Deleting tag 492
 // (op 1279) Deallocate 488
 // Freeing : 6 VGPRs (Value: FP6x16) (op 251): v50, v51, v52, v53, v54, v55

 // Deleting tag 488
 // (op 1279) Deallocate 486
 // Freeing : 8 VGPRs (Value: BF8x4) (op 249): v42, v43, v44, v45, v46, v47, v48, v49

 // Deleting tag 486
 // (op 60) Deallocate{}(1279) END
// Allocated : 2 SGPRs (Value: Bool64) (op 453): s38, s39
v_add_co_u32 v60, s[38:39], v60, v56 // (op 453) least significant half
v_addc_co_u32 v61, s[38:39], 0, v61, s[38:39] // (op 453) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 453): s38, s39

 // (op 453) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 453) reg expression
 // (op 453) Add(m0:U32, 1024:U32)U32
 // (op 453) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 453) 
 // (op 453) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v60, s[40:43], s18 offen offset: 0 lds // (op 453) Load value direct to lds
// VMEM: Expected complete at 4182 (current 3702)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 453): s18

 // Freeing Stride510: 2 VGPRs (Value: UInt64) (op 453): v56, v57

 // Freeing : 2 VGPRs (Value: UInt64) (op 453): v60, v61

 // Freeing : 1 SGPR (Value: UInt32) (op 453): s15

 // (op 308) LoadTileDirect2LDS Value: BF8x4(453) END
 // (op 308) Deallocate{}(1317) BEGIN
 // (op 1317) Deallocate 652
 // Deleting tag 652
 // (op 308) Deallocate{}(1317) END
 // (op 308) end: generate({453})
 // (op 60) SetCoordinate(308) END
 // (op 60) Deallocate{}(1287) BEGIN
 // (op 1287) Deallocate 509
 // Freeing Offset509: 2 VGPRs (Value: UInt64) (op 442): v58, v59

 // Deleting tag 509
 // (op 1287) Deallocate 511
 // Deleting tag 511
 // (op 60) Deallocate{}(1287) END
 // (op 60) SetCoordinate(318) BEGIN
 // (op 318) generate({})
 // (op 318) end: generate({})
 // (op 318) generate({261})
 // (op 318) LoadTiled Value: E8M0(261) BEGIN
 // (op 261) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 261) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 261) Generate 1:U64 into nullptr
 // (op 261) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 261) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = Offset547: VGPR Value: UInt32 x 1: v41
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 6
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 261) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Assign VGPR Add(DataFlowTag(507)U64, ShiftL(DataFlowTag(508)U64, 1:U32)U64)U64(513) BEGIN
 // (op 513) Assign dim(507) = Add(DataFlowTag(507)U64, ShiftL(DataFlowTag(508)U64, 1:U32)U64)U64
 // (op 513) Generate Add(DataFlowTag(507)U64, ShiftL(DataFlowTag(508)U64, 1:U32)U64)U64 into DataFlowTag507: VGPR Value: UInt64 x 1: v[18:19]
 // (op 513) reg expression
 // (op 513) Add({DataFlowTag507: v[18:19]:U64}, 128:U64)U64
 // (op 513) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 513)      lhs (DataFlowTag507: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 513)      rhs (Literal Value: UInt64 x 0: 128)
v_add_co_u32 v18, vcc, 128, v18 // (op 513) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 513) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 60) Assign VGPR Add(DataFlowTag(507)U64, ShiftL(DataFlowTag(508)U64, 1:U32)U64)U64(513) END
buffer_load_ubyte v22, v41, s[48:51], 0 offen offset:6 // (op 261) Load value
// VMEM: Expected stall of 1. CBNW: 8, Inc: 3
// VMEM: Expected complete at 4186 (current 3706)
 // (op 318) LoadTiled Value: E8M0(261) END
 // (op 318) end: generate({261})
 // (op 60) SetCoordinate(318) END
 // (op 60) Deallocate{}(1291) BEGIN
 // (op 1291) Deallocate 549
 // Deleting tag 549
 // (op 1291) Deallocate 547
 // Freeing Offset547: 1 VGPR (Value: UInt32) (op 17): v41

 // Deleting tag 547
 // (op 60) Deallocate{}(1291) END
 // (op 60) Assign VGPR Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32(641) BEGIN
 // (op 641) Assign dim(545) = Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32
 // (op 641) Generate Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32 into DataFlowTag545: VGPR Value: UInt32 x 1: v13
 // (op 641) reg expression
 // (op 641) Add({DataFlowTag545: v13:U32}, 4:U32)U32
 // (op 60) Barrier(1148) BEGIN
s_barrier  // (op 1148) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 60) Barrier(1148) END
v_add_u32 v13, 4, v13 // (op 641) 
 // (op 60) Assign VGPR Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32(641) END
 // (op 60) StoreLDSTile Value: FP6x16(13) BEGIN
 // (op 13) GEN: StoreLDSTile
 // (op 13) GEN: storeMacroTileLDS OP 13 LDS 25 MacroTile 9
 // (op 13) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 13) Generate 0:U32 into nullptr
 // (op 13) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 13) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag592: VGPR Value: UInt32 x 1: v16
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 13) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] offset:4096 // (op 13) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 3895 (current 3800)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:4112 // (op 13) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 3897 (current 3802)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 60) StoreLDSTile Value: FP6x16(13) END
 // (op 60) Deallocate{}(1211) BEGIN
 // (op 1211) Deallocate 594
 // Deleting tag 594
 // (op 60) Deallocate{}(1211) END
 // (op 60) Barrier(1176) BEGIN
s_barrier  // (op 1176) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 60) Barrier(1176) END
 // (op 60) StoreLDSTile Value: E8M0(30) BEGIN
 // (op 30) GEN: StoreLDSTile
 // (op 30) GEN: storeMacroTileLDS OP 30 LDS 41 MacroTile 11
 // (op 30) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 30) Generate 1:U32 into nullptr
 // (op 30) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 30) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = DataFlowTag624: VGPR Value: UInt32 x 1: v37
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 30) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v37, v20 offset:5632 // (op 30) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3900 (current 3805)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(30) END
 // (op 60) Deallocate{}(1215) BEGIN
 // (op 1215) Deallocate 626
 // Deleting tag 626
 // (op 60) Deallocate{}(1215) END
 // (op 60) NOP(444) BEGIN
 // (op 444) generate({})
 // (op 444) end: generate({})
 // (op 60) NOP(444) END
 // (op 60) Barrier(1162) BEGIN
s_barrier  // (op 1162) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 60) Barrier(1162) END
 // (op 60) StoreLDSTile Value: E8M0(21) BEGIN
 // (op 21) GEN: StoreLDSTile
 // (op 21) GEN: storeMacroTileLDS OP 21 LDS 33 MacroTile 10
 // (op 21) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 21) Generate 1:U32 into nullptr
 // (op 21) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 21) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = DataFlowTag608: VGPR Value: UInt32 x 1: v36
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 21) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v36, v21 offset:5696 // (op 21) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 3992 (current 3897)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(21) END
 // (op 60) Deallocate{}(1213) BEGIN
 // (op 1213) Deallocate 610
 // Deleting tag 610
 // (op 60) Deallocate{}(1213) END
 // (op 60) NOP(411) BEGIN
 // (op 411) generate({})
 // (op 411) end: generate({})
 // (op 60) NOP(411) END
 // (op 60) end: generate({62, 292, 1132})
 // (op 60) For Loop Increment
 // (op 60) generate({})
 // (op 60) end: generate({})
 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 if true)
 // (op 60) Generate LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 60) LessThan({DataFlowTag397: s14:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 60) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 60) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 60): s18, s19
s_ashr_i64 s[18:19], s[28:29], 6 // (op 60) 
// Allocated : 1 SGPR (Value: UInt32) (op 60): s15
s_mov_b32 s15, s18 // (op 60) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 60): s18, s19

// Allocated : 1 SGPR (Value: UInt32) (op 60): s18
s_and_b32 s18, s15, 4294967294 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s15

s_cmp_lt_u32 s14, s18 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 60) 
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60:
 // (op 60) 
 // (op 60) Unlock For Loop
 // Unlock None
 // (op 494) ForLoopOp KLoop: LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(60) END
 // (op 494) Deallocate{}(1221) BEGIN
 // (op 1221) Deallocate 507
 // Freeing DataFlowTag507: 2 VGPRs (Value: UInt64) (op 521): v18, v19

 // Deleting tag 507
 // (op 1221) Deallocate 510
 // Deleting tag 510
 // (op 1221) Deallocate 512
 // Deleting tag 512
 // (op 1221) Deallocate 624
 // Freeing DataFlowTag624: 1 VGPR (Value: UInt32) (op 909): v37

 // Deleting tag 624
 // (op 1221) Deallocate 650
 // Freeing DataFlowTag650: 1 SGPR (Value: UInt32) (op 996): s1

 // Deleting tag 650
 // (op 1221) Deallocate 651
 // Deleting tag 651
 // (op 1221) Deallocate 526
 // Freeing DataFlowTag526: 1 VGPR (Value: UInt32) (op 585): v15

 // Deleting tag 526
 // (op 1221) Deallocate 653
 // Deleting tag 653
 // (op 1221) Deallocate 662
 // Freeing DataFlowTag662: 1 VGPR (Value: UInt32) (op 1035): v17

 // Deleting tag 662
 // (op 1221) Deallocate 670
 // Freeing DataFlowTag670: 1 VGPR (Value: UInt32) (op 1059): v39

 // Deleting tag 670
 // (op 1221) Deallocate 671
 // Deleting tag 671
 // (op 1221) Deallocate 678
 // Freeing DataFlowTag678: 1 VGPR (Value: UInt32) (op 1083): v40

 // Deleting tag 678
 // (op 1221) Deallocate 681
 // Deleting tag 681
 // (op 1221) Deallocate 611
 // Deleting tag 611
 // (op 1221) Deallocate 625
 // Deleting tag 625
 // (op 1221) Deallocate 627
 // Deleting tag 627
 // (op 1221) Deallocate 665
 // Deleting tag 665
 // (op 1221) Deallocate 663
 // Deleting tag 663
 // (op 1221) Deallocate 679
 // Deleting tag 679
 // (op 1221) Deallocate 397
 // Freeing DataFlowTag397: 1 SGPR (Value: UInt32) (op 61): s14

 // Deleting tag 397
 // (op 1221) Deallocate 673
 // Deleting tag 673
 // (op 1221) Deallocate 474
 // Deleting tag 474
 // (op 1221) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 379): v21

 // Deleting tag 10
 // (op 1221) Deallocate 609
 // Deleting tag 609
 // (op 1221) Deallocate 470
 // Deleting tag 470
 // (op 1221) Deallocate 472
 // Deleting tag 472
 // (op 1221) Deallocate 482
 // Deleting tag 482
 // (op 1221) Deallocate 576
 // Freeing DataFlowTag576: 1 SGPR (Value: UInt32) (op 751): s0

 // Deleting tag 576
 // (op 1221) Deallocate 577
 // Deleting tag 577
 // (op 1221) Deallocate 579
 // Deleting tag 579
 // (op 1221) Deallocate 592
 // Freeing DataFlowTag592: 1 VGPR (Value: UInt32) (op 803): v16

 // Deleting tag 592
 // (op 1221) Deallocate 593
 // Deleting tag 593
 // (op 1221) Deallocate 595
 // Deleting tag 595
 // (op 1221) Deallocate 608
 // Freeing DataFlowTag608: 1 VGPR (Value: UInt32) (op 856): v36

 // Deleting tag 608
 // (op 1221) Deallocate 11
 // Freeing : 1 VGPR (Value: E8M0) (op 373): v20

 // Deleting tag 11
 // (op 1221) Deallocate 9
 // Freeing : 6 VGPRs (Value: FP6x16) (op 370): v24, v25, v26, v27, v28, v29

 // Deleting tag 9
 // (op 1221) Deallocate 569
 // Deleting tag 569
 // (op 1221) Deallocate 567
 // Deleting tag 567
 // (op 1221) Deallocate 565
 // Deleting tag 565
 // (op 1221) Deallocate 564
 // Freeing DataFlowTag564: 1 VGPR (Value: UInt32) (op 713): v23

 // Deleting tag 564
 // (op 1221) Deallocate 527
 // Deleting tag 527
 // (op 1221) Deallocate 546
 // Deleting tag 546
 // (op 1221) Deallocate 545
 // Freeing DataFlowTag545: 1 VGPR (Value: UInt32) (op 649): v13

 // Deleting tag 545
 // (op 1221) Deallocate 508
 // Deleting tag 508
 // (op 1221) Deallocate 548
 // Deleting tag 548
 // (op 1221) Deallocate 529
 // Deleting tag 529
 // (op 1221) Deallocate 478
 // Deleting tag 478
 // (op 1221) Deallocate 8
 // Deleting tag 8
 // (op 1221) Deallocate 550
 // Deleting tag 550
 // (op 1221) Deallocate 531
 // Deleting tag 531
 // (op 494) Deallocate{}(1221) END
 // (op 494) Barrier(1223) BEGIN
s_barrier  // (op 1223) 
 // (op 494) Barrier(1223) END
 // (op 494) end: generate({489})
 // (op 494) Unlock Scope 494
 // Unlock None
 // (op 344) Scope(494) END
 // (op 344) SetCoordinate(221) BEGIN
 // (op 221) generate({})
 // (op 221) end: generate({})
 // (op 221) generate({222})
 // (op 221) SetCoordinate(222) BEGIN
 // (op 222) generate({})
 // (op 222) end: generate({})
 // (op 222) generate({471})
 // (op 222) Scope(471) BEGIN
 // (op 471) Lock Scope 471
 // Lock Branch
 // (op 471) generate({465})
 // (op 471) NOP(465) BEGIN
 // (op 465) generate({})
 // (op 465) end: generate({})
 // (op 471) NOP(465) END
 // (op 471) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}(476) BEGIN
 // (op 476) Assign dim(494) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}
 // tag 494: v**UNALLOCATED**
 // (op 476) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64} into DataFlowTag494: VGPR Value: UInt64 x 1: (unallocated)
 // (op 476) reg expression
 // (op 476) reg expression
 // (op 476) reg expression
 // (op 476) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 476) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 476): v12
v_lshrrev_b32 v12, 2, v1 // (op 476) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 476): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 476) 
 // (op 476) BEGIN: Tile
 // (op 476) convert
// Allocated : 1 VGPR (Value: UInt32) (op 476): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 476) 
 // (op 476) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 476): v12

 // (op 476) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 476) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 476): s14
s_lshr_b32 s14, s0, 1 // (op 476) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 476): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 476): v12
v_and_b32 v12, -4, v1 // (op 476) 
 // (op 476) reg expression
 // (op 476) BEGIN: Flatten[1]
 // (op 476) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 476): v15
v_sub_u32 v15, v1, v12 // (op 476) 
 // (op 476) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 476): v12

 // (op 476) BEGIN: Tile
 // (op 476) {Tile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 476) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 476)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 476)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 476): v12
v_lshlrev_b32 v12, 4, v15 // (op 476) 
 // (op 476) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 476): v15

 // (op 476) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 476): v15
v_lshl_add_u32 v15, s14, 7, v12 // (op 476) 
 // (op 476) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 476): v12

 // Freeing : 1 SGPR (Value: UInt32) (op 476): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 476): v16, v17
v_mov_b32 v17, 0 // (op 476) convert
v_mov_b32 v16, v15 // (op 476) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 476): v15

 // (op 476) BEGIN: Split
 // (op 476) Multiply: dest (DataFlowTag494: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 476)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 476)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag494: 2 VGPRs (Value: UInt64) (op 476): v18, v19
v_mul_lo_u32 v19, v13, s3 // (op 476) most significant: low of low * high
 // (op 476) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 476): v12
v_mul_hi_u32 v12, v13, s2 // (op 476) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 476) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 476) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 476): v12

 // (op 476) Add: dest (DataFlowTag494: VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 476)      lhs (DataFlowTag494: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 476)      rhs (Tile: VGPR Value: UInt64 x 1: v[16:17])
// Allocated : 2 SGPRs (Value: Bool64) (op 476): s0, s1
v_add_co_u32 v18, s[0:1], v18, v16 // (op 476) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 476) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 476): s0, s1

 // (op 476) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 476): v16, v17

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 476): v13

 // (op 471) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}(476) END
 // (op 471) Assign VGPR 64:U64(477) BEGIN
 // (op 477) Assign dim(495) = 64:U64
 // (op 477) Assign stride expression
 // (op 471) Assign VGPR 64:U64(477) END
 // (op 471) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(478) BEGIN
 // (op 478) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 478) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // (op 478) reg expression
 // (op 478) reg expression
 // (op 478) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 478) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 478)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 478)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 478): s0, s1
s_mul_i32 s1, s2, s7 // (op 478) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 478): s14
s_mul_i32 s14, s3, s6 // (op 478) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 478) least significant
// Allocated : 1 SGPR (Value: Int32) (op 478): s15
s_mul_hi_u32 s15, s2, s6 // (op 478) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 478) most significant: sum
s_add_u32 s1, s1, s15 // (op 478) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 478): s15

 // Freeing : 1 SGPR (Value: Int32) (op 478): s14

 // (op 478) reg expression
 // (op 478) reg expression
s_mov_b32 s40, s4 // (op 478) reg expression
s_mov_b32 s41, s5 // (op 478) reg expression
s_mov_b32 s42, s0 // (op 478) reg expression
s_mov_b32 s43, 131072 // (op 478) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 478): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(478) END
 // (op 471) NOP(467) BEGIN
 // (op 467) generate({})
 // (op 467) end: generate({})
 // (op 471) NOP(467) END
 // (op 471) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(483) BEGIN
 // (op 483) Assign dim(499) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 483) Assign stride expression
 // (op 471) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(483) END
 // (op 471) NOP(468) BEGIN
 // (op 468) generate({})
 // (op 468) end: generate({})
 // (op 471) NOP(468) END
 // (op 471) Assign VGPR 1:U64(486) BEGIN
 // (op 486) Assign dim(501) = 1:U64
 // (op 486) Assign stride expression
 // (op 471) Assign VGPR 1:U64(486) END
 // (op 471) NOP(534) BEGIN
 // (op 534) generate({})
 // (op 534) end: generate({})
 // (op 471) NOP(534) END
 // (op 471) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(543) BEGIN
 // (op 543) Assign dim(513) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 513: v**UNALLOCATED**
 // (op 543) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag513: VGPR Value: UInt32 x 1: (unallocated)
 // (op 543) reg expression
 // (op 543) reg expression
 // (op 543) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 543) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 543): v12
v_and_b32 v12, -2, v1 // (op 543) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 543): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 543) 
 // (op 543) BEGIN: Flatten[1]
 // (op 543) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
 // (op 543) convert
// Allocated : 1 VGPR (Value: UInt32) (op 543): v13
v_sub_u32 v13, v1, v12 // (op 543) 
 // (op 543) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 543): v12

 // (op 543) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 543) BEGIN: Tile
 // (op 543) {Tile: ShiftL({Flatten[1]: v13:U32}, 5:U32)U32}
 // (op 543) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 543)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 543)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 SGPR (Value: UInt32) (op 543): s14
s_lshr_b32 s14, s0, 1 // (op 543) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 543): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 543): v12
v_lshlrev_b32 v12, 5, v13 // (op 543) 
 // (op 543) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 543): v13

 // (op 543) reg expression
 // (op 543) BEGIN: Tile
 // (op 543) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 543): v13
v_lshl_add_u32 v13, s14, 7, v12 // (op 543) 
 // (op 543) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 543): v12

 // Freeing : 1 SGPR (Value: UInt32) (op 543): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 543): v16, v17
v_mov_b32 v17, 0 // (op 543) convert
v_mov_b32 v16, v13 // (op 543) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 543): v13

 // (op 543) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 543): v15
v_lshrrev_b32 v15, 1, v1 // (op 543) 
 // (op 543) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 543): v12
v_lshl_add_u32 v12, s17, 5, v15 // (op 543) 
 // (op 543) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 543): v15

 // (op 543) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 543) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 543)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 543)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 543): v20, v21
v_mul_lo_u32 v21, v12, s9 // (op 543) most significant: low of low * high
 // (op 543) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 543): v13
v_mul_hi_u32 v13, v12, s8 // (op 543) most significant: high of low * low
v_mul_lo_u32 v20, v12, s8 // (op 543) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 543) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 543): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 543): v12

 // (op 543) BEGIN: Split
 // (op 543) {Split: Add({Tile: v[16:17]:U64}, v[20:21]:I64)U64}
 // (op 543) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 543)      lhs (Tile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 543)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated : 2 VGPRs (Value: UInt64) (op 543): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 543): s0, s1
v_add_co_u32 v12, s[0:1], v16, v20 // (op 543) least significant half
v_addc_co_u32 v13, s[0:1], v17, v21, s[0:1] // (op 543) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 543): s0, s1

 // (op 543) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 543): v20, v21

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 543): v16, v17

 // (op 543) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 543) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 543)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 543)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 543) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 543): v16, v17
v_mul_lo_u32 v17, v13, 6 // (op 543) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 543): v15
v_mul_hi_u32 v15, v12, 6 // (op 543) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 543) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 543) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 543): v15

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 543): v12, v13

 // (op 543) LogicalShiftR(v[16:17]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 543): v12, v13
v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 543) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 543): v16, v17

// Allocated DataFlowTag513: 1 VGPR (Value: UInt32) (op 543): v15
v_mov_b32 v15, v12 // (op 543) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 543): v12, v13

 // (op 471) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(543) END
 // (op 471) Assign VGPR 48:U64(544) BEGIN
 // (op 544) Assign dim(514) = 48:U64
 // (op 544) Assign stride expression
 // (op 471) Assign VGPR 48:U64(544) END
 // (op 471) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(545) BEGIN
 // (op 545) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 545) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // (op 545) reg expression
 // (op 545) reg expression
 // (op 545) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 545) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 545)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 545)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 545) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 545): s0, s1
s_mul_i32 s1, s13, 6 // (op 545) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 545) least significant
// Allocated : 1 SGPR (Value: Int32) (op 545): s14
s_mul_hi_u32 s14, s12, 6 // (op 545) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 545) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 545): s14

 // (op 545) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 545) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 545)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 545)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 545): s14, s15
s_mul_i32 s15, s8, s1 // (op 545) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 545): s18
s_mul_i32 s18, s9, s0 // (op 545) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 545) least significant
// Allocated : 1 SGPR (Value: Int32) (op 545): s19
s_mul_hi_u32 s19, s8, s0 // (op 545) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 545) most significant: sum
s_add_u32 s15, s15, s19 // (op 545) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 545): s19

 // Freeing : 1 SGPR (Value: Int32) (op 545): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 545): s0, s1

 // (op 545) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 545): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 545) 
 // Freeing : 2 SGPRs (Value: Int64) (op 545): s14, s15

 // (op 545) reg expression
 // (op 545) reg expression
s_mov_b32 s44, s10 // (op 545) reg expression
s_mov_b32 s45, s11 // (op 545) reg expression
s_mov_b32 s46, s0 // (op 545) reg expression
s_mov_b32 s47, 131072 // (op 545) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 545): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(545) END
 // (op 471) NOP(536) BEGIN
 // (op 536) generate({})
 // (op 536) end: generate({})
 // (op 471) NOP(536) END
 // (op 471) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(550) BEGIN
 // (op 550) Assign dim(518) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 550) Assign stride expression
 // (op 471) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(550) END
 // (op 471) NOP(537) BEGIN
 // (op 537) generate({})
 // (op 537) end: generate({})
 // (op 471) NOP(537) END
 // (op 471) Assign VGPR 0:U64(553) BEGIN
 // (op 553) Assign dim(520) = 0:U64
 // (op 553) Assign stride expression
 // (op 471) Assign VGPR 0:U64(553) END
 // (op 471) NOP(598) BEGIN
 // (op 598) generate({})
 // (op 598) end: generate({})
 // (op 471) NOP(598) END
 // (op 471) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}(607) BEGIN
 // (op 607) Assign dim(532) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}
 // tag 532: v**UNALLOCATED**
 // (op 607) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32} into DataFlowTag532: VGPR Value: UInt32 x 1: (unallocated)
 // (op 607) reg expression
 // (op 607) reg expression
 // (op 607) reg expression
 // (op 607) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 607) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 607): v12
v_lshrrev_b32 v12, 1, v1 // (op 607) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 607): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 607) 
 // (op 607) BEGIN: Tile
 // (op 607) convert
// Allocated : 1 VGPR (Value: UInt32) (op 607): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 607) 
 // (op 607) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 607): v12

 // (op 607) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 607) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 607): s14
s_lshr_b32 s14, s0, 1 // (op 607) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 607): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 607): v12
v_and_b32 v12, -2, v1 // (op 607) 
 // (op 607) reg expression
 // (op 607) BEGIN: Flatten[1]Tile
 // (op 607) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
 // (op 607) convert
// Allocated : 1 VGPR (Value: UInt32) (op 607): v16
v_sub_u32 v16, v1, v12 // (op 607) 
 // (op 607) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 607): v12

 // (op 607) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 607): v12
v_lshl_add_u32 v12, s14, 2, v16 // (op 607) 
 // (op 607) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 607): v16

 // Freeing : 1 SGPR (Value: UInt32) (op 607): s14

 // (op 607) BEGIN: Split
// Allocated DataFlowTag532: 1 VGPR (Value: UInt32) (op 607): v16
v_mul_lo_u32 v16, v13, s20 // (op 607) 
v_add_u32 v16, v16, v12 // (op 607) 
 // (op 607) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 607): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 607): v13

 // (op 471) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}(607) END
 // (op 471) Assign VGPR 2:U64(608) BEGIN
 // (op 608) Assign dim(533) = 2:U64
 // (op 608) Assign stride expression
 // (op 471) Assign VGPR 2:U64(608) END
 // (op 471) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(609) BEGIN
 // (op 609) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 609) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // (op 609) reg expression
 // (op 609) reg expression
 // (op 609) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 609) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 609)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 609)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 609): s0, s1
s_mul_i32 s1, s20, s7 // (op 609) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 609): s14
s_mul_i32 s14, s21, s6 // (op 609) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 609) least significant
// Allocated : 1 SGPR (Value: Int32) (op 609): s15
s_mul_hi_u32 s15, s20, s6 // (op 609) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 609) most significant: sum
s_add_u32 s1, s1, s15 // (op 609) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 609): s15

 // Freeing : 1 SGPR (Value: Int32) (op 609): s14

 // (op 609) reg expression
 // (op 609) reg expression
s_mov_b32 s48, s22 // (op 609) reg expression
s_mov_b32 s49, s23 // (op 609) reg expression
s_mov_b32 s50, s0 // (op 609) reg expression
s_mov_b32 s51, 131072 // (op 609) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 609): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(609) END
 // (op 471) NOP(600) BEGIN
 // (op 600) generate({})
 // (op 600) end: generate({})
 // (op 471) NOP(600) END
 // (op 471) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(614) BEGIN
 // (op 614) Assign dim(537) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 614) Assign stride expression
 // (op 471) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(614) END
 // (op 471) NOP(601) BEGIN
 // (op 601) generate({})
 // (op 601) end: generate({})
 // (op 471) NOP(601) END
 // (op 471) Assign VGPR 1:U64(617) BEGIN
 // (op 617) Assign dim(539) = 1:U64
 // (op 617) Assign stride expression
 // (op 471) Assign VGPR 1:U64(617) END
 // (op 471) NOP(662) BEGIN
 // (op 662) generate({})
 // (op 662) end: generate({})
 // (op 471) NOP(662) END
 // (op 471) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(671) BEGIN
 // (op 671) Assign dim(551) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 551: v**UNALLOCATED**
 // (op 671) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag551: VGPR Value: UInt32 x 1: (unallocated)
 // (op 671) reg expression
 // (op 671) reg expression
 // (op 671) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 671) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 671): v17
v_and_b32 v17, -2, v1 // (op 671) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 671): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 671) 
 // (op 671) BEGIN: Flatten[1]Tile
 // (op 671) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v17:U32)U32}
 // (op 671) convert
// Allocated : 1 VGPR (Value: UInt32) (op 671): v12
v_sub_u32 v12, v1, v17 // (op 671) 
 // (op 671) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 671): v17

 // (op 671) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 671) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 671): s14
s_lshr_b32 s14, s0, 1 // (op 671) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 671): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 671): v13
v_lshrrev_b32 v13, 1, v1 // (op 671) 
 // (op 671) reg expression
 // (op 671) BEGIN: Tile
 // (op 671) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 671): v17
v_lshl_add_u32 v17, s14, 2, v12 // (op 671) 
 // (op 671) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 671): v12

 // Freeing : 1 SGPR (Value: UInt32) (op 671): s14

 // (op 671) convert
 // (op 671) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 671): v12
v_lshl_add_u32 v12, s17, 5, v13 // (op 671) 
 // (op 671) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 671): v13

 // (op 671) Multiply({Tile: v12:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 671): v13
v_mul_lo_u32 v13, v12, s24 // (op 671) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 671): v12

 // (op 671) BEGIN: Split
 // (op 671) {Split: Add({Tile: v17:U32}, v13:U32)U32}
// Allocated DataFlowTag551: 1 VGPR (Value: UInt32) (op 671): v12
v_add_u32 v12, v17, v13 // (op 671) 
 // (op 671) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 671): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 671): v17

 // (op 471) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(671) END
 // (op 471) Assign VGPR 2:U64(672) BEGIN
 // (op 672) Assign dim(552) = 2:U64
 // (op 672) Assign stride expression
 // (op 471) Assign VGPR 2:U64(672) END
 // (op 471) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(673) BEGIN
 // (op 673) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 673) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // (op 673) reg expression
 // (op 673) reg expression
 // (op 673) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 673) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 673)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 673)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 673): s0, s1
s_mul_i32 s1, s24, s13 // (op 673) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 673): s14
s_mul_i32 s14, s25, s12 // (op 673) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 673) least significant
// Allocated : 1 SGPR (Value: Int32) (op 673): s15
s_mul_hi_u32 s15, s24, s12 // (op 673) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 673) most significant: sum
s_add_u32 s1, s1, s15 // (op 673) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 673): s15

 // Freeing : 1 SGPR (Value: Int32) (op 673): s14

 // (op 673) reg expression
 // (op 673) reg expression
s_mov_b32 s52, s26 // (op 673) reg expression
s_mov_b32 s53, s27 // (op 673) reg expression
s_mov_b32 s54, s0 // (op 673) reg expression
s_mov_b32 s55, 131072 // (op 673) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 673): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(673) END
 // (op 471) NOP(664) BEGIN
 // (op 664) generate({})
 // (op 664) end: generate({})
 // (op 471) NOP(664) END
 // (op 471) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(678) BEGIN
 // (op 678) Assign dim(556) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 678) Assign stride expression
 // (op 471) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(678) END
 // (op 471) NOP(665) BEGIN
 // (op 665) generate({})
 // (op 665) end: generate({})
 // (op 471) NOP(665) END
 // (op 471) Assign VGPR 1:U64(681) BEGIN
 // (op 681) Assign dim(558) = 1:U64
 // (op 681) Assign stride expression
 // (op 471) Assign VGPR 1:U64(681) END
 // (op 471) NOP(759) BEGIN
 // (op 759) generate({})
 // (op 759) end: generate({})
 // (op 471) NOP(759) END
 // (op 471) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(764) BEGIN
 // (op 764) Assign dim(580) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 580: s**UNALLOCATED**
 // (op 764) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag580: SGPR Value: UInt32 x 1: (unallocated)
 // (op 764) reg expression
 // (op 764) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 764): v13
v_and_b32 v13, -4, v1 // (op 764) 
 // (op 764) BEGIN: Tile[1]
 // (op 764) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 764): v17
v_sub_u32 v17, v1, v13 // (op 764) 
 // (op 764) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 764): v13

 // (op 764) BEGIN: Flatten
 // (op 764) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 764) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 764)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 764)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 764): v13
v_lshlrev_b32 v13, 4, v17 // (op 764) 
 // (op 764) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 764): v17

 // (op 764) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 764): v17
v_lshrrev_b32 v17, 2, v1 // (op 764) 
 // (op 764) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 764): v20
v_lshl_add_u32 v20, v17, 6, v13 // (op 764) 
 // (op 764) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 764): v13

 // Freeing : 1 VGPR (Value: UInt32) (op 764): v17

// Allocated DataFlowTag580: 1 SGPR (Value: UInt32) (op 764): s0
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 764) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 764): v20

 // (op 471) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(764) END
 // (op 471) Assign VGPR 1024:U32(765) BEGIN
 // (op 765) Assign dim(581) = 1024:U32
 // (op 765) Assign stride expression
 // (op 471) Assign VGPR 1024:U32(765) END
 // (op 471) NOP(760) BEGIN
 // (op 760) generate({})
 // (op 760) end: generate({})
 // (op 471) NOP(760) END
 // (op 471) Assign VGPR 1:U32(769) BEGIN
 // (op 769) Assign dim(583) = 1:U32
 // (op 769) Assign stride expression
 // (op 471) Assign VGPR 1:U32(769) END
 // (op 471) NOP(811) BEGIN
 // (op 811) generate({})
 // (op 811) end: generate({})
 // (op 471) NOP(811) END
 // (op 471) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(816) BEGIN
 // (op 816) Assign dim(596) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 596: v**UNALLOCATED**
 // (op 816) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag596: VGPR Value: UInt32 x 1: (unallocated)
 // (op 816) reg expression
 // (op 816) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 816): v13
v_and_b32 v13, -2, v1 // (op 816) 
 // (op 816) BEGIN: Tile[1]
 // (op 816) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 816): v17
v_sub_u32 v17, v1, v13 // (op 816) 
 // (op 816) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 816): v13

 // (op 816) BEGIN: Flatten
 // (op 816) {Flatten: ShiftL({Tile[1]: v17:U32}, 5:U32)U32}
 // (op 816) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 816)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 816)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 816): v13
v_lshlrev_b32 v13, 5, v17 // (op 816) 
 // (op 816) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 816): v17

 // (op 816) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 816): v17
v_lshrrev_b32 v17, 1, v1 // (op 816) 
 // (op 816) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 816): v20
v_lshl_add_u32 v20, v17, 6, v13 // (op 816) 
 // (op 816) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 816): v13

 // Freeing : 1 VGPR (Value: UInt32) (op 816): v17

 // (op 816) Multiply({Flatten: v20:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 816): v13
v_mul_lo_u32 v13, v20, 6 // (op 816) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 816): v20

 // (op 816) LogicalShiftR(v13:U32, 3:U32)U32
// Allocated DataFlowTag596: 1 VGPR (Value: UInt32) (op 816): v17
v_lshrrev_b32 v17, 3, v13 // (op 816) 
 // Freeing : 1 VGPR (Value: UInt32) (op 816): v13

 // (op 471) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(816) END
 // (op 471) Assign VGPR 48:U32(817) BEGIN
 // (op 817) Assign dim(597) = 48:U32
 // (op 817) Assign stride expression
 // (op 471) Assign VGPR 48:U32(817) END
 // (op 471) NOP(812) BEGIN
 // (op 812) generate({})
 // (op 812) end: generate({})
 // (op 471) NOP(812) END
 // (op 471) Assign VGPR 0:U32(821) BEGIN
 // (op 821) Assign dim(599) = 0:U32
 // (op 821) Assign stride expression
 // (op 471) Assign VGPR 0:U32(821) END
 // (op 471) NOP(864) BEGIN
 // (op 864) generate({})
 // (op 864) end: generate({})
 // (op 471) NOP(864) END
 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(869) BEGIN
 // (op 869) Assign dim(612) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 612: v**UNALLOCATED**
 // (op 869) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag612: VGPR Value: UInt32 x 1: (unallocated)
 // (op 869) reg expression
 // (op 869) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 869): v13
v_and_b32 v13, -2, v1 // (op 869) 
 // (op 869) BEGIN: Tile[1]Flatten
 // (op 869) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 869): v20
v_sub_u32 v20, v1, v13 // (op 869) 
 // (op 869) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 869): v13

 // (op 869) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 869): v13
v_and_b32 v13, 4294967294, v1 // (op 869) 
 // (op 869) BEGIN: Flatten
 // (op 869) {Flatten: Add(v13:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag612: 1 VGPR (Value: UInt32) (op 869): v21
v_add_u32 v21, v13, v20 // (op 869) 
 // (op 869) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 869): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 869): v13

 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(869) END
 // (op 471) Assign VGPR 2:U32(870) BEGIN
 // (op 870) Assign dim(613) = 2:U32
 // (op 870) Assign stride expression
 // (op 471) Assign VGPR 2:U32(870) END
 // (op 471) NOP(865) BEGIN
 // (op 865) generate({})
 // (op 865) end: generate({})
 // (op 471) NOP(865) END
 // (op 471) Assign VGPR 1:U32(874) BEGIN
 // (op 874) Assign dim(615) = 1:U32
 // (op 874) Assign stride expression
 // (op 471) Assign VGPR 1:U32(874) END
 // (op 471) NOP(917) BEGIN
 // (op 917) generate({})
 // (op 917) end: generate({})
 // (op 471) NOP(917) END
 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(922) BEGIN
 // (op 922) Assign dim(628) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 628: v**UNALLOCATED**
 // (op 922) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag628: VGPR Value: UInt32 x 1: (unallocated)
 // (op 922) reg expression
 // (op 922) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 922): v13
v_and_b32 v13, -2, v1 // (op 922) 
 // (op 922) BEGIN: Tile[1]Flatten
 // (op 922) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 922): v20
v_sub_u32 v20, v1, v13 // (op 922) 
 // (op 922) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 922): v13

 // (op 922) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 922): v13
v_and_b32 v13, 4294967294, v1 // (op 922) 
 // (op 922) BEGIN: Flatten
 // (op 922) {Flatten: Add(v13:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag628: 1 VGPR (Value: UInt32) (op 922): v23
v_add_u32 v23, v13, v20 // (op 922) 
 // (op 922) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 922): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 922): v13

 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(922) END
 // (op 471) Assign VGPR 2:U32(923) BEGIN
 // (op 923) Assign dim(629) = 2:U32
 // (op 923) Assign stride expression
 // (op 471) Assign VGPR 2:U32(923) END
 // (op 471) NOP(918) BEGIN
 // (op 918) generate({})
 // (op 918) end: generate({})
 // (op 471) NOP(918) END
 // (op 471) Assign VGPR 1:U32(927) BEGIN
 // (op 927) Assign dim(631) = 1:U32
 // (op 927) Assign stride expression
 // (op 471) Assign VGPR 1:U32(927) END
 // (op 471) ForLoopOp KLoopTail: LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(175) BEGIN
 // (op 175) Initialize For Loop
 // (op 175) generate({176})
 // (op 175) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(176) BEGIN
 // (op 176) Assign dim(452) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32
 // tag 452: s**UNALLOCATED**
 // (op 176) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag452: SGPR Value: UInt32 x 1: (unallocated)
 // (op 176) reg expression
 // (op 176) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 176): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 176) 
 // (op 176) convert
 // (op 176) BitwiseAnd({ convertForwardSubset: s14:U32}, 4294967294:U32)U32
// Allocated DataFlowTag452: 1 SGPR (Value: UInt32) (op 176): s1
s_and_b32 s1, s14, 4294967294 // (op 176) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 176): s14, s15

 // (op 175) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(176) END
 // (op 175) end: generate({176})
 // (op 175) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 175) 
 // (op 175) Generate LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 175) LessThan({DataFlowTag452: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 175) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 175): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 175) 
// Allocated : 1 SGPR (Value: UInt32) (op 175): s18
s_mov_b32 s18, s14 // (op 175) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 175): s14, s15

s_cmp_lt_u32 s1, s18 // (op 175) 
 // Freeing : 1 SGPR (Value: UInt32) (op 175): s18

 // Freeing : 6 VGPRs (Value: FP6x16) (op 382): v30, v31, v32, v33, v34, v35

 // Freeing : 1 VGPR (Value: E8M0) (op 385): v14

 // Freeing : 1 VGPR (Value: E8M0) (op 391): v22

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 175) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 175) 
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 175) 
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175:
 // (op 175) 
 // (op 175) generate({177, 1138})
 // (op 175) Assign SGPR Add(DataFlowTag(452)U32, 1:U32)U32(177) BEGIN
 // (op 177) Assign dim(452) = Add(DataFlowTag(452)U32, 1:U32)U32
 // (op 177) Generate Add(DataFlowTag(452)U32, 1:U32)U32 into DataFlowTag452: SGPR Value: UInt32 x 1: s1
 // (op 177) reg expression
 // (op 177) Add({DataFlowTag452: s1:U32}, 1:U32)U32
 // (op 175) Barrier(1138) BEGIN
s_barrier  // (op 1138) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 175) Barrier(1138) END
s_add_u32 s1, s1, 1 // (op 177) 
 // (op 175) Assign SGPR Add(DataFlowTag(452)U32, 1:U32)U32(177) END
 // (op 175) SetCoordinate(226) BEGIN
 // (op 226) generate({})
 // (op 226) end: generate({})
 // (op 226) generate({447})
 // (op 226) LoadTileDirect2LDS Value: BF8x4(447) BEGIN
 // (op 447) GEN: LoadTileDirect2LDS
 // (op 447) GEN: loadMacroTileDirect2LDS OP 447 LDS 17 MacroTile 454
 // (op 447) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 447) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride499: VGPR Value: UInt64 x 1: (unallocated)
 // (op 447) reg expression
 // (op 447) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 447) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 447)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 447)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 447): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 447) 
// Allocated Stride499: 2 VGPRs (Value: UInt64) (op 447): v24, v25
v_mov_b64 v[24:25], s[14:15] // (op 447) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 447): s14, s15

 // (op 447) Generate 1:U64 into nullptr
 // (op 447) Generate 1024:U32 into nullptr
 // (op 447) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride499: VGPR Value: UInt64 x 1: v[24:25]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 498: v**UNALLOCATED**
// Allocated Offset498: 2 VGPRs (Value: UInt64) (op 447): v26, v27
v_mov_b64 v[26:27], v[18:19] // (op 447) 
// Allocated : 2 VGPRs (Value: UInt64) (op 447): v28, v29
v_mov_b64 v[28:29], v[26:27] // (op 447) 
 // (op 447) Generate Add(0:U32, {DataFlowTag580Flatten: s0:U32})U32 into nullptr
 // (op 447)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride499: v[24:25]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 447) Generate {DataFlowTag580Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 447) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 447) Load value direct to lds
// VMEM: Expected complete at 4678 (current 4198)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 447) Generate Add(v[28:29]:U64, v24:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 447) reg expression
 // (op 447) reg expression
 // (op 447) Add(v[28:29]:U64, v24:R)U64
 // (op 447) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 447)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 447)      rhs (VGPR Value: Raw32 x 1: v24)
// Allocated : 2 SGPRs (Value: Bool64) (op 447): s14, s15
v_add_co_u32 v28, s[14:15], v28, v24 // (op 447) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 447) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 447): s14, s15

 // (op 447) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 447) reg expression
 // (op 447) Add(m0:U32, 1024:U32)U32
 // (op 447) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 447) 
 // (op 447) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 447) Load value direct to lds
// VMEM: Expected complete at 4682 (current 4202)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing Stride499: 2 VGPRs (Value: UInt64) (op 447): v24, v25

 // Freeing : 2 VGPRs (Value: UInt64) (op 447): v28, v29

 // (op 226) LoadTileDirect2LDS Value: BF8x4(447) END
 // (op 226) Deallocate{}(1315) BEGIN
 // (op 1315) Deallocate 582
 // Deleting tag 582
 // (op 1315) Deallocate 498
 // Freeing Offset498: 2 VGPRs (Value: UInt64) (op 447): v26, v27

 // Deleting tag 498
 // (op 1315) Deallocate 500
 // Deleting tag 500
 // (op 1315) Deallocate 454
 // Deleting tag 454
 // (op 226) Deallocate{}(1315) END
 // (op 226) Assign VGPR Add(DataFlowTag(494)U64, DataFlowTag(495)U64)U64(466) BEGIN
 // (op 466) Assign dim(494) = Add(DataFlowTag(494)U64, DataFlowTag(495)U64)U64
 // (op 466) Generate Add(DataFlowTag(494)U64, DataFlowTag(495)U64)U64 into DataFlowTag494: VGPR Value: UInt64 x 1: v[18:19]
 // (op 466) reg expression
 // (op 466) Add({DataFlowTag494: v[18:19]:U64}, 64:U64)U64
 // (op 466) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 466)      lhs (DataFlowTag494: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 466)      rhs (Literal Value: UInt64 x 0: 64)
// Allocated : 2 SGPRs (Value: Bool64) (op 466): s14, s15
v_add_co_u32 v18, s[14:15], 64, v18 // (op 466) least significant half
v_addc_co_u32 v19, s[14:15], 0, v19, s[14:15] // (op 466) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 466): s14, s15

 // (op 226) Assign VGPR Add(DataFlowTag(494)U64, DataFlowTag(495)U64)U64(466) END
 // (op 226) end: generate({447})
 // (op 175) SetCoordinate(226) END
 // (op 175) SetCoordinate(231) BEGIN
 // (op 231) generate({})
 // (op 231) end: generate({})
 // (op 231) generate({194})
 // (op 231) LoadTiled Value: FP6x16(194) BEGIN
 // (op 194) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 194) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 194) Generate 0:U64 into nullptr
 // tag 456: v**UNALLOCATED**
 // (op 194) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 517: v**UNALLOCATED**
 // (op 175) NOP(449) BEGIN
 // (op 449) generate({})
 // (op 449) end: generate({})
 // (op 175) NOP(449) END
// Allocated Offset517: 1 VGPR (Value: UInt32) (op 194): v20
v_mov_b32 v20, v15 // (op 194) 
 // (op 194) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = Offset517: VGPR Value: UInt32 x 1: v20
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 194) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 194): v24, v25, v26, v27, v28, v29
buffer_load_dwordx4 v[24:27], v20, s[44:47], 0 offen // (op 194) Load value
// VMEM: Expected complete at 4686 (current 4206)
buffer_load_dwordx2 v[28:29], v20, s[44:47], 0 offen offset:16 // (op 194) Load value
// VMEM: Expected stall of 472. CBNW: 0, Inc: 3
// VMEM: Expected complete at 5159 (current 4679)
 // (op 231) LoadTiled Value: FP6x16(194) END
 // (op 231) Deallocate{}(1249) BEGIN
 // (op 1249) Deallocate 519
 // Deleting tag 519
 // (op 1249) Deallocate 517
 // Freeing Offset517: 1 VGPR (Value: UInt32) (op 194): v20

 // Deleting tag 517
 // (op 231) Deallocate{}(1249) END
 // (op 231) Assign VGPR Add(DataFlowTag(513)U32, Convert(DataFlowTag(514)U64)U32)U32(535) BEGIN
 // (op 535) Assign dim(513) = Add(DataFlowTag(513)U32, Convert(DataFlowTag(514)U64)U32)U32
 // (op 535) Generate Add(DataFlowTag(513)U32, Convert(DataFlowTag(514)U64)U32)U32 into DataFlowTag513: VGPR Value: UInt32 x 1: v15
 // (op 535) reg expression
 // (op 535) Add({DataFlowTag513: v15:U32}, 48:U32)U32
v_add_u32 v15, 48, v15 // (op 535) 
 // (op 231) Assign VGPR Add(DataFlowTag(513)U32, Convert(DataFlowTag(514)U64)U32)U32(535) END
 // (op 231) end: generate({194})
 // (op 175) SetCoordinate(231) END
 // (op 175) SetCoordinate(236) BEGIN
 // (op 236) generate({})
 // (op 236) end: generate({})
 // (op 236) generate({195})
 // (op 236) LoadTiled Value: E8M0(195) BEGIN
 // (op 195) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 195) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 195) Generate 1:U64 into nullptr
 // tag 458: v**UNALLOCATED**
 // (op 195) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 536: v**UNALLOCATED**
 // (op 175) Barrier(1151) BEGIN
s_barrier  // (op 1151) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 175) Barrier(1151) END
// Allocated Offset536: 1 VGPR (Value: UInt32) (op 195): v20
v_mov_b32 v20, v16 // (op 195) 
 // (op 195) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset536: VGPR Value: UInt32 x 1: v20
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 195) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 175) StoreLDSTile Value: FP6x16(191) BEGIN
 // (op 191) GEN: StoreLDSTile
 // (op 191) GEN: storeMacroTileLDS OP 191 LDS 25 MacroTile 456
 // (op 191) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 191) Generate 0:U32 into nullptr
 // (op 191) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 191) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag596: VGPR Value: UInt32 x 1: v17
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 191) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 195): v22
buffer_load_ubyte v22, v20, s[48:51], 0 offen offset: 0 // (op 195) Load value
// VMEM: Expected complete at 5163 (current 4683)
 // (op 236) LoadTiled Value: E8M0(195) END
 // (op 236) Deallocate{}(1251) BEGIN
 // (op 1251) Deallocate 538
 // Deleting tag 538
 // (op 1251) Deallocate 536
 // Freeing Offset536: 1 VGPR (Value: UInt32) (op 195): v20

 // Deleting tag 536
 // (op 236) Deallocate{}(1251) END
 // (op 236) Assign VGPR Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32(599) BEGIN
 // (op 599) Assign dim(532) = Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32
 // (op 599) Generate Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32 into DataFlowTag532: VGPR Value: UInt32 x 1: v16
 // (op 599) reg expression
 // (op 599) Add({DataFlowTag532: v16:U32}, 2:U32)U32
v_add_u32 v16, 2, v16 // (op 599) 
 // (op 236) Assign VGPR Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32(599) END
 // (op 236) end: generate({195})
 // (op 175) SetCoordinate(236) END
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:4096 // (op 191) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 4627 (current 4532)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 175) SetCoordinate(241) BEGIN
 // (op 241) generate({})
 // (op 241) end: generate({})
 // (op 241) generate({196})
 // (op 241) LoadTiled Value: E8M0(196) BEGIN
 // (op 196) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 196) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 196) Generate 1:U64 into nullptr
 // tag 460: v**UNALLOCATED**
 // (op 196) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 555: v**UNALLOCATED**
// Allocated Offset555: 1 VGPR (Value: UInt32) (op 196): v20
v_mov_b32 v20, v12 // (op 196) 
 // (op 196) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset555: VGPR Value: UInt32 x 1: v20
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 196) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 196): v13
buffer_load_ubyte v13, v20, s[52:55], 0 offen offset: 0 // (op 196) Load value
// VMEM: Expected complete at 5170 (current 4690)
 // (op 241) LoadTiled Value: E8M0(196) END
 // (op 241) Deallocate{}(1253) BEGIN
 // (op 1253) Deallocate 557
 // Deleting tag 557
 // (op 1253) Deallocate 555
 // Freeing Offset555: 1 VGPR (Value: UInt32) (op 196): v20

 // Deleting tag 555
 // (op 241) Deallocate{}(1253) END
 // (op 241) Assign VGPR Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32(663) BEGIN
 // (op 663) Assign dim(551) = Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32
 // (op 663) Generate Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32 into DataFlowTag551: VGPR Value: UInt32 x 1: v12
 // (op 663) reg expression
 // (op 663) Add({DataFlowTag551: v12:U32}, 2:U32)U32
v_add_u32 v12, 2, v12 // (op 663) 
 // (op 241) Assign VGPR Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32(663) END
 // (op 241) end: generate({196})
 // (op 175) SetCoordinate(241) END
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[28:29] offset:4112 // (op 191) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 4632 (current 4537)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 175) StoreLDSTile Value: FP6x16(191) END
 // (op 175) Barrier(1141) BEGIN
 // (op 175) Deallocate{}(1243) BEGIN
 // (op 1243) Deallocate 598
 // Deleting tag 598
 // (op 1243) Deallocate 456
 // Freeing : 6 VGPRs (Value: FP6x16) (op 194): v24, v25, v26, v27, v28, v29

 // Deleting tag 456
 // (op 175) Deallocate{}(1243) END
s_barrier  // (op 1141) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 175) Barrier(1141) END
 // (op 175) Barrier(1165) BEGIN
 // (op 175) SetCoordinate(182) BEGIN
 // (op 182) generate({})
 // (op 182) end: generate({})
 // (op 182) generate({183})
 // (op 182) LoadLDSTile{Value: BF8x4}(183) BEGIN
 // (op 183) GEN: loadMacroTileWAVELDS OP 183 LDS 17 WaveTile 105
 // (op 183) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 183) Generate 1:U32 into nullptr
 // tag 462: v**UNALLOCATED**
 // (op 183) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 183) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 183) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

s_barrier  // (op 1165) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 175) Barrier(1165) END
// Allocated : 8 VGPRs (Value: BF8x4) (op 183): v24, v25, v26, v27, v28, v29, v30, v31
ds_read_b128 v[24:27], v3 // (op 183) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4635 (current 4540)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 175) StoreLDSTile Value: E8M0(192) BEGIN
 // (op 192) GEN: StoreLDSTile
 // (op 192) GEN: storeMacroTileLDS OP 192 LDS 33 MacroTile 458
 // (op 192) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 192) Generate 1:U32 into nullptr
 // (op 192) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 192) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = DataFlowTag612: VGPR Value: UInt32 x 1: v21
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 192) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_b128 v[28:31], v3 offset:32 // (op 183) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 87. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4723 (current 4628)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 182) LoadLDSTile{Value: BF8x4}(183) END
 // (op 182) end: generate({183})
 // (op 175) SetCoordinate(182) END
s_waitcnt vmcnt(1)
ds_write_b8 v21, v22 offset:5696 // (op 192) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 4. CBNW: 1, Inc: 3
// DSMEM: Expected complete at 4729 (current 4634)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 175) StoreLDSTile Value: E8M0(192) END
 // (op 175) Barrier(1155) BEGIN
 // (op 175) Deallocate{}(1245) BEGIN
 // (op 1245) Deallocate 614
 // Deleting tag 614
 // (op 1245) Deallocate 458
 // Freeing : 1 VGPR (Value: E8M0) (op 195): v22

 // Deleting tag 458
 // (op 175) Deallocate{}(1245) END
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1155) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 175) Barrier(1155) END
 // (op 175) Barrier(1179) BEGIN
 // (op 175) SetCoordinate(184) BEGIN
 // (op 184) generate({})
 // (op 184) end: generate({})
 // (op 184) generate({185})
 // (op 184) LoadLDSTile{Value: FP6x16}(185) BEGIN
 // (op 185) GEN: loadMacroTileWAVELDS OP 185 LDS 25 WaveTile 171
 // (op 185) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 185) Generate 0:U32 into nullptr
 // tag 464: v**UNALLOCATED**
 // (op 185) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 185) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 185) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_barrier  // (op 1179) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 175) Barrier(1179) END
// Allocated : 6 VGPRs (Value: FP6x16) (op 185): v32, v33, v34, v35, v36, v37
ds_read_b128 v[32:35], v4 offset:4096 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 4733 (current 4638)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 175) StoreLDSTile Value: E8M0(193) BEGIN
 // (op 193) GEN: StoreLDSTile
 // (op 193) GEN: storeMacroTileLDS OP 193 LDS 41 MacroTile 460
 // (op 193) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 193) Generate 1:U32 into nullptr
 // (op 193) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 193) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = DataFlowTag628: VGPR Value: UInt32 x 1: v23
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 193) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_b64 v[36:37], v4 offset:4112 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected stall of 85. CBNW: 1, Inc: 3
// DSMEM: Expected complete at 4819 (current 4724)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 184) LoadLDSTile{Value: FP6x16}(185) END
 // (op 184) end: generate({185})
 // (op 175) SetCoordinate(184) END
s_waitcnt vmcnt(0)
ds_write_b8 v23, v13 offset:5632 // (op 193) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 5. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 4826 (current 4731)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 175) StoreLDSTile Value: E8M0(193) END
 // (op 175) Barrier(1169) BEGIN
 // (op 175) Deallocate{}(1247) BEGIN
 // (op 1247) Deallocate 460
 // Freeing : 1 VGPR (Value: E8M0) (op 196): v13

 // Deleting tag 460
 // (op 1247) Deallocate 630
 // Deleting tag 630
 // (op 175) Deallocate{}(1247) END
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1169) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 175) Barrier(1169) END
 // (op 175) SetCoordinate(186) BEGIN
 // (op 186) generate({})
 // (op 186) end: generate({})
 // (op 186) generate({187})
 // (op 186) LoadLDSTile{Value: E8M0}(187) BEGIN
 // (op 187) GEN: loadMacroTileWAVELDS OP 187 LDS 33 WaveTile 237
 // (op 187) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 187) Generate 1:U32 into nullptr
 // tag 466: v**UNALLOCATED**
 // (op 187) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 187) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag620: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 187) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 187): v20
ds_read_u8 v20, v0 offset:5696 // (op 187) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4829 (current 4734)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 186) LoadLDSTile{Value: E8M0}(187) END
 // (op 186) end: generate({187})
 // (op 175) SetCoordinate(186) END
 // (op 175) Barrier(1183) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1183) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 175) Barrier(1183) END
 // (op 175) SetCoordinate(188) BEGIN
 // (op 188) generate({})
 // (op 188) end: generate({})
 // (op 188) generate({189})
 // (op 188) LoadLDSTile{Value: E8M0}(189) BEGIN
 // (op 189) GEN: loadMacroTileWAVELDS OP 189 LDS 41 WaveTile 297
 // (op 189) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 189) Generate 1:U32 into nullptr
 // tag 468: v**UNALLOCATED**
 // (op 189) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 189) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag636: VGPR Value: UInt32 x 1: v6
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 189) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 189): v22
ds_read_u8 v22, v6 offset:5632 // (op 189) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4924 (current 4829)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 188) LoadLDSTile{Value: E8M0}(189) END
 // (op 188) end: generate({189})
 // (op 175) SetCoordinate(188) END
 // (op 175) Multiply(190) BEGIN
 // (op 190) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile16xFP6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[24:31], v[32:37], a[0:15], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 190) 
 // (op 175) Multiply(190) END
 // (op 175) Deallocate{}(1241) BEGIN
 // (op 1241) Deallocate 462
 // Freeing : 8 VGPRs (Value: BF8x4) (op 183): v24, v25, v26, v27, v28, v29, v30, v31

 // Deleting tag 462
 // (op 1241) Deallocate 464
 // Freeing : 6 VGPRs (Value: FP6x16) (op 185): v32, v33, v34, v35, v36, v37

 // Deleting tag 464
 // (op 1241) Deallocate 466
 // Freeing : 1 VGPR (Value: E8M0) (op 187): v20

 // Deleting tag 466
 // (op 1241) Deallocate 468
 // Freeing : 1 VGPR (Value: E8M0) (op 189): v22

 // Deleting tag 468
 // (op 175) Deallocate{}(1241) END
 // (op 175) end: generate({177, 1138})
 // (op 175) For Loop Increment
 // (op 175) generate({})
 // (op 175) end: generate({})
 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 if true)
 // (op 175) Generate LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 175) LessThan({DataFlowTag452: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 175) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 175): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 175) 
// Allocated : 1 SGPR (Value: UInt32) (op 175): s18
s_mov_b32 s18, s14 // (op 175) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 175): s14, s15

s_cmp_lt_u32 s1, s18 // (op 175) 
 // Freeing : 1 SGPR (Value: UInt32) (op 175): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 175) 
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175:
 // (op 175) 
 // (op 175) Unlock For Loop
 // Unlock None
 // (op 471) ForLoopOp KLoopTail: LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(175) END
 // (op 471) Deallocate{}(1239) BEGIN
 // (op 1239) Deallocate 518
 // Deleting tag 518
 // (op 1239) Deallocate 537
 // Deleting tag 537
 // (op 1239) Deallocate 494
 // Freeing DataFlowTag494: 2 VGPRs (Value: UInt64) (op 476): v18, v19

 // Deleting tag 494
 // (op 1239) Deallocate 495
 // Deleting tag 495
 // (op 1239) Deallocate 499
 // Deleting tag 499
 // (op 1239) Deallocate 514
 // Deleting tag 514
 // (op 1239) Deallocate 533
 // Deleting tag 533
 // (op 1239) Deallocate 452
 // Freeing DataFlowTag452: 1 SGPR (Value: UInt32) (op 176): s1

 // Deleting tag 452
 // (op 1239) Deallocate 629
 // Deleting tag 629
 // (op 1239) Deallocate 520
 // Deleting tag 520
 // (op 1239) Deallocate 597
 // Deleting tag 597
 // (op 1239) Deallocate 513
 // Freeing DataFlowTag513: 1 VGPR (Value: UInt32) (op 543): v15

 // Deleting tag 513
 // (op 1239) Deallocate 581
 // Deleting tag 581
 // (op 1239) Deallocate 631
 // Deleting tag 631
 // (op 1239) Deallocate 501
 // Deleting tag 501
 // (op 1239) Deallocate 628
 // Freeing DataFlowTag628: 1 VGPR (Value: UInt32) (op 922): v23

 // Deleting tag 628
 // (op 1239) Deallocate 539
 // Deleting tag 539
 // (op 1239) Deallocate 558
 // Deleting tag 558
 // (op 1239) Deallocate 532
 // Freeing DataFlowTag532: 1 VGPR (Value: UInt32) (op 607): v16

 // Deleting tag 532
 // (op 1239) Deallocate 551
 // Freeing DataFlowTag551: 1 VGPR (Value: UInt32) (op 671): v12

 // Deleting tag 551
 // (op 1239) Deallocate 552
 // Deleting tag 552
 // (op 1239) Deallocate 583
 // Deleting tag 583
 // (op 1239) Deallocate 580
 // Freeing DataFlowTag580: 1 SGPR (Value: UInt32) (op 764): s0

 // Deleting tag 580
 // (op 1239) Deallocate 596
 // Freeing DataFlowTag596: 1 VGPR (Value: UInt32) (op 816): v17

 // Deleting tag 596
 // (op 1239) Deallocate 599
 // Deleting tag 599
 // (op 1239) Deallocate 612
 // Freeing DataFlowTag612: 1 VGPR (Value: UInt32) (op 869): v21

 // Deleting tag 612
 // (op 1239) Deallocate 613
 // Deleting tag 613
 // (op 1239) Deallocate 556
 // Deleting tag 556
 // (op 1239) Deallocate 615
 // Deleting tag 615
 // (op 471) Deallocate{}(1239) END
 // (op 471) end: generate({465})
 // (op 471) Unlock Scope 471
 // Unlock None
 // (op 222) Scope(471) END
 // (op 222) end: generate({471})
 // (op 221) SetCoordinate(222) END
 // (op 221) end: generate({222})
 // (op 344) SetCoordinate(221) END
 // (op 344) Deallocate{}(1255) BEGIN
 // (op 1255) Deallocate 4
 // Deleting tag 4
 // (op 1255) Deallocate 435
 // Deleting tag 435
 // (op 1255) Deallocate 535
 // Freeing DataFlowTag535: 4 SGPRs (Buffer: None) (op 632): s48, s49, s50, s51

 // Deleting tag 535
 // (op 1255) Deallocate 590
 // Deleting tag 590
 // (op 1255) Deallocate 606
 // Deleting tag 606
 // (op 1255) Deallocate 638
 // Deleting tag 638
 // (op 1255) Deallocate 622
 // Deleting tag 622
 // (op 1255) Deallocate 497
 // Freeing DataFlowTag497: 4 SGPRs (Buffer: None) (op 504): s40, s41, s42, s43

 // Deleting tag 497
 // (op 1255) Deallocate 159
 // Deleting tag 159
 // (op 1255) Deallocate 144
 // Deleting tag 144
 // (op 1255) Deallocate 17
 // Deleting tag 17
 // (op 1255) Deallocate 210
 // Deleting tag 210
 // (op 1255) Deallocate 2
 // Deleting tag 2
 // (op 1255) Deallocate 516
 // Freeing DataFlowTag516: 4 SGPRs (Buffer: None) (op 568): s44, s45, s46, s47

 // Deleting tag 516
 // (op 1255) Deallocate 78
 // Deleting tag 78
 // (op 1255) Deallocate 225
 // Deleting tag 225
 // (op 1255) Deallocate 41
 // Deleting tag 41
 // (op 1255) Deallocate 25
 // Deleting tag 25
 // (op 1255) Deallocate 3
 // Deleting tag 3
 // (op 1255) Deallocate 93
 // Deleting tag 93
 // (op 1255) Deallocate 554
 // Freeing DataFlowTag554: 4 SGPRs (Buffer: None) (op 696): s52, s53, s54, s55

 // Deleting tag 554
 // (op 1255) Deallocate 437
 // Deleting tag 437
 // (op 1255) Deallocate 270
 // Deleting tag 270
 // (op 1255) Deallocate 433
 // Deleting tag 433
 // (op 1255) Deallocate 431
 // Deleting tag 431
 // (op 1255) Deallocate 33
 // Deleting tag 33
 // (op 1255) Deallocate 285
 // Deleting tag 285
 // (op 344) Deallocate{}(1255) END
 // (op 344) Deallocate{Tensor_4_stride_0_6, Tensor_0_stride_0_0, Tensor_7_pointer, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer, Tensor_2_stride_1_3, Tensor_4_pointer}(1391) BEGIN
 // (op 1391) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1391) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1391) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1391) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1391) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1391) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1391) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1391) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 1391) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 1391) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 344) Deallocate{Tensor_4_stride_0_6, Tensor_0_stride_0_0, Tensor_7_pointer, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer, Tensor_2_stride_1_3, Tensor_4_pointer}(1391) END
 // (op 344) Barrier(1257) BEGIN
s_barrier  // (op 1257) 
 // (op 344) Barrier(1257) END
 // (op 344) end: generate({349})
 // (op 344) Unlock Scope 344
 // Unlock None
 // (op 1) Scope(344) END
 // (op 1) Deallocate{}(1295) BEGIN
 // (op 1295) Deallocate 637
 // Deleting tag 637
 // (op 1295) Deallocate 639
 // Deleting tag 639
 // (op 1295) Deallocate 669
 // Deleting tag 669
 // (op 1295) Deallocate 589
 // Deleting tag 589
 // (op 1295) Deallocate 674
 // Freeing DataFlowTag674: 1 VGPR (Value: UInt32) (op 1071): v5

 // Deleting tag 674
 // (op 1295) Deallocate 667
 // Deleting tag 667
 // (op 1295) Deallocate 658
 // Freeing DataFlowTag658: 1 VGPR (Value: UInt32) (op 1023): v7

 // Deleting tag 658
 // (op 1295) Deallocate 636
 // Freeing DataFlowTag636: 1 VGPR (Value: UInt32) (op 950): v6

 // Deleting tag 636
 // (op 1295) Deallocate 620
 // Freeing DataFlowTag620: 1 VGPR (Value: UInt32) (op 897): v0

 // Deleting tag 620
 // (op 1295) Deallocate 677
 // Deleting tag 677
 // (op 1295) Deallocate 591
 // Deleting tag 591
 // (op 1295) Deallocate 607
 // Deleting tag 607
 // (op 1295) Deallocate 661
 // Deleting tag 661
 // (op 1295) Deallocate 623
 // Deleting tag 623
 // (op 1295) Deallocate 682
 // Freeing DataFlowTag682: 1 VGPR (Value: UInt32) (op 1095): v10

 // Deleting tag 682
 // (op 1295) Deallocate 683
 // Deleting tag 683
 // (op 1295) Deallocate 685
 // Deleting tag 685
 // (op 1295) Deallocate 659
 // Deleting tag 659
 // (op 1295) Deallocate 621
 // Deleting tag 621
 // (op 1295) Deallocate 675
 // Deleting tag 675
 // (op 1295) Deallocate 588
 // Freeing DataFlowTag588: 1 VGPR (Value: UInt32) (op 791): v3

 // Deleting tag 588
 // (op 1295) Deallocate 666
 // Freeing DataFlowTag666: 1 VGPR (Value: UInt32) (op 1047): v8

 // Deleting tag 666
 // (op 1295) Deallocate 605
 // Deleting tag 605
 // (op 1295) Deallocate 604
 // Freeing DataFlowTag604: 1 VGPR (Value: UInt32) (op 844): v4

 // Deleting tag 604
 // (op 1) Deallocate{}(1295) END
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(135) BEGIN
 // (op 135) Lock for Conditional Branch
 // Lock Branch
 // (op 135) Generate Equal(0.00000:S, DataFlowTag(44)S)BL into VCC Value: Bool64 x 1: vcc
 // (op 135) reg expression
 // (op 135) Equal(0.00000:S, v9:S)BL64
v_cmp_eq_f32 vcc, 0.00000, v9 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
 // (op 135) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 135) 
s_cbranch_vccz RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135
 // (op 135) generate({352})
 // (op 135) Scope(352) BEGIN
 // (op 352) Lock Scope 352
 // Lock Branch
 // (op 352) generate({355})
 // (op 352) NOP(355) BEGIN
 // (op 355) generate({})
 // (op 355) end: generate({})
 // (op 352) NOP(355) END
 // (op 352) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(50) BEGIN
 // (op 50) Assign dim(59) = Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA
 // (op 50) Generate Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA into nullptr
 // (op 50) reg expression
 // (op 50) reg expression
 // (op 50) Multiply(v11:S, {DataFlowTag55: a[0:15]:S})S
// Allocated : 16 VGPRs (Value: Float) (op 50): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
s_nop 4
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
// Allocated : 16 VGPRs (Value: Float) (op 50): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43
v_mul_f32 v28, v11, v12 // (op 50) 
v_mul_f32 v29, v11, v13 // (op 50) 
v_mul_f32 v30, v11, v14 // (op 50) 
v_mul_f32 v31, v11, v15 // (op 50) 
v_mul_f32 v32, v11, v16 // (op 50) 
v_mul_f32 v33, v11, v17 // (op 50) 
v_mul_f32 v34, v11, v18 // (op 50) 
v_mul_f32 v35, v11, v19 // (op 50) 
v_mul_f32 v36, v11, v20 // (op 50) 
v_mul_f32 v37, v11, v21 // (op 50) 
v_mul_f32 v38, v11, v22 // (op 50) 
v_mul_f32 v39, v11, v23 // (op 50) 
v_mul_f32 v40, v11, v24 // (op 50) 
v_mul_f32 v41, v11, v25 // (op 50) 
v_mul_f32 v42, v11, v26 // (op 50) 
v_mul_f32 v43, v11, v27 // (op 50) 
 // Freeing : 16 VGPRs (Value: Float) (op 50): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // (op 352) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(50) END
 // (op 352) Assign VGPR Convert(DataFlowTag(59)NA)H(56) BEGIN
 // (op 56) Assign dim(61) = Convert(DataFlowTag(59)NA)H
 // (op 56) Generate Convert(DataFlowTag(59)NA)H into nullptr
 // (op 56) reg expression
// Allocated DataFlowTag59: 16 VGPRs (Value: Half) (op 56): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
v_cvt_f16_f32 v12, v28 // (op 56) 
v_cvt_f16_f32 v13, v29 // (op 56) 
v_cvt_f16_f32 v14, v30 // (op 56) 
v_cvt_f16_f32 v15, v31 // (op 56) 
v_cvt_f16_f32 v16, v32 // (op 56) 
v_cvt_f16_f32 v17, v33 // (op 56) 
v_cvt_f16_f32 v18, v34 // (op 56) 
v_cvt_f16_f32 v19, v35 // (op 56) 
v_cvt_f16_f32 v20, v36 // (op 56) 
v_cvt_f16_f32 v21, v37 // (op 56) 
v_cvt_f16_f32 v22, v38 // (op 56) 
v_cvt_f16_f32 v23, v39 // (op 56) 
v_cvt_f16_f32 v24, v40 // (op 56) 
v_cvt_f16_f32 v25, v41 // (op 56) 
v_cvt_f16_f32 v26, v42 // (op 56) 
v_cvt_f16_f32 v27, v43 // (op 56) 
 // (op 352) Assign VGPR Convert(DataFlowTag(59)NA)H(56) END
 // (op 352) Deallocate{}(1217) BEGIN
 // (op 1217) Deallocate 59
 // Freeing DataFlowTag59: 16 VGPRs (Value: Float) (op 50): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 59
 // (op 352) Deallocate{}(1217) END
 // (op 352) NOP(960) BEGIN
 // (op 960) generate({})
 // (op 960) end: generate({})
 // (op 352) NOP(960) END
 // (op 352) NOP(957) BEGIN
 // (op 957) generate({})
 // (op 957) end: generate({})
 // (op 352) NOP(957) END
 // (op 352) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(964) BEGIN
 // (op 964) Assign dim(640) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}
 // tag 640: v**UNALLOCATED**
 // (op 964) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32} into DataFlowTag640: VGPR Value: UInt32 x 1: (unallocated)
 // (op 964) reg expression
 // (op 964) reg expression
 // (op 964) reg expression
 // (op 964) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 964) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 964): v0
v_lshrrev_b32 v0, 6, v1 // (op 964) 
 // (op 964) convert
 // (op 964) BEGIN: Tile[1]
 // (op 964) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 964): v10
v_and_b32 v10, 63, v1 // (op 964) 
 // (op 964) END: Tile[1]
 // (op 964) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 964): v3
v_lshrrev_b32 v3, 5, v10 // (op 964) 
 // (op 964) BEGIN: Flatten
 // (op 964) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 964) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 964)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 964)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 964): v4
v_lshlrev_b32 v4, 2, v3 // (op 964) 
 // (op 964) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 964): v3

 // (op 964) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 964): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 964) 
 // (op 964) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 964): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 964): v0

 // (op 964) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 964): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 964) 
 // (op 964) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 964): v3

 // (op 964) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 964): v3
v_lshrrev_b32 v3, 2, v10 // (op 964) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 964): v10

 // (op 964) BEGIN: Tile[1]
 // (op 964) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 964): v10
v_and_b32 v10, 7, v3 // (op 964) 
 // (op 964) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 964): v3

 // (op 964) BEGIN: Tile[1]
 // (op 964) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 964): v3
v_and_b32 v3, 3, v1 // (op 964) 
 // (op 964) END: Tile[1]
 // (op 964) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 964): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 964) 
 // (op 964) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 964): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 964): v10

 // (op 964) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 964): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 964) 
 // (op 964) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 964): v4

 // (op 964) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 964): v10
v_mul_lo_u32 v10, v3, s30 // (op 964) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 964): v3

 // (op 964) BEGIN: Join
// Allocated DataFlowTag640: 1 VGPR (Value: UInt32) (op 964): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 964) 
 // (op 964) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 964): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 964): v0

 // (op 352) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(964) END
 // (op 352) Assign VGPR 16:U64(965) BEGIN
 // (op 965) Assign dim(641) = 16:U64
 // (op 965) Assign stride expression
 // (op 352) Assign VGPR 16:U64(965) END
 // (op 352) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(966) BEGIN
 // (op 966) Assign dim(643) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 643: s**UNALLOCATED**
 // (op 966) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag643: SGPR Buffer: None x 1: (unallocated)
 // (op 966) reg expression
 // (op 966) reg expression
 // (op 966) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 966) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 966)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 966)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 966): s14, s15
s_mul_i32 s15, s30, s13 // (op 966) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 966): s0
s_mul_i32 s0, s31, s12 // (op 966) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 966) least significant
// Allocated : 1 SGPR (Value: Int32) (op 966): s1
s_mul_hi_u32 s1, s30, s12 // (op 966) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 966) most significant: sum
s_add_u32 s15, s15, s1 // (op 966) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 966): s1

 // Freeing : 1 SGPR (Value: Int32) (op 966): s0

 // (op 966) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 966) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 966)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 966)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 966): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 966) 
 // Freeing : 2 SGPRs (Value: Int64) (op 966): s14, s15

 // (op 966) reg expression
 // (op 966) reg expression
// Allocated DataFlowTag643: 4 SGPRs (Buffer: None) (op 966): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 966) reg expression
s_mov_b32 s9, s33 // (op 966) reg expression
s_mov_b32 s10, s0 // (op 966) reg expression
s_mov_b32 s11, 131072 // (op 966) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 966): s0, s1

 // (op 352) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(966) END
 // (op 352) NOP(958) BEGIN
 // (op 958) generate({})
 // (op 958) end: generate({})
 // (op 352) NOP(958) END
 // (op 352) Assign VGPR 2:U64(971) BEGIN
 // (op 971) Assign dim(645) = 2:U64
 // (op 971) Assign stride expression
 // (op 352) Assign VGPR 2:U64(971) END
 // (op 352) StoreTiled Value: Halfx2(58) BEGIN
 // (op 58) GEN: storeMacroTileWAVE OP 58 MacroTile 61 WaveTile 367
 // (op 58) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 58) Generate 16:U64 into nullptr
 // (op 58) Generate 2:U64 into nullptr
 // (op 58) Convert in LSTGen
 // (op 58) Generate Convert({DataFlowTag59: v[12:27]:H})2xH into nullptr
 // (op 58) reg expression
// Allocated DataFlowTag59: 8 VGPRs (Value: Halfx2) (op 58): v28, v29, v30, v31, v32, v33, v34, v35
v_pack_b32_f16 v28, v12, v13 // (op 58) 
v_pack_b32_f16 v29, v14, v15 // (op 58) 
v_pack_b32_f16 v30, v16, v17 // (op 58) 
v_pack_b32_f16 v31, v18, v19 // (op 58) 
v_pack_b32_f16 v32, v20, v21 // (op 58) 
v_pack_b32_f16 v33, v22, v23 // (op 58) 
v_pack_b32_f16 v34, v24, v25 // (op 58) 
v_pack_b32_f16 v35, v26, v27 // (op 58) 
 // (op 58) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag59: VGPR Value: Halfx2 x 8: v[28:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 58) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag59: VGPR Value: Halfx2 x 8: v[28:35]
 // 	info.rowOffsetReg = DataFlowTag640: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 58) 	info.m = 4
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 16
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx2 v[28:29], v3, s[8:11], 0 offen // (op 58) Store value
// VMEM: Expected complete at 5939 (current 5459)
buffer_store_dwordx2 v[30:31], v3, s[8:11], 0 offen offset:16 // (op 58) Store value
// VMEM: Expected complete at 5940 (current 5460)
buffer_store_dwordx2 v[32:33], v3, s[8:11], 0 offen offset:32 // (op 58) Store value
// VMEM: Expected complete at 5941 (current 5461)
buffer_store_dwordx2 v[34:35], v3, s[8:11], 0 offen offset:48 // (op 58) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 6420 (current 5940)
 // Freeing DataFlowTag59: 8 VGPRs (Value: Halfx2) (op 58): v28, v29, v30, v31, v32, v33, v34, v35

 // (op 352) StoreTiled Value: Halfx2(58) END
 // (op 352) Deallocate{}(1219) BEGIN
 // (op 1219) Deallocate 641
 // Deleting tag 641
 // (op 1219) Deallocate 61
 // Freeing DataFlowTag59: 16 VGPRs (Value: Half) (op 56): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 61
 // (op 1219) Deallocate 645
 // Deleting tag 645
 // (op 1219) Deallocate 644
 // Deleting tag 644
 // (op 1219) Deallocate 640
 // Freeing DataFlowTag640: 1 VGPR (Value: UInt32) (op 964): v3

 // Deleting tag 640
 // (op 352) Deallocate{}(1219) END
 // (op 352) end: generate({355})
 // (op 352) Unlock Scope 352
 // Unlock None
 // (op 135) Scope(352) END
 // (op 135) end: generate({352})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 135) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 135) 
s_branch RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135:
 // (op 135) 
 // (op 135) generate({358})
 // (op 135) Scope(358) BEGIN
 // (op 358) Lock Scope 358
 // Lock Branch
 // (op 358) generate({361})
 // (op 358) NOP(361) BEGIN
 // (op 361) generate({})
 // (op 361) end: generate({})
 // (op 358) NOP(361) END
 // (op 358) NOP(143) BEGIN
 // (op 143) generate({})
 // (op 143) end: generate({})
 // (op 358) NOP(143) END
 // (op 358) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(144) BEGIN
 // (op 144) Assign dim(425) = Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA
 // (op 144) Generate Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA into nullptr
 // (op 144) reg expression
 // (op 144) reg expression
 // (op 144) Multiply(v11:S, {DataFlowTag55: a[0:15]:S})S
 // (op 358) Scope(729) BEGIN
 // (op 729) Lock Scope 729
 // Lock Branch
 // (op 729) generate({726})
 // (op 729) NOP(726) BEGIN
 // (op 726) generate({})
 // (op 726) end: generate({})
 // (op 729) NOP(726) END
 // (op 729) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}(734) BEGIN
 // (op 734) Assign dim(570) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}
 // tag 570: v**UNALLOCATED**
 // (op 734) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32} into DataFlowTag570: VGPR Value: UInt32 x 1: (unallocated)
 // (op 734) reg expression
 // (op 734) reg expression
 // (op 734) reg expression
 // (op 734) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 734) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 734): v0
v_lshrrev_b32 v0, 6, v1 // (op 734) 
 // (op 734) convert
 // (op 734) BEGIN: Flatten[1]
 // (op 734) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 734): v10
v_and_b32 v10, 63, v1 // (op 734) 
 // (op 734) END: Flatten[1]
 // (op 734) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 734): v3
v_lshrrev_b32 v3, 5, v10 // (op 734) 
 // (op 734) BEGIN: Tile
 // (op 734) {Tile: ShiftL(v3:U32, 2:U32)U32}
 // (op 734) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 734)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 734)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 734): v4
v_lshlrev_b32 v4, 2, v3 // (op 734) 
 // (op 734) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 734): v3

 // (op 734) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 734): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 734) 
 // (op 734) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 734): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 734): v0

 // (op 734) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 734): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 734) 
 // (op 734) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 734): v3

 // (op 734) LogicalShiftR({Flatten[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 734): v3
v_lshrrev_b32 v3, 2, v10 // (op 734) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 734): v10

 // (op 734) BEGIN: Flatten[1]
 // (op 734) {Flatten[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 734): v10
v_and_b32 v10, 7, v3 // (op 734) 
 // (op 734) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 734): v3

 // (op 734) BEGIN: Flatten[1]
 // (op 734) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 734): v3
v_and_b32 v3, 3, v1 // (op 734) 
 // (op 734) END: Flatten[1]
 // (op 734) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 734): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 734) 
 // (op 734) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 734): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 734): v10

 // (op 734) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 734): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 734) 
 // (op 734) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 734): v4

 // (op 734) Multiply({Tile: v3:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 734): v10
v_mul_lo_u32 v10, v3, s34 // (op 734) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 734): v3

 // (op 734) BEGIN: Split
// Allocated DataFlowTag570: 1 VGPR (Value: UInt32) (op 734): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 734) 
 // (op 734) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 734): v10

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 734): v0

 // (op 729) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}(734) END
 // (op 729) Deallocate{}(1323) BEGIN
 // (op 1323) Deallocate 439
 // Deleting tag 439
 // (op 1323) Deallocate 441
 // Deleting tag 441
 // (op 1323) Deallocate 326
 // Deleting tag 326
 // (op 729) Deallocate{}(1323) END
 // (op 729) Assign VGPR 16:U64(735) BEGIN
 // (op 735) Assign dim(571) = 16:U64
 // (op 735) Assign stride expression
 // (op 729) Assign VGPR 16:U64(735) END
 // (op 729) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(736) BEGIN
 // (op 736) Assign dim(573) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 573: s**UNALLOCATED**
 // (op 736) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag573: SGPR Buffer: None x 1: (unallocated)
 // (op 736) reg expression
 // (op 736) reg expression
 // (op 736) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 736) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 736)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 736)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 736): s14, s15
s_mul_i32 s15, s34, s13 // (op 736) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 736): s0
s_mul_i32 s0, s35, s12 // (op 736) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 736) least significant
// Allocated : 1 SGPR (Value: Int32) (op 736): s1
s_mul_hi_u32 s1, s34, s12 // (op 736) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 736) most significant: sum
s_add_u32 s15, s15, s1 // (op 736) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 736): s1

 // Freeing : 1 SGPR (Value: Int32) (op 736): s0

 // (op 736) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 736) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 736)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 736)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 736): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 736) 
 // Freeing : 2 SGPRs (Value: Int64) (op 736): s14, s15

 // (op 736) reg expression
 // (op 736) reg expression
// Allocated DataFlowTag573: 4 SGPRs (Buffer: None) (op 736): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 736) reg expression
s_mov_b32 s5, s37 // (op 736) reg expression
s_mov_b32 s6, s0 // (op 736) reg expression
s_mov_b32 s7, 131072 // (op 736) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 736): s0, s1

 // (op 729) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(736) END
 // (op 729) Deallocate{Tensor_10_pointer}(1395) BEGIN
 // (op 1395) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 729) Deallocate{Tensor_10_pointer}(1395) END
 // (op 729) NOP(727) BEGIN
 // (op 727) generate({})
 // (op 727) end: generate({})
 // (op 729) NOP(727) END
 // (op 729) Assign VGPR 2:U64(741) BEGIN
 // (op 741) Assign dim(575) = 2:U64
 // (op 741) Assign stride expression
 // (op 729) Assign VGPR 2:U64(741) END
 // (op 729) LoadTiled Value: Halfx2(148) BEGIN
 // (op 148) GEN: loadMacroTileWAVECIACCUM OP 148 WaveTile 327
 // (op 148) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 148) Generate 16:U64 into nullptr
 // (op 148) Generate 2:U64 into nullptr
 // tag 421: v**UNALLOCATED**
 // (op 148) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Halfx2 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag573: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 148) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Halfx2 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag570: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag573: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 148) 	info.m = 4
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 16
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 8 VGPRs (Value: Halfx2) (op 148): v12, v13, v14, v15, v16, v17, v18, v19
buffer_load_dwordx2 v[12:13], v3, s[4:7], 0 offen // (op 148) Load value
// VMEM: Expected complete at 6928 (current 6448)
buffer_load_dwordx2 v[14:15], v3, s[4:7], 0 offen offset:16 // (op 148) Load value
// VMEM: Expected complete at 6929 (current 6449)
buffer_load_dwordx2 v[16:17], v3, s[4:7], 0 offen offset:32 // (op 148) Load value
// VMEM: Expected complete at 6930 (current 6450)
buffer_load_dwordx2 v[18:19], v3, s[4:7], 0 offen offset:48 // (op 148) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 7409 (current 6929)
 // (op 729) LoadTiled Value: Halfx2(148) END
 // (op 729) Deallocate{Tensor_10_stride_1_10}(1235) BEGIN
 // (op 1235) Deallocate 571
 // Deleting tag 571
 // (op 1235) Deallocate 573
 // Freeing DataFlowTag573: 4 SGPRs (Buffer: None) (op 736): s4, s5, s6, s7

 // Deleting tag 573
 // (op 1235) Deallocate 575
 // Deleting tag 575
 // (op 1235) Deallocate 570
 // Freeing DataFlowTag570: 1 VGPR (Value: UInt32) (op 734): v3

 // Deleting tag 570
 // (op 1235) Deallocate 5
 // Deleting tag 5
 // (op 1235) Deallocate 574
 // Deleting tag 574
 // (op 1235) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 729) Deallocate{Tensor_10_stride_1_10}(1235) END
 // (op 729) end: generate({726})
 // (op 729) Unlock Scope 729
 // Unlock None
 // (op 358) Scope(729) END
// Allocated : 16 VGPRs (Value: Float) (op 144): v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35
v_accvgpr_read v20, a0 // (op 144) 
 // (op 358) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(421)NA)NA(149) BEGIN
 // (op 149) Assign dim(423) = Multiply(DataFlowTag(44)NA, DataFlowTag(421)NA)NA
 // (op 149) Generate Multiply(DataFlowTag(44)NA, DataFlowTag(421)NA)NA into nullptr
 // (op 149) reg expression
 // (op 149) reg expression
 // (op 149) Multiply(v9:S, v[12:19]:2xH)S
v_accvgpr_read v21, a1 // (op 144) 
v_accvgpr_read v22, a2 // (op 144) 
v_accvgpr_read v23, a3 // (op 144) 
v_accvgpr_read v24, a4 // (op 144) 
v_accvgpr_read v25, a5 // (op 144) 
v_accvgpr_read v26, a6 // (op 144) 
v_accvgpr_read v27, a7 // (op 144) 
v_accvgpr_read v28, a8 // (op 144) 
v_accvgpr_read v29, a9 // (op 144) 
v_accvgpr_read v30, a10 // (op 144) 
v_accvgpr_read v31, a11 // (op 144) 
v_accvgpr_read v32, a12 // (op 144) 
v_accvgpr_read v33, a13 // (op 144) 
v_accvgpr_read v34, a14 // (op 144) 
v_accvgpr_read v35, a15 // (op 144) 
// Allocated : 16 VGPRs (Value: Float) (op 144): v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51
v_mul_f32 v36, v11, v20 // (op 144) 
v_mul_f32 v37, v11, v21 // (op 144) 
v_mul_f32 v38, v11, v22 // (op 144) 
v_mul_f32 v39, v11, v23 // (op 144) 
v_mul_f32 v40, v11, v24 // (op 144) 
v_mul_f32 v41, v11, v25 // (op 144) 
v_mul_f32 v42, v11, v26 // (op 144) 
v_mul_f32 v43, v11, v27 // (op 144) 
v_mul_f32 v44, v11, v28 // (op 144) 
v_mul_f32 v45, v11, v29 // (op 144) 
v_mul_f32 v46, v11, v30 // (op 144) 
v_mul_f32 v47, v11, v31 // (op 144) 
v_mul_f32 v48, v11, v32 // (op 144) 
v_mul_f32 v49, v11, v33 // (op 144) 
v_mul_f32 v50, v11, v34 // (op 144) 
v_mul_f32 v51, v11, v35 // (op 144) 
 // Freeing : 16 VGPRs (Value: Float) (op 144): v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35

 // (op 358) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(144) END
// Allocated : 2 VGPRs (Value: Float) (op 149): v20, v21
s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 149) 
v_lshrrev_b32 v21, 16, v12 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
// Allocated : 16 VGPRs (Value: Float) (op 149): v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67
v_mul_f32 v52, v9, v20 // (op 149) 
v_mul_f32 v53, v9, v21 // (op 149) 
v_cvt_f32_f16 v20, v13 // (op 149) 
v_lshrrev_b32 v21, 16, v13 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v54, v9, v20 // (op 149) 
v_mul_f32 v55, v9, v21 // (op 149) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 149) 
v_lshrrev_b32 v21, 16, v14 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v56, v9, v20 // (op 149) 
v_mul_f32 v57, v9, v21 // (op 149) 
v_cvt_f32_f16 v20, v15 // (op 149) 
v_lshrrev_b32 v21, 16, v15 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v58, v9, v20 // (op 149) 
v_mul_f32 v59, v9, v21 // (op 149) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 149) 
v_lshrrev_b32 v21, 16, v16 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v60, v9, v20 // (op 149) 
v_mul_f32 v61, v9, v21 // (op 149) 
v_cvt_f32_f16 v20, v17 // (op 149) 
v_lshrrev_b32 v21, 16, v17 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v62, v9, v20 // (op 149) 
v_mul_f32 v63, v9, v21 // (op 149) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 149) 
v_lshrrev_b32 v21, 16, v18 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v64, v9, v20 // (op 149) 
v_mul_f32 v65, v9, v21 // (op 149) 
v_cvt_f32_f16 v20, v19 // (op 149) 
v_lshrrev_b32 v21, 16, v19 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v66, v9, v20 // (op 149) 
v_mul_f32 v67, v9, v21 // (op 149) 
 // Freeing : 2 VGPRs (Value: Float) (op 149): v20, v21

 // (op 358) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(421)NA)NA(149) END
 // (op 358) Deallocate{}(1237) BEGIN
 // (op 1237) Deallocate 421
 // Freeing : 8 VGPRs (Value: Halfx2) (op 148): v12, v13, v14, v15, v16, v17, v18, v19

 // Deleting tag 421
 // (op 358) Deallocate{}(1237) END
 // (op 358) Assign VGPR Add(DataFlowTag(423)NA, DataFlowTag(425)NA)NA(145) BEGIN
 // (op 145) Assign dim(427) = Add(DataFlowTag(423)NA, DataFlowTag(425)NA)NA
 // (op 145) Generate Add(DataFlowTag(423)NA, DataFlowTag(425)NA)NA into nullptr
 // (op 145) reg expression
 // (op 145) reg expression
 // (op 145) Add({DataFlowTag423: v[52:67]:S}, {DataFlowTag425: v[36:51]:S})S
// Allocated : 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
v_add_f32 v12, v52, v36 // (op 145) 
v_add_f32 v13, v53, v37 // (op 145) 
v_add_f32 v14, v54, v38 // (op 145) 
v_add_f32 v15, v55, v39 // (op 145) 
v_add_f32 v16, v56, v40 // (op 145) 
v_add_f32 v17, v57, v41 // (op 145) 
v_add_f32 v18, v58, v42 // (op 145) 
v_add_f32 v19, v59, v43 // (op 145) 
v_add_f32 v20, v60, v44 // (op 145) 
v_add_f32 v21, v61, v45 // (op 145) 
v_add_f32 v22, v62, v46 // (op 145) 
v_add_f32 v23, v63, v47 // (op 145) 
v_add_f32 v24, v64, v48 // (op 145) 
v_add_f32 v25, v65, v49 // (op 145) 
v_add_f32 v26, v66, v50 // (op 145) 
v_add_f32 v27, v67, v51 // (op 145) 
 // (op 358) Assign VGPR Add(DataFlowTag(423)NA, DataFlowTag(425)NA)NA(145) END
 // (op 358) Deallocate{}(1229) BEGIN
 // (op 1229) Deallocate 423
 // Freeing DataFlowTag423: 16 VGPRs (Value: Float) (op 149): v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67

 // Deleting tag 423
 // (op 1229) Deallocate 425
 // Freeing DataFlowTag425: 16 VGPRs (Value: Float) (op 144): v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51

 // Deleting tag 425
 // (op 358) Deallocate{}(1229) END
 // (op 358) Assign VGPR Convert(DataFlowTag(427)NA)H(146) BEGIN
 // (op 146) Assign dim(429) = Convert(DataFlowTag(427)NA)H
 // (op 146) Generate Convert(DataFlowTag(427)NA)H into nullptr
 // (op 146) reg expression
// Allocated DataFlowTag427: 16 VGPRs (Value: Half) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43
v_cvt_f16_f32 v28, v12 // (op 146) 
v_cvt_f16_f32 v29, v13 // (op 146) 
v_cvt_f16_f32 v30, v14 // (op 146) 
v_cvt_f16_f32 v31, v15 // (op 146) 
v_cvt_f16_f32 v32, v16 // (op 146) 
v_cvt_f16_f32 v33, v17 // (op 146) 
v_cvt_f16_f32 v34, v18 // (op 146) 
v_cvt_f16_f32 v35, v19 // (op 146) 
v_cvt_f16_f32 v36, v20 // (op 146) 
v_cvt_f16_f32 v37, v21 // (op 146) 
v_cvt_f16_f32 v38, v22 // (op 146) 
v_cvt_f16_f32 v39, v23 // (op 146) 
v_cvt_f16_f32 v40, v24 // (op 146) 
v_cvt_f16_f32 v41, v25 // (op 146) 
v_cvt_f16_f32 v42, v26 // (op 146) 
v_cvt_f16_f32 v43, v27 // (op 146) 
 // (op 358) Assign VGPR Convert(DataFlowTag(427)NA)H(146) END
 // (op 358) Deallocate{}(1231) BEGIN
 // (op 1231) Deallocate 427
 // Freeing DataFlowTag427: 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 427
 // (op 358) Deallocate{}(1231) END
 // (op 358) NOP(977) BEGIN
 // (op 977) generate({})
 // (op 977) end: generate({})
 // (op 358) NOP(977) END
 // (op 358) NOP(974) BEGIN
 // (op 974) generate({})
 // (op 974) end: generate({})
 // (op 358) NOP(974) END
 // (op 358) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(981) BEGIN
 // (op 981) Assign dim(646) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}
 // tag 646: v**UNALLOCATED**
 // (op 981) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32} into DataFlowTag646: VGPR Value: UInt32 x 1: (unallocated)
 // (op 981) reg expression
 // (op 981) reg expression
 // (op 981) reg expression
 // (op 981) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 981) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 981): v0
v_lshrrev_b32 v0, 6, v1 // (op 981) 
 // (op 981) convert
 // (op 981) BEGIN: Tile[1]
 // (op 981) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 981): v10
v_and_b32 v10, 63, v1 // (op 981) 
 // (op 981) END: Tile[1]
 // (op 981) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 981): v3
v_lshrrev_b32 v3, 5, v10 // (op 981) 
 // (op 981) BEGIN: Flatten
 // (op 981) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 981) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 981)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 981)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 981): v4
v_lshlrev_b32 v4, 2, v3 // (op 981) 
 // (op 981) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 981): v3

 // (op 981) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 981): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 981) 
 // (op 981) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 981): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 981): v0

 // (op 981) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 981): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 981) 
 // (op 981) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 981): v3

 // (op 981) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 981): v3
v_lshrrev_b32 v3, 2, v10 // (op 981) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 981): v10

 // (op 981) BEGIN: Tile[1]
 // (op 981) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 981): v10
v_and_b32 v10, 7, v3 // (op 981) 
 // (op 981) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 981): v3

 // (op 981) BEGIN: Tile[1]
 // (op 981) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 981): v3
v_and_b32 v3, 3, v1 // (op 981) 
 // (op 981) END: Tile[1]
 // (op 981) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 981): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 981) 
 // (op 981) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 981): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 981): v10

 // (op 981) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 981): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 981) 
 // (op 981) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 981): v4

 // (op 981) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 981): v10
v_mul_lo_u32 v10, v3, s30 // (op 981) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 981): v3

 // (op 981) BEGIN: Join
// Allocated DataFlowTag646: 1 VGPR (Value: UInt32) (op 981): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 981) 
 // (op 981) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 981): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 981): v0

 // (op 358) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(981) END
 // (op 358) Assign VGPR 16:U64(982) BEGIN
 // (op 982) Assign dim(647) = 16:U64
 // (op 982) Assign stride expression
 // (op 358) Assign VGPR 16:U64(982) END
 // (op 358) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(983) BEGIN
 // (op 983) Assign dim(643) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 983) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // (op 983) reg expression
 // (op 983) reg expression
 // (op 983) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 983) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 983)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 983)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 983): s14, s15
s_mul_i32 s15, s30, s13 // (op 983) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 983): s0
s_mul_i32 s0, s31, s12 // (op 983) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 983) least significant
// Allocated : 1 SGPR (Value: Int32) (op 983): s1
s_mul_hi_u32 s1, s30, s12 // (op 983) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 983) most significant: sum
s_add_u32 s15, s15, s1 // (op 983) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 983): s1

 // Freeing : 1 SGPR (Value: Int32) (op 983): s0

 // (op 983) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 983) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 983)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 983)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 983): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 983) 
 // Freeing : 2 SGPRs (Value: Int64) (op 983): s14, s15

 // (op 983) reg expression
 // (op 983) reg expression
s_mov_b32 s8, s32 // (op 983) reg expression
s_mov_b32 s9, s33 // (op 983) reg expression
s_mov_b32 s10, s0 // (op 983) reg expression
s_mov_b32 s11, 131072 // (op 983) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 983): s0, s1

 // (op 358) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(983) END
 // (op 358) NOP(975) BEGIN
 // (op 975) generate({})
 // (op 975) end: generate({})
 // (op 358) NOP(975) END
 // (op 358) Assign VGPR 2:U64(988) BEGIN
 // (op 988) Assign dim(649) = 2:U64
 // (op 988) Assign stride expression
 // (op 358) Assign VGPR 2:U64(988) END
 // (op 358) StoreTiled Value: Halfx2(147) BEGIN
 // (op 147) GEN: storeMacroTileWAVE OP 147 MacroTile 429 WaveTile 367
 // (op 147) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 147) Generate 16:U64 into nullptr
 // (op 147) Generate 2:U64 into nullptr
 // (op 147) Convert in LSTGen
 // (op 147) Generate Convert({DataFlowTag427: v[28:43]:H})2xH into nullptr
 // (op 147) reg expression
// Allocated DataFlowTag427: 8 VGPRs (Value: Halfx2) (op 147): v12, v13, v14, v15, v16, v17, v18, v19
v_pack_b32_f16 v12, v28, v29 // (op 147) 
v_pack_b32_f16 v13, v30, v31 // (op 147) 
v_pack_b32_f16 v14, v32, v33 // (op 147) 
v_pack_b32_f16 v15, v34, v35 // (op 147) 
v_pack_b32_f16 v16, v36, v37 // (op 147) 
v_pack_b32_f16 v17, v38, v39 // (op 147) 
v_pack_b32_f16 v18, v40, v41 // (op 147) 
v_pack_b32_f16 v19, v42, v43 // (op 147) 
 // (op 147) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag427: VGPR Value: Halfx2 x 8: v[12:19]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 147) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag427: VGPR Value: Halfx2 x 8: v[12:19]
 // 	info.rowOffsetReg = DataFlowTag646: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 147) 	info.m = 4
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 16
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx2 v[12:13], v3, s[8:11], 0 offen // (op 147) Store value
// VMEM: Expected complete at 7965 (current 7485)
buffer_store_dwordx2 v[14:15], v3, s[8:11], 0 offen offset:16 // (op 147) Store value
// VMEM: Expected complete at 7966 (current 7486)
buffer_store_dwordx2 v[16:17], v3, s[8:11], 0 offen offset:32 // (op 147) Store value
// VMEM: Expected complete at 7967 (current 7487)
buffer_store_dwordx2 v[18:19], v3, s[8:11], 0 offen offset:48 // (op 147) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 8446 (current 7966)
 // Freeing DataFlowTag427: 8 VGPRs (Value: Halfx2) (op 147): v12, v13, v14, v15, v16, v17, v18, v19

 // (op 358) StoreTiled Value: Halfx2(147) END
 // (op 358) Deallocate{}(1233) BEGIN
 // (op 1233) Deallocate 649
 // Deleting tag 649
 // (op 1233) Deallocate 648
 // Deleting tag 648
 // (op 1233) Deallocate 647
 // Deleting tag 647
 // (op 1233) Deallocate 646
 // Freeing DataFlowTag646: 1 VGPR (Value: UInt32) (op 981): v3

 // Deleting tag 646
 // (op 1233) Deallocate 429
 // Freeing DataFlowTag427: 16 VGPRs (Value: Half) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 429
 // (op 358) Deallocate{}(1233) END
 // (op 358) end: generate({361})
 // (op 358) Unlock Scope 358
 // Unlock None
 // (op 135) Scope(358) END
 // (op 135) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1297) BEGIN
 // (op 1297) Deallocate 643
 // Freeing DataFlowTag643: 4 SGPRs (Buffer: None) (op 966): s8, s9, s10, s11

 // Deleting tag 643
 // (op 1297) Deallocate 445
 // Deleting tag 445
 // (op 1297) Deallocate 443
 // Deleting tag 443
 // (op 1297) Deallocate 366
 // Deleting tag 366
 // (op 1297) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 1297) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 135) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1297) END
 // (op 135) end: generate({358})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 135) 
RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135:
 // (op 135) 
 // (op 135) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(135) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1227) BEGIN
 // (op 1227) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 2): v9

 // Deleting tag 44
 // (op 1227) Deallocate 53
 // Freeing : 1 VGPR (Value: Float) (op 43): v11

 // Deleting tag 53
 // (op 1227) Deallocate 55
 // Freeing DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 65): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15

 // Deleting tag 55
 // (op 1227) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1227) END
 // (op 1) end: generate({787, 840, 893, 946, 1019, 1043, 1067, 1091})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 7424 // lds bytes
  .amdhsa_accum_offset 68
 // Initial kernel state
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_length 14
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_system_sgpr_workgroup_id_y 1
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 1
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocroller
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: F6
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 7424
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 68
    .agpr_count: 16
    .max_flat_workgroup_size: 64
    .workgroup_size: [64, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

