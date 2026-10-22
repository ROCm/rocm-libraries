// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.FP6.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.FP6.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
 // T_LOAD_TILED 3 Source 2
 // Tensor.E8M0.d2 4, (base=&80, sizes={&88 &96 }, strides={&104 &112 })
 // T_LOAD_TILED 5 Source 4
 // BlockScale(Separate, {1, 32}): Data: 1, Scale: 5
 // Tensor.E8M0.d2 7, (base=&120, sizes={&128 &136 }, strides={&144 &152 })
 // T_LOAD_TILED 8 Source 7
 // BlockScale(Separate, {32, 1}): Data: 3, Scale: 8
 // Tensor.Float.d2 10, (base=&160, sizes={&168 &176 }, strides={&184 &192 })
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
 // Tensor.Float.d2 21, (base=&208, sizes={&216 &224 }, strides={&232 &240 })
 // T_STORE_TILED 22 Source 19 Dest 21

 // Command arguments:
 // Tensor_0_pointer: PointerGlobal: FP6(offset: 0, size: 8, read_write)
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
 // Tensor_10_pointer: PointerGlobal: Float(offset: 160, size: 8, read_write)
 // Tensor_10_size_0: Value: Int64(offset: 168, size: 8, read_only)
 // Tensor_10_size_1: Value: Int64(offset: 176, size: 8, read_only)
 // Tensor_10_stride_0: Value: Int64(offset: 184, size: 8, read_only)
 // Tensor_10_stride_1: Value: Int64(offset: 192, size: 8, read_only)
 // user_Float_Value_12: Value: Float(offset: 200, size: 4, read_only)
 // user_Float_Value_14: Value: Float(offset: 204, size: 4, read_only)
 // Tensor_21_pointer: PointerGlobal: Float(offset: 208, size: 8, read_write)
 // Tensor_21_size_0: Value: Int64(offset: 216, size: 8, read_only)
 // Tensor_21_size_1: Value: Int64(offset: 224, size: 8, read_only)
 // Tensor_21_stride_0: Value: Int64(offset: 232, size: 8, read_only)
 // Tensor_21_stride_1: Value: Int64(offset: 240, size: 8, read_only)

 // Kernel Arguments:
 // KernelArg{Tensor_0_stride_0_0, Value: Int64, CommandArgument(Tensor_0_stride_0)I64(c 0), o:0, s:8, preloaded}
 // KernelArg{Tensor_0_pointer, PointerGlobal: FP6, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
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
 // KernelArg{Tensor_21_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}
.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2:
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
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_exec_begin:
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
 // KernelArg{Tensor_0_pointer, PointerGlobal: FP6, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
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
 // KernelArg{Tensor_21_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}

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
 // Tensor_0_pointer: SGPR PointerGlobal: FP6 x 1: s[4:5]
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
 // Tensor_21_pointer: SGPR PointerGlobal: Float x 1: s[32:33]
 // Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]
 // Tensor_10_pointer: SGPR PointerGlobal: Float x 1: s[36:37]
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
 // (op 1) generate({735, 788, 841, 899, 923, 947})
 // (op 1) NOP(735) BEGIN
 // (op 735) generate({})
 // (op 735) end: generate({})
 // (op 1) NOP(735) END
 // (op 1) NOP(788) BEGIN
 // (op 788) generate({})
 // (op 788) end: generate({})
 // (op 1) NOP(788) END
 // (op 1) NOP(841) BEGIN
 // (op 841) generate({})
 // (op 841) end: generate({})
 // (op 1) NOP(841) END
 // (op 1) NOP(899) BEGIN
 // (op 899) generate({})
 // (op 899) end: generate({})
 // (op 1) NOP(899) END
 // (op 1) NOP(923) BEGIN
 // (op 923) generate({})
 // (op 923) end: generate({})
 // (op 1) NOP(923) END
 // (op 1) NOP(947) BEGIN
 // (op 947) generate({})
 // (op 947) end: generate({})
 // (op 1) NOP(947) END
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 6:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(739) BEGIN
 // (op 739) Assign dim(544) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 6:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 544: v**UNALLOCATED**
 // (op 739) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 6:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag544: VGPR Value: UInt32 x 1: (unallocated)
 // (op 739) reg expression
 // (op 739) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Flatten[1]TileTile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(792) BEGIN
 // (op 792) Assign dim(560) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Flatten[1]TileTile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 560: v**UNALLOCATED**
 // (op 792) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Flatten[1]TileTile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 792) reg expression
 // (op 792) BEGIN: Flatten[1]TileTile
 // (op 792) {Flatten[1]TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(230)U32, 6:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 63:U32)U32}, 4:U32)U32)U32}(845) BEGIN
 // (op 845) Assign dim(576) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(230)U32, 6:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 576: v**UNALLOCATED**
 // (op 845) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(230)U32, 6:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag576: VGPR Value: UInt32 x 1: (unallocated)
 // (op 845) reg expression
 // (op 845) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 6:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(903) BEGIN
 // (op 903) Assign dim(594) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 6:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 594: v**UNALLOCATED**
 // (op 903) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 6:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag594: VGPR Value: UInt32 x 1: (unallocated)
 // (op 903) reg expression
 // (op 903) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Flatten[1]TileTile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(927) BEGIN
 // (op 927) Assign dim(602) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Flatten[1]TileTile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 602: v**UNALLOCATED**
 // (op 927) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Flatten[1]TileTile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag602: VGPR Value: UInt32 x 1: (unallocated)
 // (op 927) reg expression
 // (op 927) BEGIN: Flatten[1]TileTile
 // (op 927) {Flatten[1]TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(230)U32, 6:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 63:U32)U32}, 4:U32)U32)U32}(951) BEGIN
 // (op 951) Assign dim(610) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(230)U32, 6:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 610: v**UNALLOCATED**
 // (op 951) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(230)U32, 6:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag610: VGPR Value: UInt32 x 1: (unallocated)
 // (op 951) reg expression
 // (op 951) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 739): v0
v_lshrrev_b32 v0, 6, v1 // (op 739) 
 // (op 739) BEGIN: Flatten[1]Tile
 // (op 739) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 739): v3
v_and_b32 v3, 15, v1 // (op 739) 
 // (op 739) END: Flatten[1]Tile
 // (op 739) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 739): v4
v_lshl_add_u32 v4, v0, 4, v3 // (op 739) 
 // (op 739) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 739): v3

 // Freeing : 1 VGPR (Value: UInt32) (op 739): v0

 // (op 739) BEGIN: Flatten[1]
 // (op 739) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 739): v0
v_and_b32 v0, 63, v1 // (op 739) 
 // (op 739) END: Flatten[1]
 // (op 739) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 739): v3
v_lshrrev_b32 v3, 4, v0 // (op 739) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 739): v0

 // (op 739) BEGIN: TileTile
 // (op 739) {TileTile: ShiftL(v3:U32, 5:U32)U32}
 // (op 739) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 739)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 739)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 739): v0
v_lshlrev_b32 v0, 5, v3 // (op 739) 
 // (op 739) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 739): v3

 // (op 739) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 739): v3
v_lshl_add_u32 v3, v4, 7, v0 // (op 739) 
 // (op 739) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 739): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 739): v4

 // (op 739) Multiply({Tile: v3:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 739): v0
v_mul_lo_u32 v0, v3, 6 // (op 739) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 739): v3

 // (op 739) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag544: 1 VGPR (Value: UInt32) (op 739): v3
v_lshrrev_b32 v3, 3, v0 // (op 739) 
 // Freeing : 1 VGPR (Value: UInt32) (op 739): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 6:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(739) END
// Allocated : 1 VGPR (Value: UInt32) (op 792): v0
v_and_b32 v0, 15, v1 // (op 792) 
 // (op 792) END: Flatten[1]TileTile
 // (op 792) BEGIN: Flatten[1]
 // (op 792) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 792): v4
v_and_b32 v4, 63, v1 // (op 792) 
 // (op 792) END: Flatten[1]
 // (op 792) LogicalShiftR({Flatten[1]: v4:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 792): v5
v_lshrrev_b32 v5, 4, v4 // (op 792) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 792): v4

 // (op 792) BEGIN: TileTile
 // (op 792) {TileTile: ShiftL(v5:U32, 5:U32)U32}
 // (op 792) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 792)         value (VGPR Value: UInt32 x 1: v5) 
 // (op 792)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 792): v4
v_lshlrev_b32 v4, 5, v5 // (op 792) 
 // (op 792) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 792): v5

 // (op 792) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 792): v5
v_lshl_add_u32 v5, v0, 7, v4 // (op 792) 
 // (op 792) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 792): v4

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 792): v0

 // (op 792) Multiply({Tile: v5:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 792): v0
v_mul_lo_u32 v0, v5, 6 // (op 792) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 792): v5

 // (op 792) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 792): v4
v_lshrrev_b32 v4, 3, v0 // (op 792) 
 // Freeing : 1 VGPR (Value: UInt32) (op 792): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Flatten[1]TileTile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(792) END
// Allocated : 1 VGPR (Value: UInt32) (op 845): v0
v_lshrrev_b32 v0, 6, v1 // (op 845) 
 // (op 845) BEGIN: Flatten[1]
 // (op 845) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 845): v5
v_and_b32 v5, 15, v1 // (op 845) 
 // (op 845) END: Flatten[1]
 // (op 845) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 845): v6
v_lshl_add_u32 v6, v0, 4, v5 // (op 845) 
 // (op 845) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 845): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 845): v0

 // (op 845) BEGIN: Flatten[1]
 // (op 845) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 845): v0
v_and_b32 v0, 63, v1 // (op 845) 
 // (op 845) END: Flatten[1]
 // (op 845) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 845): v5
v_lshrrev_b32 v5, 4, v0 // (op 845) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 845): v0

 // (op 845) BEGIN: Tile
// Allocated DataFlowTag576: 1 VGPR (Value: UInt32) (op 845): v0
v_lshl_add_u32 v0, v6, 2, v5 // (op 845) 
 // (op 845) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 845): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 845): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(230)U32, 6:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 63:U32)U32}, 4:U32)U32)U32}(845) END
// Allocated : 1 VGPR (Value: UInt32) (op 903): v5
v_lshrrev_b32 v5, 6, v1 // (op 903) 
 // (op 903) BEGIN: Flatten[1]Tile
 // (op 903) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 903): v6
v_and_b32 v6, 15, v1 // (op 903) 
 // (op 903) END: Flatten[1]Tile
 // (op 903) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 903): v7
v_lshl_add_u32 v7, v5, 4, v6 // (op 903) 
 // (op 903) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 903): v6

 // Freeing : 1 VGPR (Value: UInt32) (op 903): v5

 // (op 903) BEGIN: Flatten[1]
 // (op 903) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 903): v5
v_and_b32 v5, 63, v1 // (op 903) 
 // (op 903) END: Flatten[1]
 // (op 903) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 903): v6
v_lshrrev_b32 v6, 4, v5 // (op 903) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 903): v5

 // (op 903) BEGIN: TileTile
 // (op 903) {TileTile: ShiftL(v6:U32, 5:U32)U32}
 // (op 903) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 903)         value (VGPR Value: UInt32 x 1: v6) 
 // (op 903)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 903): v5
v_lshlrev_b32 v5, 5, v6 // (op 903) 
 // (op 903) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 903): v6

 // (op 903) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 903): v6
v_lshl_add_u32 v6, v7, 7, v5 // (op 903) 
 // (op 903) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 903): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 903): v7

 // (op 903) Multiply({Tile: v6:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 903): v5
v_mul_lo_u32 v5, v6, 6 // (op 903) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 903): v6

 // (op 903) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag594: 1 VGPR (Value: UInt32) (op 903): v6
v_lshrrev_b32 v6, 3, v5 // (op 903) 
 // Freeing : 1 VGPR (Value: UInt32) (op 903): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 6:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(903) END
// Allocated : 1 VGPR (Value: UInt32) (op 927): v5
v_and_b32 v5, 15, v1 // (op 927) 
 // (op 927) END: Flatten[1]TileTile
 // (op 927) BEGIN: Flatten[1]
 // (op 927) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Deallocate{}(1141) BEGIN
 // (op 1141) Deallocate 98
 // Deleting tag 98
 // (op 1) Deallocate{}(1141) END
// Allocated : 1 VGPR (Value: UInt32) (op 927): v7
v_and_b32 v7, 63, v1 // (op 927) 
 // (op 927) END: Flatten[1]
 // (op 927) LogicalShiftR({Flatten[1]: v7:U32}, 4:U32)U32
 // (op 1) Assign VGPR 96:U32(740) BEGIN
 // (op 740) Assign dim(545) = 96:U32
 // (op 740) Assign stride expression
 // (op 1) Assign VGPR 96:U32(740) END
 // (op 1) Assign VGPR 96:U32(904) BEGIN
 // (op 904) Assign dim(595) = 96:U32
 // (op 904) Assign stride expression
 // (op 1) Assign VGPR 96:U32(904) END
 // (op 1) NOP(736) BEGIN
 // (op 736) generate({})
 // (op 736) end: generate({})
 // (op 1) NOP(736) END
 // (op 1) NOP(900) BEGIN
 // (op 900) generate({})
 // (op 900) end: generate({})
 // (op 1) NOP(900) END
 // (op 1) Assign VGPR 0:U32(744) BEGIN
 // (op 744) Assign dim(547) = 0:U32
 // (op 744) Assign stride expression
 // (op 1) Assign VGPR 0:U32(744) END
 // (op 1) Assign VGPR 0:U32(908) BEGIN
 // (op 908) Assign dim(597) = 0:U32
 // (op 908) Assign stride expression
 // (op 1) Assign VGPR 0:U32(908) END
// Allocated : 1 VGPR (Value: UInt32) (op 927): v8
v_lshrrev_b32 v8, 4, v7 // (op 927) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 927): v7

 // (op 927) BEGIN: TileTile
 // (op 927) {TileTile: ShiftL(v8:U32, 5:U32)U32}
 // (op 927) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 927)         value (VGPR Value: UInt32 x 1: v8) 
 // (op 927)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 927): v7
v_lshlrev_b32 v7, 5, v8 // (op 927) 
 // (op 927) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 927): v8

 // (op 927) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 927): v8
v_lshl_add_u32 v8, v5, 7, v7 // (op 927) 
 // (op 927) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 927): v7

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 927): v5

 // (op 927) Multiply({Tile: v8:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 927): v5
v_mul_lo_u32 v5, v8, 6 // (op 927) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 927): v8

 // (op 927) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag602: 1 VGPR (Value: UInt32) (op 927): v7
v_lshrrev_b32 v7, 3, v5 // (op 927) 
 // Freeing : 1 VGPR (Value: UInt32) (op 927): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Flatten[1]TileTile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(927) END
// Allocated : 1 VGPR (Value: UInt32) (op 951): v5
v_lshrrev_b32 v5, 6, v1 // (op 951) 
 // (op 951) BEGIN: Flatten[1]
 // (op 951) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Deallocate{}(1144) BEGIN
 // (op 1144) Deallocate 164
 // Deleting tag 164
 // (op 1) Deallocate{}(1144) END
// Allocated : 1 VGPR (Value: UInt32) (op 951): v8
v_and_b32 v8, 15, v1 // (op 951) 
 // (op 951) END: Flatten[1]
 // (op 951) BEGIN: Tile
 // (op 1) Assign VGPR 96:U32(793) BEGIN
 // (op 793) Assign dim(561) = 96:U32
 // (op 793) Assign stride expression
 // (op 1) Assign VGPR 96:U32(793) END
 // (op 1) Assign VGPR 96:U32(928) BEGIN
 // (op 928) Assign dim(603) = 96:U32
 // (op 928) Assign stride expression
 // (op 1) Assign VGPR 96:U32(928) END
 // (op 1) NOP(789) BEGIN
 // (op 789) generate({})
 // (op 789) end: generate({})
 // (op 1) NOP(789) END
 // (op 1) NOP(924) BEGIN
 // (op 924) generate({})
 // (op 924) end: generate({})
 // (op 1) NOP(924) END
 // (op 1) Assign VGPR 0:U32(797) BEGIN
 // (op 797) Assign dim(563) = 0:U32
 // (op 797) Assign stride expression
 // (op 1) Assign VGPR 0:U32(797) END
 // (op 1) Assign VGPR 0:U32(932) BEGIN
 // (op 932) Assign dim(605) = 0:U32
 // (op 932) Assign stride expression
 // (op 1) Assign VGPR 0:U32(932) END
// Allocated : 1 VGPR (Value: UInt32) (op 951): v9
v_lshl_add_u32 v9, v5, 4, v8 // (op 951) 
 // (op 951) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 951): v8

 // Freeing : 1 VGPR (Value: UInt32) (op 951): v5

 // (op 951) BEGIN: Flatten[1]
 // (op 951) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 951): v5
v_and_b32 v5, 63, v1 // (op 951) 
 // (op 951) END: Flatten[1]
 // (op 951) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 951): v8
v_lshrrev_b32 v8, 4, v5 // (op 951) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 951): v5

 // (op 951) BEGIN: Tile
// Allocated DataFlowTag610: 1 VGPR (Value: UInt32) (op 951): v5
v_lshl_add_u32 v5, v9, 2, v8 // (op 951) 
 // (op 951) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 951): v8

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 951): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(230)U32, 6:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(230)U32, 63:U32)U32}, 4:U32)U32)U32}(951) END
 // (op 1) Deallocate{}(1147) BEGIN
 // (op 1147) Deallocate 230
 // Deleting tag 230
 // (op 1) Deallocate{}(1147) END
 // (op 1) Assign VGPR 4:U32(846) BEGIN
 // (op 846) Assign dim(577) = 4:U32
 // (op 846) Assign stride expression
 // (op 1) Assign VGPR 4:U32(846) END
 // (op 1) Assign VGPR 4:U32(952) BEGIN
 // (op 952) Assign dim(611) = 4:U32
 // (op 952) Assign stride expression
 // (op 1) Assign VGPR 4:U32(952) END
 // (op 1) NOP(842) BEGIN
 // (op 842) generate({})
 // (op 842) end: generate({})
 // (op 1) NOP(842) END
 // (op 1) NOP(948) BEGIN
 // (op 948) generate({})
 // (op 948) end: generate({})
 // (op 1) NOP(948) END
 // (op 1) Assign VGPR 1:U32(850) BEGIN
 // (op 850) Assign dim(579) = 1:U32
 // (op 850) Assign stride expression
 // (op 1) Assign VGPR 1:U32(850) END
 // (op 1) Assign VGPR 1:U32(956) BEGIN
 // (op 956) Assign dim(613) = 1:U32
 // (op 956) Assign stride expression
 // (op 1) Assign VGPR 1:U32(956) END
 // (op 1) LoadVGPR Value: Float(2) BEGIN
 // tag 35: v**UNALLOCATED**
 // (op 2) GEN: LoadVGPR; scalar value
 // (op 1) LoadVGPR Value: Float(39) BEGIN
 // tag 44: v**UNALLOCATED**
 // (op 39) GEN: LoadVGPR; scalar value
// Allocated : 1 VGPR (Value: Float) (op 2): v8
v_mov_b32 v8, s14 // (op 2) Move value
 // (op 1) LoadVGPR Value: Float(2) END
// Allocated : 1 VGPR (Value: Float) (op 39): v9
v_mov_b32 v9, s15 // (op 39) Move value
 // (op 1) LoadVGPR Value: Float(39) END
 // (op 1) Deallocate{user_Float_Value_14}(1192) BEGIN
 // (op 1192) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1192) END
 // (op 1) Deallocate{user_Float_Value_12}(1194) BEGIN
 // (op 1194) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1194) END
 // (op 1) Scope(310) BEGIN
 // (op 310) Lock Scope 310
 // Lock Branch
 // (op 310) generate({315})
 // (op 310) NOP(315) BEGIN
 // (op 315) generate({})
 // (op 315) end: generate({})
 // (op 310) NOP(315) END
 // (op 310) Assign ACCVGPR 0.00000:S(59) BEGIN
 // (op 59) Assign dim(51) = 0.00000:S
 // tag 51: a**UNALLOCATED**
 // (op 59) Generate 0.00000:S into DataFlowTag51: ACCVGPR Value: Float x 4: (unallocated)
// Allocated DataFlowTag51: 4 ACCVGPRs (Value: Float) (op 59): a0, a1, a2, a3
v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
 // (op 310) Assign ACCVGPR 0.00000:S(59) END
 // (op 310) Scope(441) BEGIN
 // (op 441) Lock Scope 441
 // Lock Branch
 // (op 441) generate({436})
 // (op 441) NOP(436) BEGIN
 // (op 436) generate({})
 // (op 436) end: generate({})
 // (op 441) NOP(436) END
 // (op 441) Assign VGPR 96:U64(446) BEGIN
 // (op 446) Assign dim(458) = 96:U64
 // (op 446) Assign stride expression
 // (op 441) Assign VGPR 96:U64(446) END
 // (op 441) NOP(437) BEGIN
 // (op 437) generate({})
 // (op 437) end: generate({})
 // (op 441) NOP(437) END
 // (op 441) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(449) BEGIN
 // (op 449) Assign dim(459) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 459: v**UNALLOCATED**
 // (op 449) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag459: VGPR Value: UInt32 x 1: (unallocated)
 // (op 449) reg expression
 // (op 449) reg expression
 // (op 449) reg expression
 // (op 449) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 449): v10
v_lshrrev_b32 v10, 2, v1 // (op 449) 
 // (op 449) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 449): v11
v_lshl_add_u32 v11, s16, 5, v10 // (op 449) 
 // (op 449) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 449): v10

 // (op 449) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 449): v10
v_and_b32 v10, -4, v1 // (op 449) 
 // (op 449) BEGIN: Flatten[1]
 // (op 449) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 449): v12
v_sub_u32 v12, v1, v10 // (op 449) 
 // (op 449) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 449): v10

 // (op 449) BEGIN: TileTile
 // (op 449) {TileTile: ShiftL({Flatten[1]: v12:U32}, 5:U32)U32}
 // (op 449) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 449)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 449)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 449): v10
v_lshlrev_b32 v10, 5, v12 // (op 449) 
 // (op 449) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 449): v12

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 449): v12, v13
v_mov_b32 v13, 0 // (op 449) convert
v_mov_b32 v12, v10 // (op 449) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 449): v10

 // (op 449) BEGIN: Split
 // (op 449) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 449)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 449)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 449): v14, v15
v_mul_lo_u32 v15, v11, s3 // (op 449) most significant: low of low * high
 // (op 449) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 449): v10
v_mul_hi_u32 v10, v11, s2 // (op 449) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 449) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 449) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 449): v10

 // (op 449) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 449)      lhs (VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 449)      rhs (TileTile: VGPR Value: UInt64 x 1: v[12:13])
// Allocated : 2 SGPRs (Value: Bool64) (op 449): s0, s1
v_add_co_u32 v14, s[0:1], v14, v12 // (op 449) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 449) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 449): s0, s1

 // (op 449) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 449): v12, v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 449): v11

 // (op 449) Multiply({Split: v[14:15]:U64}, 6:U32)U64
 // (op 449) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 449)           lhs (Split: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 449)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 449) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 449): v10, v11
v_mul_lo_u32 v11, v15, 6 // (op 449) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 449): v12
v_mul_hi_u32 v12, v14, 6 // (op 449) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 449) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 449) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 449): v12

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 449): v14, v15

 // (op 449) LogicalShiftR(v[10:11]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 449): v12, v13
v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 449) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 449): v10, v11

// Allocated DataFlowTag459: 1 VGPR (Value: UInt32) (op 449): v10
v_mov_b32 v10, v12 // (op 449) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 449): v12, v13

 // (op 441) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(449) END
 // (op 441) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(450) BEGIN
 // (op 450) Assign dim(460) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 450) Assign stride expression
 // (op 441) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(450) END
 // (op 441) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(451) BEGIN
 // (op 451) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 453: s**UNALLOCATED**
 // (op 451) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: (unallocated)
 // (op 451) reg expression
 // (op 451) reg expression
 // (op 451) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 451) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 451)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 451)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 451) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 451): s0, s1
s_mul_i32 s1, s7, 6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 451) least significant
// Allocated : 1 SGPR (Value: Int32) (op 451): s14
s_mul_hi_u32 s14, s6, 6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 451) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 451): s14

 // (op 451) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 451) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 451)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 451)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 451): s14, s15
s_mul_i32 s15, s2, s1 // (op 451) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 451): s18
s_mul_i32 s18, s3, s0 // (op 451) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 451) least significant
// Allocated : 1 SGPR (Value: Int32) (op 451): s19
s_mul_hi_u32 s19, s2, s0 // (op 451) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 451) most significant: sum
s_add_u32 s15, s15, s19 // (op 451) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 451): s19

 // Freeing : 1 SGPR (Value: Int32) (op 451): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 451): s0, s1

 // (op 451) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 451): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 451) 
 // Freeing : 2 SGPRs (Value: Int64) (op 451): s14, s15

 // (op 451) reg expression
 // (op 451) reg expression
// Allocated DataFlowTag453: 4 SGPRs (Buffer: None) (op 451): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 451) reg expression
s_mov_b32 s41, s5 // (op 451) reg expression
s_mov_b32 s42, s0 // (op 451) reg expression
s_mov_b32 s43, 131072 // (op 451) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 451): s0, s1

 // (op 441) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(451) END
 // (op 441) NOP(438) BEGIN
 // (op 438) generate({})
 // (op 438) end: generate({})
 // (op 441) NOP(438) END
 // (op 441) Assign VGPR 0:U64(456) BEGIN
 // (op 456) Assign dim(462) = 0:U64
 // (op 456) Assign stride expression
 // (op 441) Assign VGPR 0:U64(456) END
 // (op 441) NOP(459) BEGIN
 // (op 459) generate({})
 // (op 459) end: generate({})
 // (op 441) NOP(459) END
 // (op 441) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(468) BEGIN
 // (op 468) Assign dim(463) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 463: v**UNALLOCATED**
 // (op 468) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag463: VGPR Value: UInt32 x 1: (unallocated)
 // (op 468) reg expression
 // (op 468) reg expression
 // (op 468) reg expression
 // (op 468) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 468): v11
v_lshrrev_b32 v11, 2, v1 // (op 468) 
 // (op 468) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 468): v12
v_lshl_add_u32 v12, s16, 5, v11 // (op 468) 
 // (op 468) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 468): v11

 // (op 468) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 468): v11
v_and_b32 v11, -4, v1 // (op 468) 
 // (op 468) BEGIN: Flatten[1]
 // (op 468) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 468): v13
v_sub_u32 v13, v1, v11 // (op 468) 
 // (op 468) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 468): v11

 // (op 468) BEGIN: TileTile
 // (op 468) {TileTile: ShiftL({Flatten[1]: v13:U32}, 5:U32)U32}
 // (op 468) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 468)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 468)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 468): v11
v_lshlrev_b32 v11, 5, v13 // (op 468) 
 // (op 468) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 468): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 468): v14, v15
v_mov_b32 v15, 0 // (op 468) convert
v_mov_b32 v14, v11 // (op 468) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 468): v11

 // (op 468) BEGIN: Split
 // (op 468) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 468)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 468)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 468): v16, v17
v_mul_lo_u32 v17, v12, s3 // (op 468) most significant: low of low * high
 // (op 468) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 468): v11
v_mul_hi_u32 v11, v12, s2 // (op 468) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 468) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 468) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 468): v11

 // (op 468) Add: dest (VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 468)      lhs (VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 468)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 468): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 468) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 468) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 468): s0, s1

 // (op 468) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 468): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 468): v12

 // (op 468) Multiply({Split: v[16:17]:U64}, 6:U32)U64
 // (op 468) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 468)           lhs (Split: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 468)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 468) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 468): v14, v15
v_mul_lo_u32 v15, v17, 6 // (op 468) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 468): v11
v_mul_hi_u32 v11, v16, 6 // (op 468) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 468) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 468) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 468): v11

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 468): v16, v17

 // (op 468) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 468): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 468) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 468): v14, v15

// Allocated DataFlowTag463: 1 VGPR (Value: UInt32) (op 468): v11
v_mov_b32 v11, v12 // (op 468) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 468): v12, v13

 // (op 441) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(468) END
 // (op 441) Assign VGPR 96:U64(469) BEGIN
 // (op 469) Assign dim(464) = 96:U64
 // (op 469) Assign stride expression
 // (op 441) Assign VGPR 96:U64(469) END
 // (op 441) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(470) BEGIN
 // (op 470) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 470) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // (op 470) reg expression
 // (op 470) reg expression
 // (op 470) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 470) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 470)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 470)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 470) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 470): s0, s1
s_mul_i32 s1, s7, 6 // (op 470) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 470) least significant
// Allocated : 1 SGPR (Value: Int32) (op 470): s14
s_mul_hi_u32 s14, s6, 6 // (op 470) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 470) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 470): s14

 // (op 470) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 470) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 470)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 470)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 470): s14, s15
s_mul_i32 s15, s2, s1 // (op 470) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 470): s18
s_mul_i32 s18, s3, s0 // (op 470) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 470) least significant
// Allocated : 1 SGPR (Value: Int32) (op 470): s19
s_mul_hi_u32 s19, s2, s0 // (op 470) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 470) most significant: sum
s_add_u32 s15, s15, s19 // (op 470) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 470): s19

 // Freeing : 1 SGPR (Value: Int32) (op 470): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 470): s0, s1

 // (op 470) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 470): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 470) 
 // Freeing : 2 SGPRs (Value: Int64) (op 470): s14, s15

 // (op 470) reg expression
 // (op 470) reg expression
s_mov_b32 s40, s4 // (op 470) reg expression
s_mov_b32 s41, s5 // (op 470) reg expression
s_mov_b32 s42, s0 // (op 470) reg expression
s_mov_b32 s43, 131072 // (op 470) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 470): s0, s1

 // (op 441) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(470) END
 // (op 441) NOP(461) BEGIN
 // (op 461) generate({})
 // (op 461) end: generate({})
 // (op 441) NOP(461) END
 // (op 441) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(475) BEGIN
 // (op 475) Assign dim(466) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 475) Assign stride expression
 // (op 441) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(475) END
 // (op 441) NOP(462) BEGIN
 // (op 462) generate({})
 // (op 462) end: generate({})
 // (op 441) NOP(462) END
 // (op 441) Assign VGPR 0:U64(478) BEGIN
 // (op 478) Assign dim(468) = 0:U64
 // (op 478) Assign stride expression
 // (op 441) Assign VGPR 0:U64(478) END
 // (op 441) NOP(503) BEGIN
 // (op 503) generate({})
 // (op 503) end: generate({})
 // (op 441) NOP(503) END
 // (op 441) Assign VGPR 96:U64(510) BEGIN
 // (op 510) Assign dim(477) = 96:U64
 // (op 510) Assign stride expression
 // (op 441) Assign VGPR 96:U64(510) END
 // (op 441) NOP(504) BEGIN
 // (op 504) generate({})
 // (op 504) end: generate({})
 // (op 441) NOP(504) END
 // (op 441) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(513) BEGIN
 // (op 513) Assign dim(478) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 478: v**UNALLOCATED**
 // (op 513) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag478: VGPR Value: UInt32 x 1: (unallocated)
 // (op 513) reg expression
 // (op 513) reg expression
 // (op 513) reg expression
 // (op 513) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 513): v12
v_and_b32 v12, -8, v1 // (op 513) 
 // (op 513) BEGIN: Flatten[1]
 // (op 513) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 513): v13
v_sub_u32 v13, v1, v12 // (op 513) 
 // (op 513) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 513): v12

 // (op 513) BEGIN: TileTile
 // (op 513) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 513) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 513)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 513)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 513): v12
v_lshlrev_b32 v12, 4, v13 // (op 513) 
 // (op 513) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 513): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 513): v14, v15
v_mov_b32 v15, 0 // (op 513) convert
v_mov_b32 v14, v12 // (op 513) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 513): v12

 // (op 513) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 513): v12
v_lshrrev_b32 v12, 3, v1 // (op 513) 
 // (op 513) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 513): v13
v_lshl_add_u32 v13, s17, 4, v12 // (op 513) 
 // (op 513) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 513): v12

 // (op 513) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 513) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 513)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 513)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 513): v16, v17
v_mul_lo_u32 v17, v13, s9 // (op 513) most significant: low of low * high
 // (op 513) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 513): v12
v_mul_hi_u32 v12, v13, s8 // (op 513) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 513) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 513) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 513): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 513): v13

 // (op 513) BEGIN: Split
 // (op 513) {Split: Add({TileTile: v[14:15]:U64}, v[16:17]:I64)U64}
 // (op 513) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 513)      lhs (TileTile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 513)      rhs (VGPR Value: Int64 x 1: v[16:17])
// Allocated : 2 VGPRs (Value: UInt64) (op 513): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 513): s0, s1
v_add_co_u32 v12, s[0:1], v14, v16 // (op 513) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 513) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 513): s0, s1

 // (op 513) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 513): v16, v17

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 513): v14, v15

 // (op 513) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 513) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 513)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 513)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 513) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 513): v14, v15
v_mul_lo_u32 v15, v13, 6 // (op 513) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 513): v16
v_mul_hi_u32 v16, v12, 6 // (op 513) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 513) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 513) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 513): v16

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 513): v12, v13

 // (op 513) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 513): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 513) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 513): v14, v15

// Allocated DataFlowTag478: 1 VGPR (Value: UInt32) (op 513): v14
v_mov_b32 v14, v12 // (op 513) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 513): v12, v13

 // (op 441) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(513) END
 // (op 441) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(514) BEGIN
 // (op 514) Assign dim(479) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 514) Assign stride expression
 // (op 441) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(514) END
 // (op 441) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(515) BEGIN
 // (op 515) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 472: s**UNALLOCATED**
 // (op 515) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: (unallocated)
 // (op 515) reg expression
 // (op 515) reg expression
 // (op 515) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 515) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 515)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 515)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 515) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 515): s0, s1
s_mul_i32 s1, s13, 6 // (op 515) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 515) least significant
// Allocated : 1 SGPR (Value: Int32) (op 515): s14
s_mul_hi_u32 s14, s12, 6 // (op 515) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 515) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 515): s14

 // (op 515) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 515) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 515)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 515)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 515): s14, s15
s_mul_i32 s15, s8, s1 // (op 515) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 515): s18
s_mul_i32 s18, s9, s0 // (op 515) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 515) least significant
// Allocated : 1 SGPR (Value: Int32) (op 515): s19
s_mul_hi_u32 s19, s8, s0 // (op 515) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 515) most significant: sum
s_add_u32 s15, s15, s19 // (op 515) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 515): s19

 // Freeing : 1 SGPR (Value: Int32) (op 515): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 515): s0, s1

 // (op 515) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 515): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 515) 
 // Freeing : 2 SGPRs (Value: Int64) (op 515): s14, s15

 // (op 515) reg expression
 // (op 515) reg expression
// Allocated DataFlowTag472: 4 SGPRs (Buffer: None) (op 515): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 515) reg expression
s_mov_b32 s45, s11 // (op 515) reg expression
s_mov_b32 s46, s0 // (op 515) reg expression
s_mov_b32 s47, 131072 // (op 515) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 515): s0, s1

 // (op 441) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(515) END
 // (op 441) NOP(505) BEGIN
 // (op 505) generate({})
 // (op 505) end: generate({})
 // (op 441) NOP(505) END
 // (op 441) Assign VGPR 0:U64(520) BEGIN
 // (op 520) Assign dim(481) = 0:U64
 // (op 520) Assign stride expression
 // (op 441) Assign VGPR 0:U64(520) END
 // (op 441) NOP(523) BEGIN
 // (op 523) generate({})
 // (op 523) end: generate({})
 // (op 441) NOP(523) END
 // (op 441) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(532) BEGIN
 // (op 532) Assign dim(482) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 482: v**UNALLOCATED**
 // (op 532) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag482: VGPR Value: UInt32 x 1: (unallocated)
 // (op 532) reg expression
 // (op 532) reg expression
 // (op 532) reg expression
 // (op 532) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 532): v12
v_and_b32 v12, -8, v1 // (op 532) 
 // (op 532) BEGIN: Flatten[1]
 // (op 532) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 532): v13
v_sub_u32 v13, v1, v12 // (op 532) 
 // (op 532) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 532): v12

 // (op 532) BEGIN: TileTile
 // (op 532) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 532) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 532)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 532)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 532): v12
v_lshlrev_b32 v12, 4, v13 // (op 532) 
 // (op 532) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 532): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 532): v16, v17
v_mov_b32 v17, 0 // (op 532) convert
v_mov_b32 v16, v12 // (op 532) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 532): v12

 // (op 532) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 532): v15
v_lshrrev_b32 v15, 3, v1 // (op 532) 
 // (op 532) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 532): v12
v_lshl_add_u32 v12, s17, 4, v15 // (op 532) 
 // (op 532) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 532): v15

 // (op 532) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 532) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 532)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 532)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 532): v18, v19
v_mul_lo_u32 v19, v12, s9 // (op 532) most significant: low of low * high
 // (op 532) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 532): v13
v_mul_hi_u32 v13, v12, s8 // (op 532) most significant: high of low * low
v_mul_lo_u32 v18, v12, s8 // (op 532) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 532) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 532): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 532): v12

 // (op 532) BEGIN: Split
 // (op 532) {Split: Add({TileTile: v[16:17]:U64}, v[18:19]:I64)U64}
 // (op 532) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 532)      lhs (TileTile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 532)      rhs (VGPR Value: Int64 x 1: v[18:19])
// Allocated : 2 VGPRs (Value: UInt64) (op 532): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 532): s0, s1
v_add_co_u32 v12, s[0:1], v16, v18 // (op 532) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 532) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 532): s0, s1

 // (op 532) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 532): v18, v19

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 532): v16, v17

 // (op 532) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 532) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 532)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 532)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 532) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 532): v16, v17
v_mul_lo_u32 v17, v13, 6 // (op 532) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 532): v15
v_mul_hi_u32 v15, v12, 6 // (op 532) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 532) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 532) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 532): v15

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 532): v12, v13

 // (op 532) LogicalShiftR(v[16:17]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 532): v12, v13
v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 532) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 532): v16, v17

// Allocated DataFlowTag482: 1 VGPR (Value: UInt32) (op 532): v15
v_mov_b32 v15, v12 // (op 532) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 532): v12, v13

 // (op 441) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(532) END
 // (op 441) Assign VGPR 96:U64(533) BEGIN
 // (op 533) Assign dim(483) = 96:U64
 // (op 533) Assign stride expression
 // (op 441) Assign VGPR 96:U64(533) END
 // (op 441) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(534) BEGIN
 // (op 534) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 534) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // (op 534) reg expression
 // (op 534) reg expression
 // (op 534) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 534) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 534)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 534)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 534) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 534): s0, s1
s_mul_i32 s1, s13, 6 // (op 534) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 534) least significant
// Allocated : 1 SGPR (Value: Int32) (op 534): s14
s_mul_hi_u32 s14, s12, 6 // (op 534) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 534) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 534): s14

 // (op 534) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 534) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 534)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 534)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 534): s14, s15
s_mul_i32 s15, s8, s1 // (op 534) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 534): s18
s_mul_i32 s18, s9, s0 // (op 534) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 534) least significant
// Allocated : 1 SGPR (Value: Int32) (op 534): s19
s_mul_hi_u32 s19, s8, s0 // (op 534) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 534) most significant: sum
s_add_u32 s15, s15, s19 // (op 534) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 534): s19

 // Freeing : 1 SGPR (Value: Int32) (op 534): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 534): s0, s1

 // (op 534) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 534): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 534) 
 // Freeing : 2 SGPRs (Value: Int64) (op 534): s14, s15

 // (op 534) reg expression
 // (op 534) reg expression
s_mov_b32 s44, s10 // (op 534) reg expression
s_mov_b32 s45, s11 // (op 534) reg expression
s_mov_b32 s46, s0 // (op 534) reg expression
s_mov_b32 s47, 131072 // (op 534) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 534): s0, s1

 // (op 441) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(534) END
 // (op 441) NOP(525) BEGIN
 // (op 525) generate({})
 // (op 525) end: generate({})
 // (op 441) NOP(525) END
 // (op 441) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(539) BEGIN
 // (op 539) Assign dim(485) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 539) Assign stride expression
 // (op 441) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(539) END
 // (op 441) NOP(526) BEGIN
 // (op 526) generate({})
 // (op 526) end: generate({})
 // (op 441) NOP(526) END
 // (op 441) Assign VGPR 0:U64(542) BEGIN
 // (op 542) Assign dim(487) = 0:U64
 // (op 542) Assign stride expression
 // (op 441) Assign VGPR 0:U64(542) END
 // (op 441) NOP(567) BEGIN
 // (op 567) generate({})
 // (op 567) end: generate({})
 // (op 441) NOP(567) END
 // (op 441) Assign VGPR 4:U64(574) BEGIN
 // (op 574) Assign dim(496) = 4:U64
 // (op 574) Assign stride expression
 // (op 441) Assign VGPR 4:U64(574) END
 // (op 441) NOP(568) BEGIN
 // (op 568) generate({})
 // (op 568) end: generate({})
 // (op 441) NOP(568) END
 // (op 441) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32}(577) BEGIN
 // (op 577) Assign dim(497) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32}
 // tag 497: v**UNALLOCATED**
 // (op 577) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32} into DataFlowTag497: VGPR Value: UInt32 x 1: (unallocated)
 // (op 577) reg expression
 // (op 577) reg expression
 // (op 577) reg expression
 // (op 577) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 577) convert
// Allocated : 1 VGPR (Value: UInt32) (op 577): v12
v_lshrrev_b32 v12, 2, v1 // (op 577) 
 // (op 577) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 577): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 577) 
 // (op 577) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 577): v12

 // (op 577) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 577): v12
v_and_b32 v12, -4, v1 // (op 577) 
 // (op 577) BEGIN: Flatten[1]TileTile
 // (op 577) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 577): v16
v_sub_u32 v16, v1, v12 // (op 577) 
 // (op 577) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 577): v12

 // (op 577) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag497: 1 VGPR (Value: UInt32) (op 577): v12
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v12, v13, s20 // (op 577) 
v_add_u32 v12, v12, v16 // (op 577) 
 // (op 577) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 577): v16

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 577): v13

 // (op 441) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32}(577) END
 // (op 441) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(578) BEGIN
 // (op 578) Assign dim(498) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 578) Assign stride expression
 // (op 441) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(578) END
 // (op 441) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(579) BEGIN
 // (op 579) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 491: s**UNALLOCATED**
 // (op 579) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: (unallocated)
 // (op 579) reg expression
 // (op 579) reg expression
 // (op 579) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 579) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 579)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 579)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 579): s0, s1
s_mul_i32 s1, s20, s7 // (op 579) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 579): s14
s_mul_i32 s14, s21, s6 // (op 579) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 579) least significant
// Allocated : 1 SGPR (Value: Int32) (op 579): s15
s_mul_hi_u32 s15, s20, s6 // (op 579) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 579) most significant: sum
s_add_u32 s1, s1, s15 // (op 579) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 579): s15

 // Freeing : 1 SGPR (Value: Int32) (op 579): s14

 // (op 579) reg expression
 // (op 579) reg expression
// Allocated DataFlowTag491: 4 SGPRs (Buffer: None) (op 579): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 579) reg expression
s_mov_b32 s49, s23 // (op 579) reg expression
s_mov_b32 s50, s0 // (op 579) reg expression
s_mov_b32 s51, 131072 // (op 579) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 579): s0, s1

 // (op 441) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(579) END
 // (op 441) NOP(569) BEGIN
 // (op 569) generate({})
 // (op 569) end: generate({})
 // (op 441) NOP(569) END
 // (op 441) Assign VGPR 1:U64(584) BEGIN
 // (op 584) Assign dim(500) = 1:U64
 // (op 584) Assign stride expression
 // (op 441) Assign VGPR 1:U64(584) END
 // (op 441) NOP(587) BEGIN
 // (op 587) generate({})
 // (op 587) end: generate({})
 // (op 441) NOP(587) END
 // (op 441) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32}(596) BEGIN
 // (op 596) Assign dim(501) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32}
 // tag 501: v**UNALLOCATED**
 // (op 596) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32} into DataFlowTag501: VGPR Value: UInt32 x 1: (unallocated)
 // (op 596) reg expression
 // (op 596) reg expression
 // (op 596) reg expression
 // (op 596) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 596) convert
// Allocated : 1 VGPR (Value: UInt32) (op 596): v13
v_lshrrev_b32 v13, 2, v1 // (op 596) 
 // (op 596) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 596): v16
v_lshl_add_u32 v16, s16, 5, v13 // (op 596) 
 // (op 596) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 596): v13

 // (op 596) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 596): v13
v_and_b32 v13, -4, v1 // (op 596) 
 // (op 596) BEGIN: Flatten[1]TileTile
 // (op 596) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 596): v17
v_sub_u32 v17, v1, v13 // (op 596) 
 // (op 596) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 596): v13

 // (op 596) BEGIN: Split
// Allocated DataFlowTag501: 1 VGPR (Value: UInt32) (op 596): v13
v_mul_lo_u32 v13, v16, s20 // (op 596) 
v_add_u32 v13, v13, v17 // (op 596) 
 // (op 596) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 596): v17

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 596): v16

 // (op 441) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32}(596) END
 // (op 441) Assign VGPR 4:U64(597) BEGIN
 // (op 597) Assign dim(502) = 4:U64
 // (op 597) Assign stride expression
 // (op 441) Assign VGPR 4:U64(597) END
 // (op 441) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(598) BEGIN
 // (op 598) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 598) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // (op 598) reg expression
 // (op 598) reg expression
 // (op 598) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 598) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 598)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 598)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 598): s0, s1
s_mul_i32 s1, s20, s7 // (op 598) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 598): s14
s_mul_i32 s14, s21, s6 // (op 598) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 598) least significant
// Allocated : 1 SGPR (Value: Int32) (op 598): s15
s_mul_hi_u32 s15, s20, s6 // (op 598) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 598) most significant: sum
s_add_u32 s1, s1, s15 // (op 598) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 598): s15

 // Freeing : 1 SGPR (Value: Int32) (op 598): s14

 // (op 598) reg expression
 // (op 598) reg expression
s_mov_b32 s48, s22 // (op 598) reg expression
s_mov_b32 s49, s23 // (op 598) reg expression
s_mov_b32 s50, s0 // (op 598) reg expression
s_mov_b32 s51, 131072 // (op 598) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 598): s0, s1

 // (op 441) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(598) END
 // (op 441) NOP(589) BEGIN
 // (op 589) generate({})
 // (op 589) end: generate({})
 // (op 441) NOP(589) END
 // (op 441) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(603) BEGIN
 // (op 603) Assign dim(504) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 603) Assign stride expression
 // (op 441) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(603) END
 // (op 441) NOP(590) BEGIN
 // (op 590) generate({})
 // (op 590) end: generate({})
 // (op 441) NOP(590) END
 // (op 441) Assign VGPR 1:U64(606) BEGIN
 // (op 606) Assign dim(506) = 1:U64
 // (op 606) Assign stride expression
 // (op 441) Assign VGPR 1:U64(606) END
 // (op 441) NOP(631) BEGIN
 // (op 631) generate({})
 // (op 631) end: generate({})
 // (op 441) NOP(631) END
 // (op 441) Assign VGPR 4:U64(638) BEGIN
 // (op 638) Assign dim(515) = 4:U64
 // (op 638) Assign stride expression
 // (op 441) Assign VGPR 4:U64(638) END
 // (op 441) NOP(632) BEGIN
 // (op 632) generate({})
 // (op 632) end: generate({})
 // (op 441) NOP(632) END
 // (op 441) Assign VGPR {Split: Add(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(641) BEGIN
 // (op 641) Assign dim(516) = {Split: Add(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 516: v**UNALLOCATED**
 // (op 641) Generate {Split: Add(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag516: VGPR Value: UInt32 x 1: (unallocated)
 // (op 641) reg expression
 // (op 641) reg expression
 // (op 641) reg expression
 // (op 641) BEGIN: Flatten[1]Tile
 // (op 641) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 641) convert
// Allocated : 1 VGPR (Value: UInt32) (op 641): v16
v_and_b32 v16, 15, v1 // (op 641) 
 // (op 641) END: Flatten[1]Tile
 // (op 641) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 641): v17
v_lshl_add_u32 v17, s17, 4, v16 // (op 641) 
 // (op 641) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 641): v16

 // (op 641) Multiply({Tile: v17:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 641): v16
v_mul_lo_u32 v16, v17, s24 // (op 641) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 641): v17

 // (op 641) BEGIN: Flatten[1]
 // (op 641) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 641): v17
v_and_b32 v17, 63, v1 // (op 641) 
 // (op 641) END: Flatten[1]
 // (op 641) LogicalShiftR({Flatten[1]: v17:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 641): v18
v_lshrrev_b32 v18, 4, v17 // (op 641) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 641): v17

 // (op 641) BEGIN: Split
 // (op 641) {Split: Add(v18:U32, v16:U32)U32}
// Allocated DataFlowTag516: 1 VGPR (Value: UInt32) (op 641): v17
v_add_u32 v17, v18, v16 // (op 641) 
 // (op 641) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 641): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 641): v18

 // (op 441) Assign VGPR {Split: Add(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(641) END
 // (op 441) Assign VGPR 4:U64(642) BEGIN
 // (op 642) Assign dim(517) = 4:U64
 // (op 642) Assign stride expression
 // (op 441) Assign VGPR 4:U64(642) END
 // (op 441) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(643) BEGIN
 // (op 643) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 510: s**UNALLOCATED**
 // (op 643) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: (unallocated)
 // (op 643) reg expression
 // (op 643) reg expression
 // (op 643) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 643) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 643)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 643)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 643): s0, s1
s_mul_i32 s1, s24, s13 // (op 643) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 643): s14
s_mul_i32 s14, s25, s12 // (op 643) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 643) least significant
// Allocated : 1 SGPR (Value: Int32) (op 643): s15
s_mul_hi_u32 s15, s24, s12 // (op 643) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 643) most significant: sum
s_add_u32 s1, s1, s15 // (op 643) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 643): s15

 // Freeing : 1 SGPR (Value: Int32) (op 643): s14

 // (op 643) reg expression
 // (op 643) reg expression
// Allocated DataFlowTag510: 4 SGPRs (Buffer: None) (op 643): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 643) reg expression
s_mov_b32 s53, s27 // (op 643) reg expression
s_mov_b32 s54, s0 // (op 643) reg expression
s_mov_b32 s55, 131072 // (op 643) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 643): s0, s1

 // (op 441) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(643) END
 // (op 441) NOP(633) BEGIN
 // (op 633) generate({})
 // (op 633) end: generate({})
 // (op 441) NOP(633) END
 // (op 441) Assign VGPR 1:U64(648) BEGIN
 // (op 648) Assign dim(519) = 1:U64
 // (op 648) Assign stride expression
 // (op 441) Assign VGPR 1:U64(648) END
 // (op 441) NOP(651) BEGIN
 // (op 651) generate({})
 // (op 651) end: generate({})
 // (op 441) NOP(651) END
 // (op 441) Assign VGPR {Split: Add(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(660) BEGIN
 // (op 660) Assign dim(520) = {Split: Add(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 520: v**UNALLOCATED**
 // (op 660) Generate {Split: Add(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag520: VGPR Value: UInt32 x 1: (unallocated)
 // (op 660) reg expression
 // (op 660) reg expression
 // (op 660) reg expression
 // (op 660) BEGIN: Flatten[1]Tile
 // (op 660) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 660) convert
// Allocated : 1 VGPR (Value: UInt32) (op 660): v16
v_and_b32 v16, 15, v1 // (op 660) 
 // (op 660) END: Flatten[1]Tile
 // (op 660) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 660): v18
v_lshl_add_u32 v18, s17, 4, v16 // (op 660) 
 // (op 660) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 660): v16

 // (op 660) Multiply({Tile: v18:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 660): v16
v_mul_lo_u32 v16, v18, s24 // (op 660) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 660): v18

 // (op 660) BEGIN: Flatten[1]
 // (op 660) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 660): v18
v_and_b32 v18, 63, v1 // (op 660) 
 // (op 660) END: Flatten[1]
 // (op 660) LogicalShiftR({Flatten[1]: v18:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 660): v19
v_lshrrev_b32 v19, 4, v18 // (op 660) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 660): v18

 // (op 660) BEGIN: Split
 // (op 660) {Split: Add(v19:U32, v16:U32)U32}
// Allocated DataFlowTag520: 1 VGPR (Value: UInt32) (op 660): v18
v_add_u32 v18, v19, v16 // (op 660) 
 // (op 660) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 660): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 660): v19

 // (op 441) Assign VGPR {Split: Add(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(660) END
 // (op 441) Assign VGPR 4:U64(661) BEGIN
 // (op 661) Assign dim(521) = 4:U64
 // (op 661) Assign stride expression
 // (op 441) Assign VGPR 4:U64(661) END
 // (op 441) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(662) BEGIN
 // (op 662) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 662) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // (op 662) reg expression
 // (op 662) reg expression
 // (op 662) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 662) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 662)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 662)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 662): s0, s1
s_mul_i32 s1, s24, s13 // (op 662) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 662): s14
s_mul_i32 s14, s25, s12 // (op 662) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 662) least significant
// Allocated : 1 SGPR (Value: Int32) (op 662): s15
s_mul_hi_u32 s15, s24, s12 // (op 662) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 662) most significant: sum
s_add_u32 s1, s1, s15 // (op 662) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 662): s15

 // Freeing : 1 SGPR (Value: Int32) (op 662): s14

 // (op 662) reg expression
 // (op 662) reg expression
s_mov_b32 s52, s26 // (op 662) reg expression
s_mov_b32 s53, s27 // (op 662) reg expression
s_mov_b32 s54, s0 // (op 662) reg expression
s_mov_b32 s55, 131072 // (op 662) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 662): s0, s1

 // (op 441) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(662) END
 // (op 441) NOP(653) BEGIN
 // (op 653) generate({})
 // (op 653) end: generate({})
 // (op 441) NOP(653) END
 // (op 441) Assign VGPR 4:U64(667) BEGIN
 // (op 667) Assign dim(523) = 4:U64
 // (op 667) Assign stride expression
 // (op 441) Assign VGPR 4:U64(667) END
 // (op 441) NOP(654) BEGIN
 // (op 654) generate({})
 // (op 654) end: generate({})
 // (op 441) NOP(654) END
 // (op 441) Assign VGPR 1:U64(670) BEGIN
 // (op 670) Assign dim(525) = 1:U64
 // (op 670) Assign stride expression
 // (op 441) Assign VGPR 1:U64(670) END
 // (op 441) NOP(330) BEGIN
 // (op 330) generate({})
 // (op 330) end: generate({})
 // (op 441) NOP(330) END
 // (op 441) SetCoordinate(335) BEGIN
 // (op 335) generate({})
 // (op 335) end: generate({})
 // (op 335) generate({336})
 // (op 335) SetCoordinate(336) BEGIN
 // (op 336) generate({})
 // (op 336) end: generate({})
 // (op 336) generate({337})
 // (op 336) LoadTiled Value: E8M0(337) BEGIN
 // (op 337) GEN: loadMacroTileWAVE OP337 WaveTile 261
 // (op 337) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 337) Generate 1:U64 into nullptr
 // tag 49: v**UNALLOCATED**
 // (op 337) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 337) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag516: VGPR Value: UInt32 x 1: v17
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 337) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 337): v16
buffer_load_ubyte v16, v17, s[52:55], 0 offen offset: 0 // (op 337) Load value
// VMEM: Expected complete at 807 (current 327)
 // (op 336) LoadTiled Value: E8M0(337) END
 // (op 336) end: generate({337})
 // (op 335) SetCoordinate(336) END
 // (op 335) end: generate({336})
 // (op 441) SetCoordinate(335) END
 // (op 441) SetCoordinate(340) BEGIN
 // (op 340) generate({})
 // (op 340) end: generate({})
 // (op 340) generate({341})
 // (op 340) LoadTiled Value: FP6x16(341) BEGIN
 // (op 341) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 341) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 341) Generate 0:U64 into nullptr
 // tag 9: v**UNALLOCATED**
 // (op 341) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 341) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag478: VGPR Value: UInt32 x 1: v14
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 341) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: FP6x16) (op 341): v20, v21, v22
buffer_load_dwordx3 v[20:22], v14, s[44:47], 0 offen // (op 341) Load value
// VMEM: Expected complete at 808 (current 328)
 // (op 340) LoadTiled Value: FP6x16(341) END
 // (op 340) end: generate({341})
 // (op 441) SetCoordinate(340) END
 // (op 441) SetCoordinate(343) BEGIN
 // (op 343) generate({})
 // (op 343) end: generate({})
 // (op 343) generate({344})
 // (op 343) LoadTiled Value: FP6x16(344) BEGIN
 // (op 344) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 344) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 344) Generate 0:U64 into nullptr
 // tag 8: v**UNALLOCATED**
 // (op 344) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 344) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag459: VGPR Value: UInt32 x 1: v10
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
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 344) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 344): v24, v25, v26, v27, v28, v29
buffer_load_dwordx4 v[24:27], v10, s[40:43], 0 offen // (op 344) Load value
// VMEM: Expected complete at 809 (current 329)
buffer_load_dwordx2 v[28:29], v10, s[40:43], 0 offen offset:16 // (op 344) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1288 (current 808)
 // (op 343) LoadTiled Value: FP6x16(344) END
 // (op 343) end: generate({344})
 // (op 441) SetCoordinate(343) END
 // (op 441) SetCoordinate(346) BEGIN
 // (op 346) generate({})
 // (op 346) end: generate({})
 // (op 346) generate({347})
 // (op 346) LoadTiled Value: E8M0(347) BEGIN
 // (op 347) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 347) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 347) Generate 1:U64 into nullptr
 // tag 10: v**UNALLOCATED**
 // (op 347) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 347) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag497: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 347) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 347): v19
buffer_load_ubyte v19, v12, s[48:51], 0 offen offset: 0 // (op 347) Load value
// VMEM: Expected complete at 1289 (current 809)
 // (op 346) LoadTiled Value: E8M0(347) END
 // (op 346) end: generate({347})
 // (op 441) SetCoordinate(346) END
 // (op 441) SetCoordinate(349) BEGIN
 // (op 349) generate({})
 // (op 349) end: generate({})
 // (op 349) generate({350})
 // (op 349) SetCoordinate(350) BEGIN
 // (op 350) generate({})
 // (op 350) end: generate({})
 // (op 350) generate({351})
 // (op 350) LoadTiled Value: E8M0(351) BEGIN
 // (op 351) GEN: loadMacroTileWAVE OP351 WaveTile 261
 // (op 351) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 351) Generate 1:U64 into nullptr
 // tag 448: v**UNALLOCATED**
 // (op 351) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 351) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag516: VGPR Value: UInt32 x 1: v17
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 351) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 351): v23
buffer_load_ubyte v23, v17, s[52:55], 0 offen offset:4 // (op 351) Load value
// VMEM: Expected complete at 1290 (current 810)
 // (op 350) LoadTiled Value: E8M0(351) END
 // (op 350) end: generate({351})
 // (op 349) SetCoordinate(350) END
 // (op 349) end: generate({350})
 // (op 441) SetCoordinate(349) END
 // (op 441) Deallocate{}(1125) BEGIN
 // (op 1125) Deallocate 519
 // Deleting tag 519
 // (op 1125) Deallocate 516
 // Freeing DataFlowTag516: 1 VGPR (Value: UInt32) (op 641): v17

 // Deleting tag 516
 // (op 1125) Deallocate 518
 // Deleting tag 518
 // (op 1125) Deallocate 515
 // Deleting tag 515
 // (op 1125) Deallocate 517
 // Deleting tag 517
 // (op 441) Deallocate{}(1125) END
 // (op 441) SetCoordinate(354) BEGIN
 // (op 354) generate({})
 // (op 354) end: generate({})
 // (op 354) generate({355})
 // (op 354) LoadTiled Value: FP6x16(355) BEGIN
 // (op 355) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 355) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 355) Generate 0:U64 into nullptr
 // tag 434: v**UNALLOCATED**
 // (op 355) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 355) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag478: VGPR Value: UInt32 x 1: v14
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 355) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: FP6x16) (op 355): v30, v31, v32
buffer_load_dwordx3 v[30:32], v14, s[44:47], 0 offen offset:96 // (op 355) Load value
// VMEM: Expected stall of 478. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1769 (current 1289)
 // (op 354) LoadTiled Value: FP6x16(355) END
 // (op 354) end: generate({355})
 // (op 441) SetCoordinate(354) END
 // (op 441) Deallocate{}(1127) BEGIN
 // (op 1127) Deallocate 478
 // Freeing DataFlowTag478: 1 VGPR (Value: UInt32) (op 513): v14

 // Deleting tag 478
 // (op 1127) Deallocate 481
 // Deleting tag 481
 // (op 1127) Deallocate 477
 // Deleting tag 477
 // (op 1127) Deallocate 479
 // Deleting tag 479
 // (op 1127) Deallocate 480
 // Deleting tag 480
 // (op 441) Deallocate{}(1127) END
 // (op 441) SetCoordinate(357) BEGIN
 // (op 357) generate({})
 // (op 357) end: generate({})
 // (op 357) generate({358})
 // (op 357) LoadTiled Value: FP6x16(358) BEGIN
 // (op 358) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 358) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 358) Generate 0:U64 into nullptr
 // tag 430: v**UNALLOCATED**
 // (op 358) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 358) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag459: VGPR Value: UInt32 x 1: v10
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
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 358) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 358): v34, v35, v36, v37, v38, v39
buffer_load_dwordx4 v[34:37], v10, s[40:43], 0 offen offset:96 // (op 358) Load value
// VMEM: Expected complete at 1770 (current 1290)
buffer_load_dwordx2 v[38:39], v10, s[40:43], 0 offen offset:112 // (op 358) Load value
// VMEM: Expected complete at 1771 (current 1291)
 // (op 357) LoadTiled Value: FP6x16(358) END
 // (op 357) end: generate({358})
 // (op 441) SetCoordinate(357) END
 // (op 441) Deallocate{}(1129) BEGIN
 // (op 1129) Deallocate 462
 // Deleting tag 462
 // (op 1129) Deallocate 458
 // Deleting tag 458
 // (op 1129) Deallocate 460
 // Deleting tag 460
 // (op 1129) Deallocate 459
 // Freeing DataFlowTag459: 1 VGPR (Value: UInt32) (op 449): v10

 // Deleting tag 459
 // (op 1129) Deallocate 461
 // Deleting tag 461
 // (op 441) Deallocate{}(1129) END
 // (op 441) SetCoordinate(360) BEGIN
 // (op 360) generate({})
 // (op 360) end: generate({})
 // (op 360) generate({361})
 // (op 360) LoadTiled Value: E8M0(361) BEGIN
 // (op 361) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 361) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 361) Generate 1:U64 into nullptr
 // tag 438: v**UNALLOCATED**
 // (op 361) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 361) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag497: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 361) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 361): v10
buffer_load_ubyte v10, v12, s[48:51], 0 offen offset:4 // (op 361) Load value
// VMEM: Expected stall of 478. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2250 (current 1770)
 // (op 360) LoadTiled Value: E8M0(361) END
 // (op 360) end: generate({361})
 // (op 441) SetCoordinate(360) END
 // (op 441) Deallocate{}(1131) BEGIN
 // (op 1131) Deallocate 496
 // Deleting tag 496
 // (op 1131) Deallocate 497
 // Freeing DataFlowTag497: 1 VGPR (Value: UInt32) (op 577): v12

 // Deleting tag 497
 // (op 1131) Deallocate 498
 // Deleting tag 498
 // (op 1131) Deallocate 500
 // Deleting tag 500
 // (op 1131) Deallocate 499
 // Deleting tag 499
 // (op 441) Deallocate{}(1131) END
 // (op 441) Scope(775) BEGIN
 // (op 775) Lock Scope 775
 // Lock Branch
 // (op 775) generate({772})
 // (op 775) NOP(772) BEGIN
 // (op 772) generate({})
 // (op 772) end: generate({})
 // (op 775) NOP(772) END
 // (op 775) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(780) BEGIN
 // (op 780) Assign dim(556) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 556: v**UNALLOCATED**
 // (op 780) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag556: VGPR Value: UInt32 x 1: (unallocated)
 // (op 780) reg expression
 // (op 780) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 780): v12
v_and_b32 v12, -8, v1 // (op 780) 
 // (op 780) BEGIN: Tile[1]
 // (op 780) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 780): v14
v_sub_u32 v14, v1, v12 // (op 780) 
 // (op 780) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 780): v12

 // (op 780) BEGIN: Flatten
 // (op 780) {Flatten: ShiftL({Tile[1]: v14:U32}, 4:U32)U32}
 // (op 780) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 780)         value (Tile[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 780)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 780): v12
v_lshlrev_b32 v12, 4, v14 // (op 780) 
 // (op 780) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 780): v14

 // (op 780) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 780): v14
v_lshrrev_b32 v14, 3, v1 // (op 780) 
 // (op 780) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 780): v17
v_lshl_add_u32 v17, v14, 7, v12 // (op 780) 
 // (op 780) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 780): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 780): v14

 // (op 780) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 780): v12
v_mul_lo_u32 v12, v17, 6 // (op 780) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 780): v17

 // (op 780) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag556: 1 VGPR (Value: UInt32) (op 780): v14
v_lshrrev_b32 v14, 3, v12 // (op 780) 
 // Freeing : 1 VGPR (Value: UInt32) (op 780): v12

 // (op 775) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(780) END
 // (op 775) Assign VGPR 96:U32(781) BEGIN
 // (op 781) Assign dim(557) = 96:U32
 // (op 781) Assign stride expression
 // (op 775) Assign VGPR 96:U32(781) END
 // (op 775) NOP(773) BEGIN
 // (op 773) generate({})
 // (op 773) end: generate({})
 // (op 775) NOP(773) END
 // (op 775) Assign VGPR 0:U32(785) BEGIN
 // (op 785) Assign dim(559) = 0:U32
 // (op 785) Assign stride expression
 // (op 775) Assign VGPR 0:U32(785) END
 // (op 775) StoreLDSTile Value: FP6x16(363) BEGIN
 // (op 363) GEN: StoreLDSTile
 // (op 363) GEN: storeMacroTileLDS OP 363 LDS 24 MacroTile 9
 // (op 363) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 363) Generate 0:U32 into nullptr
 // (op 363) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[20:22]
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

 // (op 363) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[20:22]
 // 	info.rowOffsetReg = DataFlowTag556: VGPR Value: UInt32 x 1: v14
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
 // (op 363) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(8)
ds_write_b96 v14, v[20:22] // (op 363) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 1874 (current 1779)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 775) StoreLDSTile Value: FP6x16(363) END
 // (op 775) Deallocate{}(1133) BEGIN
 // (op 1133) Deallocate 559
 // Deleting tag 559
 // (op 1133) Deallocate 557
 // Deleting tag 557
 // (op 1133) Deallocate 556
 // Freeing DataFlowTag556: 1 VGPR (Value: UInt32) (op 780): v14

 // Deleting tag 556
 // (op 1133) Deallocate 558
 // Deleting tag 558
 // (op 775) Deallocate{}(1133) END
 // (op 775) end: generate({772})
 // (op 775) Unlock Scope 775
 // Unlock None
 // (op 441) Scope(775) END
 // (op 441) Scope(722) BEGIN
 // (op 722) Lock Scope 722
 // Lock Branch
 // (op 722) generate({719})
 // (op 722) NOP(719) BEGIN
 // (op 719) generate({})
 // (op 719) end: generate({})
 // (op 722) NOP(719) END
 // (op 722) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(727) BEGIN
 // (op 727) Assign dim(540) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 540: v**UNALLOCATED**
 // (op 727) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag540: VGPR Value: UInt32 x 1: (unallocated)
 // (op 727) reg expression
 // (op 727) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 727): v12
v_and_b32 v12, -4, v1 // (op 727) 
 // (op 727) BEGIN: Tile[1]
 // (op 727) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 727): v14
v_sub_u32 v14, v1, v12 // (op 727) 
 // (op 727) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 727): v12

 // (op 727) BEGIN: Flatten
 // (op 727) {Flatten: ShiftL({Tile[1]: v14:U32}, 5:U32)U32}
 // (op 727) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 727)         value (Tile[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 727)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 727): v12
v_lshlrev_b32 v12, 5, v14 // (op 727) 
 // (op 727) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 727): v14

 // (op 727) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 727): v14
v_lshrrev_b32 v14, 2, v1 // (op 727) 
 // (op 727) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 727): v17
v_lshl_add_u32 v17, v14, 7, v12 // (op 727) 
 // (op 727) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 727): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 727): v14

 // (op 727) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 727): v12
v_mul_lo_u32 v12, v17, 6 // (op 727) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 727): v17

 // (op 727) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag540: 1 VGPR (Value: UInt32) (op 727): v14
v_lshrrev_b32 v14, 3, v12 // (op 727) 
 // Freeing : 1 VGPR (Value: UInt32) (op 727): v12

 // (op 722) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(727) END
 // (op 722) Assign VGPR 96:U32(728) BEGIN
 // (op 728) Assign dim(541) = 96:U32
 // (op 728) Assign stride expression
 // (op 722) Assign VGPR 96:U32(728) END
 // (op 722) NOP(720) BEGIN
 // (op 720) generate({})
 // (op 720) end: generate({})
 // (op 722) NOP(720) END
 // (op 722) Assign VGPR 0:U32(732) BEGIN
 // (op 732) Assign dim(543) = 0:U32
 // (op 732) Assign stride expression
 // (op 722) Assign VGPR 0:U32(732) END
 // (op 722) StoreLDSTile Value: FP6x16(364) BEGIN
 // (op 364) GEN: StoreLDSTile
 // (op 364) GEN: storeMacroTileLDS OP 364 LDS 16 MacroTile 8
 // (op 364) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 364) Generate 0:U32 into nullptr
 // (op 364) 	Dir = Store
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 364) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag540: VGPR Value: UInt32 x 1: v14
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 364) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(7)
ds_write_b128 v14, v[24:27] offset:1536 // (op 364) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 1883 (current 1788)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
s_waitcnt vmcnt(6)
ds_write_b64 v14, v[28:29] offset:1552 // (op 364) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 1885 (current 1790)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 722) StoreLDSTile Value: FP6x16(364) END
 // (op 722) Deallocate{}(1135) BEGIN
 // (op 1135) Deallocate 543
 // Deleting tag 543
 // (op 1135) Deallocate 540
 // Freeing DataFlowTag540: 1 VGPR (Value: UInt32) (op 727): v14

 // Deleting tag 540
 // (op 1135) Deallocate 541
 // Deleting tag 541
 // (op 1135) Deallocate 542
 // Deleting tag 542
 // (op 722) Deallocate{}(1135) END
 // (op 722) end: generate({719})
 // (op 722) Unlock Scope 722
 // Unlock None
 // (op 441) Scope(722) END
 // (op 441) Scope(828) BEGIN
 // (op 828) Lock Scope 828
 // Lock Branch
 // (op 828) generate({825})
 // (op 828) NOP(825) BEGIN
 // (op 825) generate({})
 // (op 825) end: generate({})
 // (op 828) NOP(825) END
 // (op 828) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}(833) BEGIN
 // (op 833) Assign dim(572) = {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}
 // tag 572: v**UNALLOCATED**
 // (op 833) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32} into DataFlowTag572: VGPR Value: UInt32 x 1: (unallocated)
 // (op 833) reg expression
 // (op 833) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 833): v12
v_and_b32 v12, -4, v1 // (op 833) 
 // (op 833) BEGIN: Tile[1]Flatten
 // (op 833) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 833): v14
v_sub_u32 v14, v1, v12 // (op 833) 
 // (op 833) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 833): v12

 // (op 833) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 833): v12
v_and_b32 v12, 4294967292, v1 // (op 833) 
 // (op 833) BEGIN: Flatten
 // (op 833) {Flatten: Add(v12:U32, {Tile[1]Flatten: v14:U32})U32}
// Allocated DataFlowTag572: 1 VGPR (Value: UInt32) (op 833): v17
v_add_u32 v17, v12, v14 // (op 833) 
 // (op 833) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 833): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 833): v12

 // (op 828) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}(833) END
 // (op 828) Assign VGPR 4:U32(834) BEGIN
 // (op 834) Assign dim(573) = 4:U32
 // (op 834) Assign stride expression
 // (op 828) Assign VGPR 4:U32(834) END
 // (op 828) NOP(826) BEGIN
 // (op 826) generate({})
 // (op 826) end: generate({})
 // (op 828) NOP(826) END
 // (op 828) Assign VGPR 1:U32(838) BEGIN
 // (op 838) Assign dim(575) = 1:U32
 // (op 838) Assign stride expression
 // (op 828) Assign VGPR 1:U32(838) END
 // (op 828) StoreLDSTile Value: E8M0(365) BEGIN
 // (op 365) GEN: StoreLDSTile
 // (op 365) GEN: storeMacroTileLDS OP 365 LDS 32 MacroTile 10
 // (op 365) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 365) Generate 1:U32 into nullptr
 // (op 365) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
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

 // (op 365) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag572: VGPR Value: UInt32 x 1: v17
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 365) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b8 v17, v19 offset:4608 // (op 365) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected stall of 80. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1971 (current 1876)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 828) StoreLDSTile Value: E8M0(365) END
 // (op 828) Deallocate{}(1137) BEGIN
 // (op 1137) Deallocate 575
 // Deleting tag 575
 // (op 1137) Deallocate 572
 // Freeing DataFlowTag572: 1 VGPR (Value: UInt32) (op 833): v17

 // Deleting tag 572
 // (op 1137) Deallocate 573
 // Deleting tag 573
 // (op 1137) Deallocate 574
 // Deleting tag 574
 // (op 828) Deallocate{}(1137) END
 // (op 828) end: generate({825})
 // (op 828) Unlock Scope 828
 // Unlock None
 // (op 441) Scope(828) END
 // (op 441) NOP(333) BEGIN
 // (op 333) generate({})
 // (op 333) end: generate({})
 // (op 441) NOP(333) END
 // (op 441) NOP(694) BEGIN
 // (op 694) generate({})
 // (op 694) end: generate({})
 // (op 441) NOP(694) END
 // (op 441) NOP(691) BEGIN
 // (op 691) generate({})
 // (op 691) end: generate({})
 // (op 441) NOP(691) END
 // (op 441) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(698) BEGIN
 // (op 698) Assign dim(532) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 532: v**UNALLOCATED**
 // (op 698) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag532: VGPR Value: UInt32 x 1: (unallocated)
 // (op 698) reg expression
 // (op 698) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 698): v12
v_and_b32 v12, -4, v1 // (op 698) 
 // (op 698) BEGIN: Tile[1]
 // (op 698) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 698): v14
v_sub_u32 v14, v1, v12 // (op 698) 
 // (op 698) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 698): v12

 // (op 698) BEGIN: Flatten
 // (op 698) {Flatten: ShiftL({Tile[1]: v14:U32}, 5:U32)U32}
 // (op 698) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 698)         value (Tile[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 698)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 698): v12
v_lshlrev_b32 v12, 5, v14 // (op 698) 
 // (op 698) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 698): v14

 // (op 698) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 698): v14
v_lshrrev_b32 v14, 2, v1 // (op 698) 
 // (op 698) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 698): v17
v_lshl_add_u32 v17, v14, 7, v12 // (op 698) 
 // (op 698) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 698): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 698): v14

 // (op 698) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 698): v12
v_mul_lo_u32 v12, v17, 6 // (op 698) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 698): v17

 // (op 698) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag532: 1 VGPR (Value: UInt32) (op 698): v14
v_lshrrev_b32 v14, 3, v12 // (op 698) 
 // Freeing : 1 VGPR (Value: UInt32) (op 698): v12

 // (op 441) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(698) END
 // (op 441) Assign VGPR 96:U32(699) BEGIN
 // (op 699) Assign dim(533) = 96:U32
 // (op 699) Assign stride expression
 // (op 441) Assign VGPR 96:U32(699) END
 // (op 441) NOP(692) BEGIN
 // (op 692) generate({})
 // (op 692) end: generate({})
 // (op 441) NOP(692) END
 // (op 441) Assign VGPR 0:U32(703) BEGIN
 // (op 703) Assign dim(535) = 0:U32
 // (op 703) Assign stride expression
 // (op 441) Assign VGPR 0:U32(703) END
 // (op 441) NOP(746) BEGIN
 // (op 746) generate({})
 // (op 746) end: generate({})
 // (op 441) NOP(746) END
 // (op 441) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(751) BEGIN
 // (op 751) Assign dim(548) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 548: v**UNALLOCATED**
 // (op 751) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag548: VGPR Value: UInt32 x 1: (unallocated)
 // (op 751) reg expression
 // (op 751) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 751): v12
v_and_b32 v12, -8, v1 // (op 751) 
 // (op 751) BEGIN: Tile[1]
 // (op 751) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 751): v17
v_sub_u32 v17, v1, v12 // (op 751) 
 // (op 751) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 751): v12

 // (op 751) BEGIN: Flatten
 // (op 751) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 751) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 751)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 751)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 751): v12
v_lshlrev_b32 v12, 4, v17 // (op 751) 
 // (op 751) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 751): v17

 // (op 751) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 751): v17
v_lshrrev_b32 v17, 3, v1 // (op 751) 
 // (op 751) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 751): v33
v_lshl_add_u32 v33, v17, 7, v12 // (op 751) 
 // (op 751) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 751): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 751): v17

 // (op 751) Multiply({Flatten: v33:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 751): v12
v_mul_lo_u32 v12, v33, 6 // (op 751) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 751): v33

 // (op 751) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag548: 1 VGPR (Value: UInt32) (op 751): v17
v_lshrrev_b32 v17, 3, v12 // (op 751) 
 // Freeing : 1 VGPR (Value: UInt32) (op 751): v12

 // (op 441) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(751) END
 // (op 441) Assign VGPR 96:U32(752) BEGIN
 // (op 752) Assign dim(549) = 96:U32
 // (op 752) Assign stride expression
 // (op 441) Assign VGPR 96:U32(752) END
 // (op 441) NOP(747) BEGIN
 // (op 747) generate({})
 // (op 747) end: generate({})
 // (op 441) NOP(747) END
 // (op 441) Assign VGPR 0:U32(756) BEGIN
 // (op 756) Assign dim(551) = 0:U32
 // (op 756) Assign stride expression
 // (op 441) Assign VGPR 0:U32(756) END
 // (op 441) NOP(799) BEGIN
 // (op 799) generate({})
 // (op 799) end: generate({})
 // (op 441) NOP(799) END
 // (op 441) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}(804) BEGIN
 // (op 804) Assign dim(564) = {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}
 // tag 564: v**UNALLOCATED**
 // (op 804) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32} into DataFlowTag564: VGPR Value: UInt32 x 1: (unallocated)
 // (op 804) reg expression
 // (op 804) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 804): v12
v_and_b32 v12, -4, v1 // (op 804) 
 // (op 804) BEGIN: Tile[1]Flatten
 // (op 804) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 804): v33
v_sub_u32 v33, v1, v12 // (op 804) 
 // (op 804) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 804): v12

 // (op 804) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 804): v12
v_and_b32 v12, 4294967292, v1 // (op 804) 
 // (op 804) BEGIN: Flatten
 // (op 804) {Flatten: Add(v12:U32, {Tile[1]Flatten: v33:U32})U32}
// Allocated DataFlowTag564: 1 VGPR (Value: UInt32) (op 804): v40
v_add_u32 v40, v12, v33 // (op 804) 
 // (op 804) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 804): v33

 // Freeing : 1 VGPR (Value: UInt32) (op 804): v12

 // (op 441) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}(804) END
 // (op 441) Assign VGPR 4:U32(805) BEGIN
 // (op 805) Assign dim(565) = 4:U32
 // (op 805) Assign stride expression
 // (op 441) Assign VGPR 4:U32(805) END
 // (op 441) NOP(800) BEGIN
 // (op 800) generate({})
 // (op 800) end: generate({})
 // (op 441) NOP(800) END
 // (op 441) Assign VGPR 1:U32(809) BEGIN
 // (op 809) Assign dim(567) = 1:U32
 // (op 809) Assign stride expression
 // (op 441) Assign VGPR 1:U32(809) END
 // (op 441) NOP(886) BEGIN
 // (op 886) generate({})
 // (op 886) end: generate({})
 // (op 441) NOP(886) END
 // (op 441) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(891) BEGIN
 // (op 891) Assign dim(590) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 590: v**UNALLOCATED**
 // (op 891) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag590: VGPR Value: UInt32 x 1: (unallocated)
 // (op 891) reg expression
 // (op 891) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 891): v12
v_and_b32 v12, -4, v1 // (op 891) 
 // (op 891) BEGIN: Tile[1]
 // (op 891) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 891): v33
v_sub_u32 v33, v1, v12 // (op 891) 
 // (op 891) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 891): v12

 // (op 891) BEGIN: Flatten
 // (op 891) {Flatten: ShiftL({Tile[1]: v33:U32}, 5:U32)U32}
 // (op 891) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 891)         value (Tile[1]: VGPR Value: UInt32 x 1: v33) 
 // (op 891)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 891): v12
v_lshlrev_b32 v12, 5, v33 // (op 891) 
 // (op 891) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 891): v33

 // (op 891) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 891): v33
v_lshrrev_b32 v33, 2, v1 // (op 891) 
 // (op 891) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 891): v41
v_lshl_add_u32 v41, v33, 7, v12 // (op 891) 
 // (op 891) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 891): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 891): v33

 // (op 891) Multiply({Flatten: v41:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 891): v12
v_mul_lo_u32 v12, v41, 6 // (op 891) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 891): v41

 // (op 891) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag590: 1 VGPR (Value: UInt32) (op 891): v33
v_lshrrev_b32 v33, 3, v12 // (op 891) 
 // Freeing : 1 VGPR (Value: UInt32) (op 891): v12

 // (op 441) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(891) END
 // (op 441) Assign VGPR 96:U32(892) BEGIN
 // (op 892) Assign dim(591) = 96:U32
 // (op 892) Assign stride expression
 // (op 441) Assign VGPR 96:U32(892) END
 // (op 441) NOP(887) BEGIN
 // (op 887) generate({})
 // (op 887) end: generate({})
 // (op 441) NOP(887) END
 // (op 441) Assign VGPR 0:U32(896) BEGIN
 // (op 896) Assign dim(593) = 0:U32
 // (op 896) Assign stride expression
 // (op 441) Assign VGPR 0:U32(896) END
 // (op 441) NOP(910) BEGIN
 // (op 910) generate({})
 // (op 910) end: generate({})
 // (op 441) NOP(910) END
 // (op 441) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(915) BEGIN
 // (op 915) Assign dim(598) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 598: v**UNALLOCATED**
 // (op 915) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag598: VGPR Value: UInt32 x 1: (unallocated)
 // (op 915) reg expression
 // (op 915) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 915): v12
v_and_b32 v12, -8, v1 // (op 915) 
 // (op 915) BEGIN: Tile[1]
 // (op 915) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 915): v41
v_sub_u32 v41, v1, v12 // (op 915) 
 // (op 915) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 915): v12

 // (op 915) BEGIN: Flatten
 // (op 915) {Flatten: ShiftL({Tile[1]: v41:U32}, 4:U32)U32}
 // (op 915) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 915)         value (Tile[1]: VGPR Value: UInt32 x 1: v41) 
 // (op 915)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 915): v12
v_lshlrev_b32 v12, 4, v41 // (op 915) 
 // (op 915) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 915): v41

 // (op 915) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 915): v41
v_lshrrev_b32 v41, 3, v1 // (op 915) 
 // (op 915) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 915): v42
v_lshl_add_u32 v42, v41, 7, v12 // (op 915) 
 // (op 915) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 915): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 915): v41

 // (op 915) Multiply({Flatten: v42:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 915): v12
v_mul_lo_u32 v12, v42, 6 // (op 915) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 915): v42

 // (op 915) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag598: 1 VGPR (Value: UInt32) (op 915): v41
v_lshrrev_b32 v41, 3, v12 // (op 915) 
 // Freeing : 1 VGPR (Value: UInt32) (op 915): v12

 // (op 441) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(915) END
 // (op 441) Assign VGPR 96:U32(916) BEGIN
 // (op 916) Assign dim(599) = 96:U32
 // (op 916) Assign stride expression
 // (op 441) Assign VGPR 96:U32(916) END
 // (op 441) NOP(911) BEGIN
 // (op 911) generate({})
 // (op 911) end: generate({})
 // (op 441) NOP(911) END
 // (op 441) Assign VGPR 0:U32(920) BEGIN
 // (op 920) Assign dim(601) = 0:U32
 // (op 920) Assign stride expression
 // (op 441) Assign VGPR 0:U32(920) END
 // (op 441) NOP(934) BEGIN
 // (op 934) generate({})
 // (op 934) end: generate({})
 // (op 441) NOP(934) END
 // (op 441) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}(939) BEGIN
 // (op 939) Assign dim(606) = {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}
 // tag 606: v**UNALLOCATED**
 // (op 939) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32} into DataFlowTag606: VGPR Value: UInt32 x 1: (unallocated)
 // (op 939) reg expression
 // (op 939) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 939): v12
v_and_b32 v12, -4, v1 // (op 939) 
 // (op 939) BEGIN: Tile[1]Flatten
 // (op 939) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 939): v42
v_sub_u32 v42, v1, v12 // (op 939) 
 // (op 939) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 939): v12

 // (op 939) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 939): v12
v_and_b32 v12, 4294967292, v1 // (op 939) 
 // (op 939) BEGIN: Flatten
 // (op 939) {Flatten: Add(v12:U32, {Tile[1]Flatten: v42:U32})U32}
// Allocated DataFlowTag606: 1 VGPR (Value: UInt32) (op 939): v43
v_add_u32 v43, v12, v42 // (op 939) 
 // (op 939) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 939): v42

 // Freeing : 1 VGPR (Value: UInt32) (op 939): v12

 // (op 441) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}(939) END
 // (op 441) Assign VGPR 4:U32(940) BEGIN
 // (op 940) Assign dim(607) = 4:U32
 // (op 940) Assign stride expression
 // (op 441) Assign VGPR 4:U32(940) END
 // (op 441) NOP(935) BEGIN
 // (op 935) generate({})
 // (op 935) end: generate({})
 // (op 441) NOP(935) END
 // (op 441) Assign VGPR 1:U32(944) BEGIN
 // (op 944) Assign dim(609) = 1:U32
 // (op 944) Assign stride expression
 // (op 441) Assign VGPR 1:U32(944) END
 // (op 441) ForLoopOp KLoop: LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL(54) BEGIN
 // (op 54) Initialize For Loop
 // (op 54) generate({55})
 // (op 54) Assign SGPR 0:U32(55) BEGIN
 // (op 55) Assign dim(361) = 0:U32
 // tag 361: s**UNALLOCATED**
 // (op 55) Generate 0:U32 into DataFlowTag361: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag361: 1 SGPR (Value: UInt32) (op 55): s0
s_mov_b32 s0, 0 // (op 55) call()
 // (op 54) Assign SGPR 0:U32(55) END
 // (op 54) end: generate({55})
 // (op 54) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 54) 
 // (op 54) Generate LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 54) LessThan({DataFlowTag361: s0:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 54) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32
 // (op 54) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 54): s14, s15
s_ashr_i64 s[14:15], s[28:29], 7 // (op 54) 
// Allocated : 1 SGPR (Value: UInt32) (op 54): s1
s_mov_b32 s1, s14 // (op 54) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 54): s14, s15

// Allocated : 1 SGPR (Value: UInt32) (op 54): s14
s_and_b32 s14, s1, 4294967294 // (op 54) 
 // Freeing : 1 SGPR (Value: UInt32) (op 54): s1

s_cmp_lt_u32 s0, s14 // (op 54) 
 // Freeing : 1 SGPR (Value: UInt32) (op 54): s14

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 54) 
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54:
 // (op 54) 
 // (op 54) generate({56, 984})
 // (op 54) Assign SGPR Add(DataFlowTag(361)U32, 2:U32)U32(56) BEGIN
 // (op 56) Assign dim(361) = Add(DataFlowTag(361)U32, 2:U32)U32
 // (op 56) Generate Add(DataFlowTag(361)U32, 2:U32)U32 into DataFlowTag361: SGPR Value: UInt32 x 1: s0
 // (op 56) reg expression
 // (op 56) Add({DataFlowTag361: s0:U32}, 2:U32)U32
 // (op 54) Barrier(984) BEGIN
s_add_u32 s0, s0, 2 // (op 56) 
 // (op 54) Assign SGPR Add(DataFlowTag(361)U32, 2:U32)U32(56) END
s_waitcnt lgkmcnt(1)
s_barrier  // (op 984) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 54) Barrier(984) END
 // (op 54) SetCoordinate(63) BEGIN
 // (op 63) generate({})
 // (op 63) end: generate({})
 // (op 63) generate({65})
 // (op 63) LoadLDSTile{Value: FP6x16}(65) BEGIN
 // (op 65) GEN: loadMacroTileWAVELDS OP 65 LDS 16 WaveTile 99
 // (op 65) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 65) Generate 0:U32 into nullptr
 // tag 14: v**UNALLOCATED**
 // (op 65) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 65) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag544: VGPR Value: UInt32 x 1: v3
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1536
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 65) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 65): v44, v45, v46, v47, v48, v49
ds_read_b128 v[44:47], v3 offset:1536 // (op 65) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 2018 (current 1923)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
ds_read_b64 v[48:49], v3 offset:1552 // (op 65) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 2019 (current 1924)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 63) LoadLDSTile{Value: FP6x16}(65) END
 // (op 63) end: generate({65})
 // (op 54) SetCoordinate(63) END
 // (op 54) Barrier(997) BEGIN
s_barrier  // (op 997) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 54) Barrier(997) END
 // (op 54) SetCoordinate(68) BEGIN
 // (op 68) generate({})
 // (op 68) end: generate({})
 // (op 68) generate({70})
 // (op 68) LoadLDSTile{Value: FP6x16}(70) BEGIN
 // (op 70) GEN: loadMacroTileWAVELDS OP 70 LDS 24 WaveTile 165
 // (op 70) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 70) Generate 0:U32 into nullptr
 // tag 22: v**UNALLOCATED**
 // (op 70) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
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

 // (op 70) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v4
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
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
 // (op 70) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 70): v50, v51, v52, v53, v54, v55
ds_read_b128 v[50:53], v4 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected stall of 46. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2067 (current 1972)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
ds_read_b64 v[54:55], v4 offset:16 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected stall of 46. CBNW: 1, Inc: 3
// DSMEM: Expected complete at 2114 (current 2019)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 68) LoadLDSTile{Value: FP6x16}(70) END
 // (op 68) end: generate({70})
 // (op 54) SetCoordinate(68) END
 // (op 54) Barrier(1011) BEGIN
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1011) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 54) Barrier(1011) END
 // (op 54) SetCoordinate(73) BEGIN
 // (op 73) generate({})
 // (op 73) end: generate({})
 // (op 73) generate({75})
 // (op 73) LoadLDSTile{Value: E8M0}(75) BEGIN
 // (op 75) GEN: loadMacroTileWAVELDS OP 75 LDS 32 WaveTile 231
 // (op 75) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 75) Generate 1:U32 into nullptr
 // tag 30: v**UNALLOCATED**
 // (op 75) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
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

 // (op 75) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag576: VGPR Value: UInt32 x 1: v0
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 75) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 75): v12
ds_read_u8 v12, v0 offset:4608 // (op 75) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 2117 (current 2022)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 73) LoadLDSTile{Value: E8M0}(75) END
 // (op 73) end: generate({75})
 // (op 54) SetCoordinate(73) END
 // (op 54) Multiply(82) BEGIN
 // (op 82) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile16xFP6, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:49], v[50:55], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 82) 
 // (op 54) Multiply(82) END
 // (op 54) Deallocate{}(1053) BEGIN
 // (op 1053) Deallocate 14
 // Freeing : 6 VGPRs (Value: FP6x16) (op 65): v44, v45, v46, v47, v48, v49

 // Deleting tag 14
 // (op 1053) Deallocate 22
 // Freeing : 6 VGPRs (Value: FP6x16) (op 70): v50, v51, v52, v53, v54, v55

 // Deleting tag 22
 // (op 1053) Deallocate 30
 // Freeing : 1 VGPR (Value: E8M0) (op 75): v12

 // Deleting tag 30
 // (op 54) Deallocate{}(1053) END
 // (op 54) NOP(382) BEGIN
 // (op 382) generate({})
 // (op 382) end: generate({})
 // (op 54) NOP(382) END
 // (op 54) SetCoordinate(78) BEGIN
 // (op 78) generate({})
 // (op 78) end: generate({})
 // (op 78) generate({264})
 // (op 78) SetCoordinate(264) BEGIN
 // (op 264) generate({})
 // (op 264) end: generate({})
 // (op 264) generate({80})
 // (op 264) LoadTiled Value: E8M0(80) BEGIN
 // (op 80) GEN: loadMacroTileWAVE OP80 WaveTile 261
 // (op 80) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 80) Generate 1:U64 into nullptr
 // (op 80) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 522: v**UNALLOCATED**
// Allocated Offset522: 1 VGPR (Value: UInt32) (op 80): v12
v_mov_b32 v12, v18 // (op 80) 
 // (op 80) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
 // 	info.rowOffsetReg = Offset522: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 80) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v16, v12, s[52:55], 0 offen offset:8 // (op 80) Load value
// VMEM: Expected complete at 2506 (current 2026)
 // (op 264) LoadTiled Value: E8M0(80) END
 // (op 264) end: generate({80})
 // (op 78) SetCoordinate(264) END
 // (op 78) end: generate({264})
 // (op 54) SetCoordinate(78) END
 // (op 54) SetCoordinate(267) BEGIN
 // (op 267) generate({})
 // (op 267) end: generate({})
 // (op 267) generate({10})
 // (op 267) LoadTiled Value: FP6x16(10) BEGIN
 // (op 10) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 10) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 10) Generate 0:U64 into nullptr
 // (op 10) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[20:22]
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 484: v**UNALLOCATED**
// Allocated Offset484: 1 VGPR (Value: UInt32) (op 10): v42
v_mov_b32 v42, v15 // (op 10) 
 // (op 10) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[20:22]
 // 	info.rowOffsetReg = Offset484: VGPR Value: UInt32 x 1: v42
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
 // 	info.offset = Literal Value: UInt32 x 0: 192
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 10) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_dwordx3 v[20:22], v42, s[44:47], 0 offen offset:192 // (op 10) Load value
// VMEM: Expected complete at 2508 (current 2028)
 // (op 267) LoadTiled Value: FP6x16(10) END
 // (op 267) end: generate({10})
 // (op 54) SetCoordinate(267) END
 // (op 54) SetCoordinate(259) BEGIN
 // (op 259) generate({})
 // (op 259) end: generate({})
 // (op 259) generate({4})
 // (op 259) LoadTiled Value: FP6x16(4) BEGIN
 // (op 4) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 4) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 4) Generate 0:U64 into nullptr
 // (op 4) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 465: v**UNALLOCATED**
// Allocated Offset465: 1 VGPR (Value: UInt32) (op 4): v44
v_mov_b32 v44, v11 // (op 4) 
 // (op 4) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = Offset465: VGPR Value: UInt32 x 1: v44
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
 // 	info.offset = Literal Value: UInt32 x 0: 192
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 4) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_dwordx4 v[24:27], v44, s[40:43], 0 offen offset:192 // (op 4) Load value
// VMEM: Expected stall of 221. CBNW: 4, Inc: 3
// VMEM: Expected complete at 2731 (current 2251)
buffer_load_dwordx2 v[28:29], v44, s[40:43], 0 offen offset:208 // (op 4) Load value
// VMEM: Expected stall of 255. CBNW: 5, Inc: 3
// VMEM: Expected complete at 2987 (current 2507)
 // (op 259) LoadTiled Value: FP6x16(4) END
 // (op 259) end: generate({4})
 // (op 54) SetCoordinate(259) END
 // (op 54) SetCoordinate(272) BEGIN
 // (op 272) generate({})
 // (op 272) end: generate({})
 // (op 272) generate({17})
 // (op 272) LoadTiled Value: E8M0(17) BEGIN
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
 // 	info.data = VGPR Value: E8M0 x 1: v19
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 503: v**UNALLOCATED**
// Allocated Offset503: 1 VGPR (Value: UInt32) (op 17): v45
v_mov_b32 v45, v13 // (op 17) 
 // (op 17) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = Offset503: VGPR Value: UInt32 x 1: v45
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
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
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v19, v45, s[48:51], 0 offen offset:8 // (op 17) Load value
// VMEM: Expected complete at 2989 (current 2509)
 // (op 272) LoadTiled Value: E8M0(17) END
 // (op 272) end: generate({17})
 // (op 54) SetCoordinate(272) END
 // (op 54) Barrier(1034) BEGIN
s_barrier  // (op 1034) 
 // (op 54) Barrier(1034) END
 // (op 54) StoreLDSTile Value: FP6x16(235) BEGIN
 // (op 235) GEN: StoreLDSTile
 // (op 235) GEN: storeMacroTileLDS OP 235 LDS 436 MacroTile 434
 // (op 235) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 235) Generate 0:U32 into nullptr
 // (op 235) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[30:32]
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
 // 	info.offset = Literal Value: UInt32 x 0: 4736
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

 // (op 235) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[30:32]
 // 	info.rowOffsetReg = DataFlowTag598: VGPR Value: UInt32 x 1: v41
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
 // 	info.offset = Literal Value: UInt32 x 0: 4736
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 235) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(8)
ds_write_b96 v41, v[30:32] offset:4736 // (op 235) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 2702 (current 2607)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 1536)
 // (op 54) StoreLDSTile Value: FP6x16(235) END
 // (op 54) Deallocate{}(1111) BEGIN
 // (op 1111) Deallocate 600
 // Deleting tag 600
 // (op 54) Deallocate{}(1111) END
 // (op 54) Barrier(1028) BEGIN
s_barrier  // (op 1028) 
 // (op 54) Barrier(1028) END
 // (op 54) StoreLDSTile Value: FP6x16(225) BEGIN
 // (op 225) GEN: StoreLDSTile
 // (op 225) GEN: storeMacroTileLDS OP 225 LDS 432 MacroTile 430
 // (op 225) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 225) Generate 0:U32 into nullptr
 // (op 225) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[34:39]
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
 // 	info.offset = Literal Value: UInt32 x 0: 6272
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

 // (op 225) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[34:39]
 // 	info.rowOffsetReg = DataFlowTag590: VGPR Value: UInt32 x 1: v33
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
 // 	info.offset = Literal Value: UInt32 x 0: 6272
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 225) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(7)
ds_write_b128 v33, v[34:37] offset:6272 // (op 225) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 2705 (current 2610)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 6272, Size: 3072)
s_waitcnt vmcnt(6)
ds_write_b64 v33, v[38:39] offset:6288 // (op 225) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 2707 (current 2612)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 6272, Size: 3072)
 // (op 54) StoreLDSTile Value: FP6x16(225) END
 // (op 54) Deallocate{}(1087) BEGIN
 // (op 1087) Deallocate 592
 // Deleting tag 592
 // (op 54) Deallocate{}(1087) END
 // (op 54) Barrier(1040) BEGIN
s_barrier  // (op 1040) 
 // (op 54) Barrier(1040) END
 // (op 54) StoreLDSTile Value: E8M0(236) BEGIN
 // (op 236) GEN: StoreLDSTile
 // (op 236) GEN: storeMacroTileLDS OP 236 LDS 440 MacroTile 438
 // (op 236) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 236) Generate 1:U32 into nullptr
 // (op 236) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v10
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9344
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

 // (op 236) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v10
 // 	info.rowOffsetReg = DataFlowTag606: VGPR Value: UInt32 x 1: v43
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9344
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 236) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b8 v43, v10 offset:9344 // (op 236) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2799 (current 2704)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 9344, Size: 128)
 // (op 54) StoreLDSTile Value: E8M0(236) END
 // (op 54) Deallocate{}(1113) BEGIN
 // (op 1113) Deallocate 608
 // Deleting tag 608
 // (op 54) Deallocate{}(1113) END
 // (op 54) NOP(378) BEGIN
 // (op 378) generate({})
 // (op 378) end: generate({})
 // (op 54) NOP(378) END
 // (op 54) Barrier(1025) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1025) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 6272, Size: 3072)
 // (op 54) Barrier(1025) END
 // (op 54) SetCoordinate(226) BEGIN
 // (op 226) generate({})
 // (op 226) end: generate({})
 // (op 226) generate({227})
 // (op 226) LoadLDSTile{Value: FP6x16}(227) BEGIN
 // (op 227) GEN: loadMacroTileWAVELDS OP 227 LDS 432 WaveTile 99
 // (op 227) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 227) Generate 0:U32 into nullptr
 // tag 442: v**UNALLOCATED**
 // (op 227) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 6272
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

 // (op 227) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag594: VGPR Value: UInt32 x 1: v6
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 6272
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 227) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 227): v46, v47, v48, v49, v50, v51
ds_read_b128 v[46:49], v6 offset:6272 // (op 227) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 2805 (current 2710)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 6272, Size: 3072)
ds_read_b64 v[50:51], v6 offset:6288 // (op 227) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 2806 (current 2711)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 6272, Size: 3072)
 // (op 226) LoadLDSTile{Value: FP6x16}(227) END
 // (op 226) Deallocate{}(1093) BEGIN
 // (op 1093) Deallocate 596
 // Deleting tag 596
 // (op 226) Deallocate{}(1093) END
 // (op 226) end: generate({227})
 // (op 54) SetCoordinate(226) END
 // (op 54) Deallocate{}(1089) BEGIN
 // (op 1089) Deallocate 432
 // Deleting tag 432
 // (op 54) Deallocate{}(1089) END
 // (op 54) Barrier(1031) BEGIN
 // (op 54) Barrier(1091) BEGIN
s_barrier  // (op 1031) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 1536)
 // (op 54) Barrier(1031) END
s_barrier  // (op 1091) 
 // (op 54) Barrier(1091) END
 // (op 54) SetCoordinate(228) BEGIN
 // (op 228) generate({})
 // (op 228) end: generate({})
 // (op 228) generate({229})
 // (op 228) LoadLDSTile{Value: FP6x16}(229) BEGIN
 // (op 229) GEN: loadMacroTileWAVELDS OP 229 LDS 436 WaveTile 165
 // (op 229) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 229) Generate 0:U32 into nullptr
 // tag 444: v**UNALLOCATED**
 // (op 229) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4736
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

 // (op 229) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag602: VGPR Value: UInt32 x 1: v7
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4736
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 229) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 229): v52, v53, v54, v55, v56, v57
ds_read_b128 v[52:55], v7 offset:4736 // (op 229) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected stall of 86. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2895 (current 2800)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 1536)
ds_read_b64 v[56:57], v7 offset:4752 // (op 229) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected stall of 5. CBNW: 1, Inc: 3
// DSMEM: Expected complete at 2901 (current 2806)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 1536)
 // (op 228) LoadLDSTile{Value: FP6x16}(229) END
 // (op 228) Deallocate{}(1099) BEGIN
 // (op 1099) Deallocate 604
 // Deleting tag 604
 // (op 228) Deallocate{}(1099) END
 // (op 228) end: generate({229})
 // (op 54) SetCoordinate(228) END
 // (op 54) Deallocate{}(1095) BEGIN
 // (op 1095) Deallocate 436
 // Deleting tag 436
 // (op 54) Deallocate{}(1095) END
 // (op 54) Barrier(1037) BEGIN
 // (op 54) Barrier(1097) BEGIN
s_barrier  // (op 1097) 
 // (op 54) Barrier(1097) END
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1037) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 9344, Size: 128)
 // (op 54) Barrier(1037) END
 // (op 54) SetCoordinate(230) BEGIN
 // (op 230) generate({})
 // (op 230) end: generate({})
 // (op 230) generate({231})
 // (op 230) LoadLDSTile{Value: E8M0}(231) BEGIN
 // (op 231) GEN: loadMacroTileWAVELDS OP 231 LDS 440 WaveTile 231
 // (op 231) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 231) Generate 1:U32 into nullptr
 // tag 446: v**UNALLOCATED**
 // (op 231) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9344
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

 // (op 231) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag610: VGPR Value: UInt32 x 1: v5
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9344
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 231) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 231): v58
ds_read_u8 v58, v5 offset:9344 // (op 231) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 2905 (current 2810)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 9344, Size: 128)
 // (op 230) LoadLDSTile{Value: E8M0}(231) END
 // (op 230) Deallocate{}(1105) BEGIN
 // (op 1105) Deallocate 612
 // Deleting tag 612
 // (op 230) Deallocate{}(1105) END
 // (op 230) end: generate({231})
 // (op 54) SetCoordinate(230) END
 // (op 54) Deallocate{}(1101) BEGIN
 // (op 1101) Deallocate 440
 // Deleting tag 440
 // (op 54) Deallocate{}(1101) END
 // (op 54) Multiply(234) BEGIN
 // (op 234) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile16xFP6, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
 // (op 54) Barrier(1103) BEGIN
s_barrier  // (op 1103) 
 // (op 54) Barrier(1103) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:57], a[0:3], v58, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 234) 
 // (op 54) Multiply(234) END
 // (op 54) Deallocate{}(1109) BEGIN
 // (op 1109) Deallocate 446
 // Freeing : 1 VGPR (Value: E8M0) (op 231): v58

 // Deleting tag 446
 // (op 1109) Deallocate 444
 // Freeing : 6 VGPRs (Value: FP6x16) (op 229): v52, v53, v54, v55, v56, v57

 // Deleting tag 444
 // (op 1109) Deallocate 442
 // Freeing : 6 VGPRs (Value: FP6x16) (op 227): v46, v47, v48, v49, v50, v51

 // Deleting tag 442
 // (op 54) Deallocate{}(1109) END
 // (op 54) NOP(398) BEGIN
 // (op 398) generate({})
 // (op 398) end: generate({})
 // (op 54) NOP(398) END
 // (op 54) SetCoordinate(232) BEGIN
 // (op 232) generate({})
 // (op 232) end: generate({})
 // (op 232) generate({283})
 // (op 232) SetCoordinate(283) BEGIN
 // (op 283) generate({})
 // (op 283) end: generate({})
 // (op 283) generate({233})
 // (op 283) LoadTiled Value: E8M0(233) BEGIN
 // (op 233) GEN: loadMacroTileWAVE OP233 WaveTile 261
 // (op 233) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 233) Generate 1:U64 into nullptr
 // (op 233) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v23
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 233) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v23
 // 	info.rowOffsetReg = Offset522: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 12
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 233) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v23, v12, s[52:55], 0 offen offset:12 // (op 233) Load value
// VMEM: Expected stall of 16. CBNW: 2, Inc: 3
// VMEM: Expected complete at 3212 (current 2732)
 // (op 283) LoadTiled Value: E8M0(233) END
 // (op 283) end: generate({233})
 // (op 232) SetCoordinate(283) END
 // (op 232) end: generate({283})
 // (op 54) SetCoordinate(232) END
 // (op 54) Deallocate{}(1107) BEGIN
 // (op 1107) Deallocate 522
 // Freeing Offset522: 1 VGPR (Value: UInt32) (op 80): v12

 // Deleting tag 522
 // (op 1107) Deallocate 524
 // Deleting tag 524
 // (op 54) Deallocate{}(1107) END
 // (op 54) SetCoordinate(286) BEGIN
 // (op 286) generate({})
 // (op 286) end: generate({})
 // (op 286) generate({237})
 // (op 286) LoadTiled Value: FP6x16(237) BEGIN
 // (op 237) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 237) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 237) Generate 0:U64 into nullptr
 // (op 237) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[30:32]
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 237) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[30:32]
 // 	info.rowOffsetReg = Offset484: VGPR Value: UInt32 x 1: v42
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
 // 	info.offset = Literal Value: UInt32 x 0: 288
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 237) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 54) Assign VGPR Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32(652) BEGIN
 // (op 652) Assign dim(520) = Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32
 // (op 652) Generate Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32 into DataFlowTag520: VGPR Value: UInt32 x 1: v18
 // (op 652) reg expression
 // (op 652) Add({DataFlowTag520: v18:U32}, 8:U32)U32
v_add_u32 v18, 8, v18 // (op 652) 
 // (op 54) Assign VGPR Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32(652) END
buffer_load_dwordx3 v[30:32], v42, s[44:47], 0 offen offset:288 // (op 237) Load value
// VMEM: Expected stall of 254. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3468 (current 2988)
 // (op 286) LoadTiled Value: FP6x16(237) END
 // (op 286) end: generate({237})
 // (op 54) SetCoordinate(286) END
 // (op 54) Deallocate{}(1117) BEGIN
 // (op 1117) Deallocate 484
 // Freeing Offset484: 1 VGPR (Value: UInt32) (op 10): v42

 // Deleting tag 484
 // (op 1117) Deallocate 486
 // Deleting tag 486
 // (op 54) Deallocate{}(1117) END
 // (op 54) SetCoordinate(278) BEGIN
 // (op 278) generate({})
 // (op 278) end: generate({})
 // (op 278) generate({224})
 // (op 278) LoadTiled Value: FP6x16(224) BEGIN
 // (op 224) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 224) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 224) Generate 0:U64 into nullptr
 // (op 224) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[34:39]
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
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 224) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[34:39]
 // 	info.rowOffsetReg = Offset465: VGPR Value: UInt32 x 1: v44
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
 // 	info.offset = Literal Value: UInt32 x 0: 288
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 224) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 54) Assign VGPR Add(DataFlowTag(482)U32, ShiftL(Convert(DataFlowTag(483)U64)U32, 1:U32)U32)U32(524) BEGIN
 // (op 524) Assign dim(482) = Add(DataFlowTag(482)U32, ShiftL(Convert(DataFlowTag(483)U64)U32, 1:U32)U32)U32
 // (op 524) Generate Add(DataFlowTag(482)U32, ShiftL(Convert(DataFlowTag(483)U64)U32, 1:U32)U32)U32 into DataFlowTag482: VGPR Value: UInt32 x 1: v15
 // (op 524) reg expression
 // (op 524) Add({DataFlowTag482: v15:U32}, 192:U32)U32
v_add_u32 v15, 192, v15 // (op 524) 
 // (op 54) Assign VGPR Add(DataFlowTag(482)U32, ShiftL(Convert(DataFlowTag(483)U64)U32, 1:U32)U32)U32(524) END
buffer_load_dwordx4 v[34:37], v44, s[40:43], 0 offen offset:288 // (op 224) Load value
// VMEM: Expected complete at 3470 (current 2990)
buffer_load_dwordx2 v[38:39], v44, s[40:43], 0 offen offset:304 // (op 224) Load value
// VMEM: Expected stall of 222. CBNW: 5, Inc: 3
// VMEM: Expected complete at 3693 (current 3213)
 // (op 278) LoadTiled Value: FP6x16(224) END
 // (op 278) end: generate({224})
 // (op 54) SetCoordinate(278) END
 // (op 54) Deallocate{}(1115) BEGIN
 // (op 1115) Deallocate 467
 // Deleting tag 467
 // (op 1115) Deallocate 465
 // Freeing Offset465: 1 VGPR (Value: UInt32) (op 4): v44

 // Deleting tag 465
 // (op 54) Deallocate{}(1115) END
 // (op 54) SetCoordinate(291) BEGIN
 // (op 291) generate({})
 // (op 291) end: generate({})
 // (op 291) generate({238})
 // (op 291) LoadTiled Value: E8M0(238) BEGIN
 // (op 238) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 238) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 238) Generate 1:U64 into nullptr
 // (op 238) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v10
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 238) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v10
 // 	info.rowOffsetReg = Offset503: VGPR Value: UInt32 x 1: v45
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 12
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 238) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 54) Assign VGPR Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32(460) BEGIN
 // (op 460) Assign dim(463) = Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32
 // (op 460) Generate Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32 into DataFlowTag463: VGPR Value: UInt32 x 1: v11
 // (op 460) reg expression
 // (op 460) Add({DataFlowTag463: v11:U32}, 192:U32)U32
v_add_u32 v11, 192, v11 // (op 460) 
 // (op 54) Assign VGPR Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32(460) END
buffer_load_ubyte v10, v45, s[48:51], 0 offen offset:12 // (op 238) Load value
// VMEM: Expected stall of 254. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3949 (current 3469)
 // (op 291) LoadTiled Value: E8M0(238) END
 // (op 291) end: generate({238})
 // (op 54) SetCoordinate(291) END
 // (op 54) Deallocate{}(1119) BEGIN
 // (op 1119) Deallocate 503
 // Freeing Offset503: 1 VGPR (Value: UInt32) (op 17): v45

 // Deleting tag 503
 // (op 1119) Deallocate 505
 // Deleting tag 505
 // (op 54) Deallocate{}(1119) END
 // (op 54) Assign VGPR Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32(588) BEGIN
 // (op 588) Assign dim(501) = Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32
 // (op 588) Generate Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32 into DataFlowTag501: VGPR Value: UInt32 x 1: v13
 // (op 588) reg expression
 // (op 588) Add({DataFlowTag501: v13:U32}, 8:U32)U32
 // (op 54) Barrier(1000) BEGIN
s_barrier  // (op 1000) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 54) Barrier(1000) END
v_add_u32 v13, 8, v13 // (op 588) 
 // (op 54) Assign VGPR Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32(588) END
 // (op 54) StoreLDSTile Value: FP6x16(13) BEGIN
 // (op 13) GEN: StoreLDSTile
 // (op 13) GEN: storeMacroTileLDS OP 13 LDS 24 MacroTile 9
 // (op 13) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 13) Generate 0:U32 into nullptr
 // (op 13) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[20:22]
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

 // (op 13) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[20:22]
 // 	info.rowOffsetReg = DataFlowTag548: VGPR Value: UInt32 x 1: v17
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
 // (op 13) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(8)
ds_write_b96 v17, v[20:22] // (op 13) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 3760 (current 3665)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 54) StoreLDSTile Value: FP6x16(13) END
 // (op 54) Deallocate{}(1045) BEGIN
 // (op 1045) Deallocate 550
 // Deleting tag 550
 // (op 54) Deallocate{}(1045) END
 // (op 54) Barrier(987) BEGIN
s_barrier  // (op 987) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 54) Barrier(987) END
 // (op 54) StoreLDSTile Value: FP6x16(6) BEGIN
 // (op 6) GEN: StoreLDSTile
 // (op 6) GEN: storeMacroTileLDS OP 6 LDS 16 MacroTile 8
 // (op 6) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 6) Generate 0:U32 into nullptr
 // (op 6) 	Dir = Store
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 6) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag532: VGPR Value: UInt32 x 1: v14
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 6) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(7)
ds_write_b128 v14, v[24:27] offset:1536 // (op 6) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 3763 (current 3668)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
s_waitcnt vmcnt(6)
ds_write_b64 v14, v[28:29] offset:1552 // (op 6) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 3765 (current 3670)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 54) StoreLDSTile Value: FP6x16(6) END
 // (op 54) Deallocate{}(1043) BEGIN
 // (op 1043) Deallocate 534
 // Deleting tag 534
 // (op 54) Deallocate{}(1043) END
 // (op 54) Barrier(1014) BEGIN
s_barrier  // (op 1014) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 54) Barrier(1014) END
 // (op 54) StoreLDSTile Value: E8M0(21) BEGIN
 // (op 21) GEN: StoreLDSTile
 // (op 21) GEN: storeMacroTileLDS OP 21 LDS 32 MacroTile 10
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
 // 	info.data = VGPR Value: E8M0 x 1: v19
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
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
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag564: VGPR Value: UInt32 x 1: v40
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
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
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b8 v40, v19 offset:4608 // (op 21) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 3857 (current 3762)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 54) StoreLDSTile Value: E8M0(21) END
 // (op 54) Deallocate{}(1047) BEGIN
 // (op 1047) Deallocate 566
 // Deleting tag 566
 // (op 54) Deallocate{}(1047) END
 // (op 54) NOP(379) BEGIN
 // (op 379) generate({})
 // (op 379) end: generate({})
 // (op 54) NOP(379) END
 // (op 54) end: generate({56, 984})
 // (op 54) For Loop Increment
 // (op 54) generate({})
 // (op 54) end: generate({})
 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
 // (op 54) Generate LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 54) LessThan({DataFlowTag361: s0:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 54) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32
 // (op 54) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 54): s14, s15
s_ashr_i64 s[14:15], s[28:29], 7 // (op 54) 
// Allocated : 1 SGPR (Value: UInt32) (op 54): s1
s_mov_b32 s1, s14 // (op 54) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 54): s14, s15

// Allocated : 1 SGPR (Value: UInt32) (op 54): s14
s_and_b32 s14, s1, 4294967294 // (op 54) 
 // Freeing : 1 SGPR (Value: UInt32) (op 54): s1

s_cmp_lt_u32 s0, s14 // (op 54) 
 // Freeing : 1 SGPR (Value: UInt32) (op 54): s14

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 54) 
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoop_54:
 // (op 54) 
 // (op 54) Unlock For Loop
 // Unlock None
 // (op 441) ForLoopOp KLoop: LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL(54) END
 // (op 441) Deallocate{}(1051) BEGIN
 // (op 1051) Deallocate 591
 // Deleting tag 591
 // (op 1051) Deallocate 464
 // Deleting tag 464
 // (op 1051) Deallocate 434
 // Deleting tag 434
 // (op 1051) Deallocate 502
 // Deleting tag 502
 // (op 1051) Deallocate 590
 // Freeing DataFlowTag590: 1 VGPR (Value: UInt32) (op 891): v33

 // Deleting tag 590
 // (op 1051) Deallocate 463
 // Freeing DataFlowTag463: 1 VGPR (Value: UInt32) (op 468): v11

 // Deleting tag 463
 // (op 1051) Deallocate 468
 // Deleting tag 468
 // (op 1051) Deallocate 599
 // Deleting tag 599
 // (op 1051) Deallocate 609
 // Deleting tag 609
 // (op 1051) Deallocate 482
 // Freeing DataFlowTag482: 1 VGPR (Value: UInt32) (op 532): v15

 // Deleting tag 482
 // (op 1051) Deallocate 483
 // Deleting tag 483
 // (op 1051) Deallocate 506
 // Deleting tag 506
 // (op 1051) Deallocate 485
 // Deleting tag 485
 // (op 1051) Deallocate 466
 // Deleting tag 466
 // (op 1051) Deallocate 438
 // Deleting tag 438
 // (op 1051) Deallocate 565
 // Deleting tag 565
 // (op 1051) Deallocate 532
 // Freeing DataFlowTag532: 1 VGPR (Value: UInt32) (op 698): v14

 // Deleting tag 532
 // (op 1051) Deallocate 533
 // Deleting tag 533
 // (op 1051) Deallocate 535
 // Deleting tag 535
 // (op 1051) Deallocate 548
 // Freeing DataFlowTag548: 1 VGPR (Value: UInt32) (op 751): v17

 // Deleting tag 548
 // (op 1051) Deallocate 549
 // Deleting tag 549
 // (op 1051) Deallocate 551
 // Deleting tag 551
 // (op 1051) Deallocate 564
 // Freeing DataFlowTag564: 1 VGPR (Value: UInt32) (op 804): v40

 // Deleting tag 564
 // (op 1051) Deallocate 567
 // Deleting tag 567
 // (op 1051) Deallocate 598
 // Freeing DataFlowTag598: 1 VGPR (Value: UInt32) (op 915): v41

 // Deleting tag 598
 // (op 1051) Deallocate 601
 // Deleting tag 601
 // (op 1051) Deallocate 361
 // Freeing DataFlowTag361: 1 SGPR (Value: UInt32) (op 55): s0

 // Deleting tag 361
 // (op 1051) Deallocate 448
 // Deleting tag 448
 // (op 1051) Deallocate 607
 // Deleting tag 607
 // (op 1051) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 347): v19

 // Deleting tag 10
 // (op 1051) Deallocate 430
 // Deleting tag 430
 // (op 1051) Deallocate 49
 // Freeing : 1 VGPR (Value: E8M0) (op 337): v16

 // Deleting tag 49
 // (op 1051) Deallocate 525
 // Deleting tag 525
 // (op 1051) Deallocate 523
 // Deleting tag 523
 // (op 1051) Deallocate 9
 // Freeing : 3 VGPRs (Value: FP6x16) (op 341): v20, v21, v22

 // Deleting tag 9
 // (op 1051) Deallocate 521
 // Deleting tag 521
 // (op 1051) Deallocate 8
 // Freeing : 6 VGPRs (Value: FP6x16) (op 344): v24, v25, v26, v27, v28, v29

 // Deleting tag 8
 // (op 1051) Deallocate 520
 // Freeing DataFlowTag520: 1 VGPR (Value: UInt32) (op 660): v18

 // Deleting tag 520
 // (op 1051) Deallocate 606
 // Freeing DataFlowTag606: 1 VGPR (Value: UInt32) (op 939): v43

 // Deleting tag 606
 // (op 1051) Deallocate 504
 // Deleting tag 504
 // (op 1051) Deallocate 501
 // Freeing DataFlowTag501: 1 VGPR (Value: UInt32) (op 596): v13

 // Deleting tag 501
 // (op 1051) Deallocate 487
 // Deleting tag 487
 // (op 1051) Deallocate 593
 // Deleting tag 593
 // (op 441) Deallocate{}(1051) END
 // (op 441) end: generate({436})
 // (op 441) Unlock Scope 441
 // Unlock None
 // (op 310) Scope(441) END
 // (op 310) SetCoordinate(201) BEGIN
 // (op 201) generate({})
 // (op 201) end: generate({})
 // (op 201) generate({202})
 // (op 201) SetCoordinate(202) BEGIN
 // (op 202) generate({})
 // (op 202) end: generate({})
 // (op 202) generate({418})
 // (op 202) Scope(418) BEGIN
 // (op 418) Lock Scope 418
 // Lock Branch
 // (op 418) generate({412})
 // (op 418) NOP(412) BEGIN
 // (op 412) generate({})
 // (op 412) end: generate({})
 // (op 418) NOP(412) END
 // (op 418) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(423) BEGIN
 // (op 423) Assign dim(450) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 450: v**UNALLOCATED**
 // (op 423) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag450: VGPR Value: UInt32 x 1: (unallocated)
 // (op 423) reg expression
 // (op 423) reg expression
 // (op 423) reg expression
 // (op 423) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 423) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 423): v33
v_lshrrev_b32 v33, 2, v1 // (op 423) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 423): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 423) 
 // (op 423) BEGIN: Tile
 // (op 423) convert
// Allocated : 1 VGPR (Value: UInt32) (op 423): v11
v_lshl_add_u32 v11, s16, 5, v33 // (op 423) 
 // (op 423) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 423): v33

 // (op 423) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 423) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 423): s14
s_lshr_b32 s14, s0, 1 // (op 423) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 423): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 423): v33
v_and_b32 v33, -4, v1 // (op 423) 
 // (op 423) reg expression
 // (op 423) BEGIN: Flatten[1]
 // (op 423) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v33:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 423): v12
v_sub_u32 v12, v1, v33 // (op 423) 
 // (op 423) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 423): v33

 // (op 423) BEGIN: Tile
 // (op 423) {Tile: ShiftL({Flatten[1]: v12:U32}, 5:U32)U32}
 // (op 423) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 423)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 423)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 423): v33
v_lshlrev_b32 v33, 5, v12 // (op 423) 
 // (op 423) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 423): v12

 // (op 423) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 423): v12
v_lshl_add_u32 v12, s14, 8, v33 // (op 423) 
 // (op 423) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 423): v33

 // Freeing : 1 SGPR (Value: UInt32) (op 423): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 423): v24, v25
v_mov_b32 v25, 0 // (op 423) convert
v_mov_b32 v24, v12 // (op 423) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 423): v12

 // (op 423) BEGIN: Split
 // (op 423) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 423)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 423)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 423): v12, v13
v_mul_lo_u32 v13, v11, s3 // (op 423) most significant: low of low * high
 // (op 423) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 423): v33
v_mul_hi_u32 v33, v11, s2 // (op 423) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 423) least significant: low of low * low
v_add_u32 v13, v13, v33 // (op 423) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 423): v33

 // (op 423) Add: dest (VGPR Value: UInt64 x 1: v[12:13]) = 
 // (op 423)      lhs (VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 423)      rhs (Tile: VGPR Value: UInt64 x 1: v[24:25])
// Allocated : 2 SGPRs (Value: Bool64) (op 423): s0, s1
v_add_co_u32 v12, s[0:1], v12, v24 // (op 423) least significant half
v_addc_co_u32 v13, s[0:1], v13, v25, s[0:1] // (op 423) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 423): s0, s1

 // (op 423) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 423): v24, v25

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 423): v11

 // (op 423) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 423) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 423)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 423)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 423) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 423): v14, v15
v_mul_lo_u32 v15, v13, 6 // (op 423) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 423): v11
v_mul_hi_u32 v11, v12, 6 // (op 423) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 423) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 423) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 423): v11

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 423): v12, v13

 // (op 423) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 423): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 423) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 423): v14, v15

// Allocated DataFlowTag450: 1 VGPR (Value: UInt32) (op 423): v11
v_mov_b32 v11, v12 // (op 423) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 423): v12, v13

 // (op 418) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 5:U32, LogicalShiftR(DataFlowTag(72)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -4:I)U32)U32}, 5:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(423) END
 // (op 418) Assign VGPR 96:U64(424) BEGIN
 // (op 424) Assign dim(451) = 96:U64
 // (op 424) Assign stride expression
 // (op 418) Assign VGPR 96:U64(424) END
 // (op 418) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(425) BEGIN
 // (op 425) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 425) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // (op 425) reg expression
 // (op 425) reg expression
 // (op 425) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 425) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 425)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 425)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 425) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 425): s0, s1
s_mul_i32 s1, s7, 6 // (op 425) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 425) least significant
// Allocated : 1 SGPR (Value: Int32) (op 425): s14
s_mul_hi_u32 s14, s6, 6 // (op 425) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 425) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 425): s14

 // (op 425) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 425) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 425)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 425)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 425): s14, s15
s_mul_i32 s15, s2, s1 // (op 425) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 425): s18
s_mul_i32 s18, s3, s0 // (op 425) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 425) least significant
// Allocated : 1 SGPR (Value: Int32) (op 425): s19
s_mul_hi_u32 s19, s2, s0 // (op 425) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 425) most significant: sum
s_add_u32 s15, s15, s19 // (op 425) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 425): s19

 // Freeing : 1 SGPR (Value: Int32) (op 425): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 425): s0, s1

 // (op 425) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 425): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 425) 
 // Freeing : 2 SGPRs (Value: Int64) (op 425): s14, s15

 // (op 425) reg expression
 // (op 425) reg expression
s_mov_b32 s40, s4 // (op 425) reg expression
s_mov_b32 s41, s5 // (op 425) reg expression
s_mov_b32 s42, s0 // (op 425) reg expression
s_mov_b32 s43, 131072 // (op 425) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 425): s0, s1

 // (op 418) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(425) END
 // (op 418) NOP(414) BEGIN
 // (op 414) generate({})
 // (op 414) end: generate({})
 // (op 418) NOP(414) END
 // (op 418) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(430) BEGIN
 // (op 430) Assign dim(455) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 430) Assign stride expression
 // (op 418) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(430) END
 // (op 418) NOP(415) BEGIN
 // (op 415) generate({})
 // (op 415) end: generate({})
 // (op 418) NOP(415) END
 // (op 418) Assign VGPR 0:U64(433) BEGIN
 // (op 433) Assign dim(457) = 0:U64
 // (op 433) Assign stride expression
 // (op 418) Assign VGPR 0:U64(433) END
 // (op 418) NOP(481) BEGIN
 // (op 481) generate({})
 // (op 481) end: generate({})
 // (op 418) NOP(481) END
 // (op 418) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(490) BEGIN
 // (op 490) Assign dim(469) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 469: v**UNALLOCATED**
 // (op 490) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag469: VGPR Value: UInt32 x 1: (unallocated)
 // (op 490) reg expression
 // (op 490) reg expression
 // (op 490) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
 // (op 490) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 490): v33
v_and_b32 v33, -8, v1 // (op 490) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 490): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 490) 
 // (op 490) BEGIN: Flatten[1]
 // (op 490) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v33:U32)U32}
 // (op 490) convert
// Allocated : 1 VGPR (Value: UInt32) (op 490): v12
v_sub_u32 v12, v1, v33 // (op 490) 
 // (op 490) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 490): v33

 // (op 490) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 490) BEGIN: Tile
 // (op 490) {Tile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 490) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 490)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 490)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 SGPR (Value: UInt32) (op 490): s14
s_lshr_b32 s14, s0, 1 // (op 490) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 490): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 490): v33
v_lshlrev_b32 v33, 4, v12 // (op 490) 
 // (op 490) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 490): v12

 // (op 490) reg expression
 // (op 490) BEGIN: Tile
 // (op 490) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 490): v12
v_lshl_add_u32 v12, s14, 8, v33 // (op 490) 
 // (op 490) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 490): v33

 // Freeing : 1 SGPR (Value: UInt32) (op 490): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 490): v24, v25
v_mov_b32 v25, 0 // (op 490) convert
v_mov_b32 v24, v12 // (op 490) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 490): v12

 // (op 490) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 490): v33
v_lshrrev_b32 v33, 3, v1 // (op 490) 
 // (op 490) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 490): v12
v_lshl_add_u32 v12, s17, 4, v33 // (op 490) 
 // (op 490) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 490): v33

 // (op 490) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 490) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 490)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 490)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 490): v26, v27
v_mul_lo_u32 v27, v12, s9 // (op 490) most significant: low of low * high
 // (op 490) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 490): v33
v_mul_hi_u32 v33, v12, s8 // (op 490) most significant: high of low * low
v_mul_lo_u32 v26, v12, s8 // (op 490) least significant: low of low * low
v_add_u32 v27, v27, v33 // (op 490) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 490): v33

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 490): v12

 // (op 490) BEGIN: Split
 // (op 490) {Split: Add({Tile: v[24:25]:U64}, v[26:27]:I64)U64}
 // (op 490) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 490)      lhs (Tile: VGPR Value: UInt64 x 1: v[24:25]) 
 // (op 490)      rhs (VGPR Value: Int64 x 1: v[26:27])
// Allocated : 2 VGPRs (Value: UInt64) (op 490): v28, v29
// Allocated : 2 SGPRs (Value: Bool64) (op 490): s0, s1
v_add_co_u32 v28, s[0:1], v24, v26 // (op 490) least significant half
v_addc_co_u32 v29, s[0:1], v25, v27, s[0:1] // (op 490) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 490): s0, s1

 // (op 490) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 490): v26, v27

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 490): v24, v25

 // (op 490) Multiply({Split: v[28:29]:U64}, 6:U32)U64
 // (op 490) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 490)           lhs (Split: VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 490)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 490) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 490): v12, v13
v_mul_lo_u32 v13, v29, 6 // (op 490) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 490): v33
v_mul_hi_u32 v33, v28, 6 // (op 490) most significant: high of low * low
v_mul_lo_u32 v12, v28, 6 // (op 490) least significant: low of low * low
v_add_u32 v13, v13, v33 // (op 490) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 490): v33

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 490): v28, v29

 // (op 490) LogicalShiftR(v[12:13]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 490): v14, v15
v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 490) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 490): v12, v13

// Allocated DataFlowTag469: 1 VGPR (Value: UInt32) (op 490): v33
v_mov_b32 v33, v14 // (op 490) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 490): v14, v15

 // (op 418) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 4:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(490) END
 // (op 418) Assign VGPR 96:U64(491) BEGIN
 // (op 491) Assign dim(470) = 96:U64
 // (op 491) Assign stride expression
 // (op 418) Assign VGPR 96:U64(491) END
 // (op 418) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(492) BEGIN
 // (op 492) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 492) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // (op 492) reg expression
 // (op 492) reg expression
 // (op 492) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 492) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 492)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 492)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 492) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 492): s0, s1
s_mul_i32 s1, s13, 6 // (op 492) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 492) least significant
// Allocated : 1 SGPR (Value: Int32) (op 492): s14
s_mul_hi_u32 s14, s12, 6 // (op 492) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 492) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 492): s14

 // (op 492) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 492) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 492)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 492)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 492): s14, s15
s_mul_i32 s15, s8, s1 // (op 492) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 492): s18
s_mul_i32 s18, s9, s0 // (op 492) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 492) least significant
// Allocated : 1 SGPR (Value: Int32) (op 492): s19
s_mul_hi_u32 s19, s8, s0 // (op 492) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 492) most significant: sum
s_add_u32 s15, s15, s19 // (op 492) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 492): s19

 // Freeing : 1 SGPR (Value: Int32) (op 492): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 492): s0, s1

 // (op 492) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 492): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 492) 
 // Freeing : 2 SGPRs (Value: Int64) (op 492): s14, s15

 // (op 492) reg expression
 // (op 492) reg expression
s_mov_b32 s44, s10 // (op 492) reg expression
s_mov_b32 s45, s11 // (op 492) reg expression
s_mov_b32 s46, s0 // (op 492) reg expression
s_mov_b32 s47, 131072 // (op 492) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 492): s0, s1

 // (op 418) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(492) END
 // (op 418) NOP(483) BEGIN
 // (op 483) generate({})
 // (op 483) end: generate({})
 // (op 418) NOP(483) END
 // (op 418) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(497) BEGIN
 // (op 497) Assign dim(474) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 497) Assign stride expression
 // (op 418) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(497) END
 // (op 418) NOP(484) BEGIN
 // (op 484) generate({})
 // (op 484) end: generate({})
 // (op 418) NOP(484) END
 // (op 418) Assign VGPR 0:U64(500) BEGIN
 // (op 500) Assign dim(476) = 0:U64
 // (op 500) Assign stride expression
 // (op 418) Assign VGPR 0:U64(500) END
 // (op 418) NOP(545) BEGIN
 // (op 545) generate({})
 // (op 545) end: generate({})
 // (op 418) NOP(545) END
 // (op 418) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32})U32}(554) BEGIN
 // (op 554) Assign dim(488) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32})U32}
 // tag 488: v**UNALLOCATED**
 // (op 554) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32})U32} into DataFlowTag488: VGPR Value: UInt32 x 1: (unallocated)
 // (op 554) reg expression
 // (op 554) reg expression
 // (op 554) reg expression
 // (op 554) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 554) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 554): v12
v_lshrrev_b32 v12, 2, v1 // (op 554) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 554): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 554) 
 // (op 554) BEGIN: Tile
 // (op 554) convert
// Allocated : 1 VGPR (Value: UInt32) (op 554): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 554) 
 // (op 554) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 554): v12

 // (op 554) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 554) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 554): s14
s_lshr_b32 s14, s0, 1 // (op 554) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 554): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 554): v12
v_and_b32 v12, -4, v1 // (op 554) 
 // (op 554) reg expression
 // (op 554) BEGIN: Flatten[1]Tile
 // (op 554) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
 // (op 554) convert
// Allocated : 1 VGPR (Value: UInt32) (op 554): v14
v_sub_u32 v14, v1, v12 // (op 554) 
 // (op 554) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 554): v12

 // (op 554) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 554): v12
v_lshl_add_u32 v12, s14, 3, v14 // (op 554) 
 // (op 554) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 554): v14

 // Freeing : 1 SGPR (Value: UInt32) (op 554): s14

 // (op 554) BEGIN: Split
// Allocated DataFlowTag488: 1 VGPR (Value: UInt32) (op 554): v14
v_mul_lo_u32 v14, v13, s20 // (op 554) 
v_add_u32 v14, v14, v12 // (op 554) 
 // (op 554) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 554): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 554): v13

 // (op 418) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 5:U32, LogicalShiftR(DataFlowTag(204)U32, 2:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(204)U32, BitwiseAnd(DataFlowTag(204)U32, -4:I)U32)U32})U32})U32}(554) END
 // (op 418) Assign VGPR 4:U64(555) BEGIN
 // (op 555) Assign dim(489) = 4:U64
 // (op 555) Assign stride expression
 // (op 418) Assign VGPR 4:U64(555) END
 // (op 418) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(556) BEGIN
 // (op 556) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 556) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // (op 556) reg expression
 // (op 556) reg expression
 // (op 556) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 556) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 556)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 556)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 556): s0, s1
s_mul_i32 s1, s20, s7 // (op 556) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 556): s14
s_mul_i32 s14, s21, s6 // (op 556) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 556) least significant
// Allocated : 1 SGPR (Value: Int32) (op 556): s15
s_mul_hi_u32 s15, s20, s6 // (op 556) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 556) most significant: sum
s_add_u32 s1, s1, s15 // (op 556) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 556): s15

 // Freeing : 1 SGPR (Value: Int32) (op 556): s14

 // (op 556) reg expression
 // (op 556) reg expression
s_mov_b32 s48, s22 // (op 556) reg expression
s_mov_b32 s49, s23 // (op 556) reg expression
s_mov_b32 s50, s0 // (op 556) reg expression
s_mov_b32 s51, 131072 // (op 556) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 556): s0, s1

 // (op 418) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(556) END
 // (op 418) NOP(547) BEGIN
 // (op 547) generate({})
 // (op 547) end: generate({})
 // (op 418) NOP(547) END
 // (op 418) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(561) BEGIN
 // (op 561) Assign dim(493) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 561) Assign stride expression
 // (op 418) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(561) END
 // (op 418) NOP(548) BEGIN
 // (op 548) generate({})
 // (op 548) end: generate({})
 // (op 418) NOP(548) END
 // (op 418) Assign VGPR 1:U64(564) BEGIN
 // (op 564) Assign dim(495) = 1:U64
 // (op 564) Assign stride expression
 // (op 418) Assign VGPR 1:U64(564) END
 // (op 418) NOP(609) BEGIN
 // (op 609) generate({})
 // (op 609) end: generate({})
 // (op 418) NOP(609) END
 // (op 418) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(618) BEGIN
 // (op 618) Assign dim(507) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 507: v**UNALLOCATED**
 // (op 618) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag507: VGPR Value: UInt32 x 1: (unallocated)
 // (op 618) reg expression
 // (op 618) reg expression
 // (op 618) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
 // (op 618) BEGIN: Flatten[1]
 // (op 618) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 618): v12
v_and_b32 v12, 63, v1 // (op 618) 
 // (op 618) END: Flatten[1]
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 618): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 618) 
 // (op 618) LogicalShiftR({Flatten[1]: v12:U32}, 4:U32)U32
 // (op 618) convert
// Allocated : 1 VGPR (Value: UInt32) (op 618): v13
v_lshrrev_b32 v13, 4, v12 // (op 618) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 618): v12

 // (op 618) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 618) BEGIN: Flatten[1]Tile
 // (op 618) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 SGPR (Value: UInt32) (op 618): s14
s_lshr_b32 s14, s0, 1 // (op 618) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 618): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 618): v12
v_and_b32 v12, 15, v1 // (op 618) 
 // (op 618) END: Flatten[1]Tile
 // (op 618) reg expression
 // (op 618) BEGIN: Tile
 // (op 618) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 618): v15
v_lshl_add_u32 v15, s14, 3, v13 // (op 618) 
 // (op 618) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 618): v13

 // Freeing : 1 SGPR (Value: UInt32) (op 618): s14

 // (op 618) convert
 // (op 618) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 618): v13
v_lshl_add_u32 v13, s17, 4, v12 // (op 618) 
 // (op 618) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 618): v12

 // (op 618) Multiply({Tile: v13:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 618): v12
v_mul_lo_u32 v12, v13, s24 // (op 618) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 618): v13

 // (op 618) BEGIN: Split
 // (op 618) {Split: Add({Tile: v15:U32}, v12:U32)U32}
// Allocated DataFlowTag507: 1 VGPR (Value: UInt32) (op 618): v13
v_add_u32 v13, v15, v12 // (op 618) 
 // (op 618) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 618): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 618): v15

 // (op 418) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(618) END
 // (op 418) Assign VGPR 4:U64(619) BEGIN
 // (op 619) Assign dim(508) = 4:U64
 // (op 619) Assign stride expression
 // (op 418) Assign VGPR 4:U64(619) END
 // (op 418) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(620) BEGIN
 // (op 620) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 620) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // (op 620) reg expression
 // (op 620) reg expression
 // (op 620) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 620) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 620)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 620)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 620): s0, s1
s_mul_i32 s1, s24, s13 // (op 620) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 620): s14
s_mul_i32 s14, s25, s12 // (op 620) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 620) least significant
// Allocated : 1 SGPR (Value: Int32) (op 620): s15
s_mul_hi_u32 s15, s24, s12 // (op 620) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 620) most significant: sum
s_add_u32 s1, s1, s15 // (op 620) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 620): s15

 // Freeing : 1 SGPR (Value: Int32) (op 620): s14

 // (op 620) reg expression
 // (op 620) reg expression
s_mov_b32 s52, s26 // (op 620) reg expression
s_mov_b32 s53, s27 // (op 620) reg expression
s_mov_b32 s54, s0 // (op 620) reg expression
s_mov_b32 s55, 131072 // (op 620) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 620): s0, s1

 // (op 418) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(620) END
 // (op 418) NOP(611) BEGIN
 // (op 611) generate({})
 // (op 611) end: generate({})
 // (op 418) NOP(611) END
 // (op 418) Assign VGPR 4:U64(625) BEGIN
 // (op 625) Assign dim(512) = 4:U64
 // (op 625) Assign stride expression
 // (op 418) Assign VGPR 4:U64(625) END
 // (op 418) NOP(612) BEGIN
 // (op 612) generate({})
 // (op 612) end: generate({})
 // (op 418) NOP(612) END
 // (op 418) Assign VGPR 1:U64(628) BEGIN
 // (op 628) Assign dim(514) = 1:U64
 // (op 628) Assign stride expression
 // (op 418) Assign VGPR 1:U64(628) END
 // (op 418) NOP(706) BEGIN
 // (op 706) generate({})
 // (op 706) end: generate({})
 // (op 418) NOP(706) END
 // (op 418) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(711) BEGIN
 // (op 711) Assign dim(536) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 536: v**UNALLOCATED**
 // (op 711) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag536: VGPR Value: UInt32 x 1: (unallocated)
 // (op 711) reg expression
 // (op 711) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 711): v12
v_and_b32 v12, -4, v1 // (op 711) 
 // (op 711) BEGIN: Tile[1]
 // (op 711) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 711): v15
v_sub_u32 v15, v1, v12 // (op 711) 
 // (op 711) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 711): v12

 // (op 711) BEGIN: Flatten
 // (op 711) {Flatten: ShiftL({Tile[1]: v15:U32}, 5:U32)U32}
 // (op 711) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 711)         value (Tile[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 711)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 711): v12
v_lshlrev_b32 v12, 5, v15 // (op 711) 
 // (op 711) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 711): v15

 // (op 711) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 711): v15
v_lshrrev_b32 v15, 2, v1 // (op 711) 
 // (op 711) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 711): v16
v_lshl_add_u32 v16, v15, 7, v12 // (op 711) 
 // (op 711) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 711): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 711): v15

 // (op 711) Multiply({Flatten: v16:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 711): v12
v_mul_lo_u32 v12, v16, 6 // (op 711) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 711): v16

 // (op 711) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag536: 1 VGPR (Value: UInt32) (op 711): v15
v_lshrrev_b32 v15, 3, v12 // (op 711) 
 // Freeing : 1 VGPR (Value: UInt32) (op 711): v12

 // (op 418) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(711) END
 // (op 418) Assign VGPR 96:U32(712) BEGIN
 // (op 712) Assign dim(537) = 96:U32
 // (op 712) Assign stride expression
 // (op 418) Assign VGPR 96:U32(712) END
 // (op 418) NOP(707) BEGIN
 // (op 707) generate({})
 // (op 707) end: generate({})
 // (op 418) NOP(707) END
 // (op 418) Assign VGPR 0:U32(716) BEGIN
 // (op 716) Assign dim(539) = 0:U32
 // (op 716) Assign stride expression
 // (op 418) Assign VGPR 0:U32(716) END
 // (op 418) NOP(759) BEGIN
 // (op 759) generate({})
 // (op 759) end: generate({})
 // (op 418) NOP(759) END
 // (op 418) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(764) BEGIN
 // (op 764) Assign dim(552) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 552: v**UNALLOCATED**
 // (op 764) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag552: VGPR Value: UInt32 x 1: (unallocated)
 // (op 764) reg expression
 // (op 764) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 764): v12
v_and_b32 v12, -8, v1 // (op 764) 
 // (op 764) BEGIN: Tile[1]
 // (op 764) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 764): v16
v_sub_u32 v16, v1, v12 // (op 764) 
 // (op 764) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 764): v12

 // (op 764) BEGIN: Flatten
 // (op 764) {Flatten: ShiftL({Tile[1]: v16:U32}, 4:U32)U32}
 // (op 764) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 764)         value (Tile[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 764)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 764): v12
v_lshlrev_b32 v12, 4, v16 // (op 764) 
 // (op 764) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 764): v16

 // (op 764) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 764): v16
v_lshrrev_b32 v16, 3, v1 // (op 764) 
 // (op 764) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 764): v17
v_lshl_add_u32 v17, v16, 7, v12 // (op 764) 
 // (op 764) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 764): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 764): v16

 // (op 764) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 764): v12
v_mul_lo_u32 v12, v17, 6 // (op 764) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 764): v17

 // (op 764) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag552: 1 VGPR (Value: UInt32) (op 764): v16
v_lshrrev_b32 v16, 3, v12 // (op 764) 
 // Freeing : 1 VGPR (Value: UInt32) (op 764): v12

 // (op 418) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(764) END
 // (op 418) Assign VGPR 96:U32(765) BEGIN
 // (op 765) Assign dim(553) = 96:U32
 // (op 765) Assign stride expression
 // (op 418) Assign VGPR 96:U32(765) END
 // (op 418) NOP(760) BEGIN
 // (op 760) generate({})
 // (op 760) end: generate({})
 // (op 418) NOP(760) END
 // (op 418) Assign VGPR 0:U32(769) BEGIN
 // (op 769) Assign dim(555) = 0:U32
 // (op 769) Assign stride expression
 // (op 418) Assign VGPR 0:U32(769) END
 // (op 418) NOP(812) BEGIN
 // (op 812) generate({})
 // (op 812) end: generate({})
 // (op 418) NOP(812) END
 // (op 418) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}(817) BEGIN
 // (op 817) Assign dim(568) = {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}
 // tag 568: v**UNALLOCATED**
 // (op 817) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32} into DataFlowTag568: VGPR Value: UInt32 x 1: (unallocated)
 // (op 817) reg expression
 // (op 817) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 817): v12
v_and_b32 v12, -4, v1 // (op 817) 
 // (op 817) BEGIN: Tile[1]Flatten
 // (op 817) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 817): v17
v_sub_u32 v17, v1, v12 // (op 817) 
 // (op 817) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 817): v12

 // (op 817) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 817): v12
v_and_b32 v12, 4294967292, v1 // (op 817) 
 // (op 817) BEGIN: Flatten
 // (op 817) {Flatten: Add(v12:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag568: 1 VGPR (Value: UInt32) (op 817): v18
v_add_u32 v18, v12, v17 // (op 817) 
 // (op 817) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 817): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 817): v12

 // (op 418) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(219)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(219)U32, BitwiseAnd(DataFlowTag(219)U32, -4:I)U32)U32})U32}(817) END
 // (op 418) Assign VGPR 4:U32(818) BEGIN
 // (op 818) Assign dim(569) = 4:U32
 // (op 818) Assign stride expression
 // (op 418) Assign VGPR 4:U32(818) END
 // (op 418) NOP(813) BEGIN
 // (op 813) generate({})
 // (op 813) end: generate({})
 // (op 418) NOP(813) END
 // (op 418) Assign VGPR 1:U32(822) BEGIN
 // (op 822) Assign dim(571) = 1:U32
 // (op 822) Assign stride expression
 // (op 418) Assign VGPR 1:U32(822) END
 // (op 418) ForLoopOp KLoopTail: LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL(161) BEGIN
 // (op 161) Initialize For Loop
 // (op 161) generate({162})
 // (op 161) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32(162) BEGIN
 // (op 162) Assign dim(414) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32
 // tag 414: s**UNALLOCATED**
 // (op 162) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag414: SGPR Value: UInt32 x 1: (unallocated)
 // (op 162) reg expression
 // (op 162) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 162): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 162) 
 // (op 162) convert
 // (op 162) BitwiseAnd({ convertForwardSubset: s0:U32}, 4294967294:U32)U32
// Allocated DataFlowTag414: 1 SGPR (Value: UInt32) (op 162): s14
s_and_b32 s14, s0, 4294967294 // (op 162) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 162): s0, s1

 // (op 161) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32(162) END
 // (op 161) end: generate({162})
 // (op 161) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 161) 
 // (op 161) Generate LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 161) LessThan({DataFlowTag414: s14:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32)BL
 // (op 161) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 161): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 161) 
// Allocated : 1 SGPR (Value: UInt32) (op 161): s15
s_mov_b32 s15, s0 // (op 161) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 161): s0, s1

s_cmp_lt_u32 s14, s15 // (op 161) 
 // Freeing : 1 SGPR (Value: UInt32) (op 161): s15

 // Freeing : 1 VGPR (Value: E8M0) (op 351): v23

 // Freeing : 3 VGPRs (Value: FP6x16) (op 355): v30, v31, v32

 // Freeing : 6 VGPRs (Value: FP6x16) (op 358): v34, v35, v36, v37, v38, v39

 // Freeing : 1 VGPR (Value: E8M0) (op 361): v10

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 161) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 161) 
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 161) 
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161:
 // (op 161) 
 // (op 161) generate({163, 206})
 // (op 161) Assign SGPR Add(DataFlowTag(414)U32, 1:U32)U32(163) BEGIN
 // (op 163) Assign dim(414) = Add(DataFlowTag(414)U32, 1:U32)U32
 // (op 163) Generate Add(DataFlowTag(414)U32, 1:U32)U32 into DataFlowTag414: SGPR Value: UInt32 x 1: s14
 // (op 163) reg expression
 // (op 163) Add({DataFlowTag414: s14:U32}, 1:U32)U32
 // (op 161) SetCoordinate(206) BEGIN
 // (op 206) generate({})
 // (op 206) end: generate({})
 // (op 206) generate({166})
 // (op 206) LoadTiled Value: FP6x16(166) BEGIN
 // (op 166) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 166) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 166) Generate 0:U64 into nullptr
 // tag 416: v**UNALLOCATED**
 // (op 166) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 454: v**UNALLOCATED**
s_add_u32 s14, s14, 1 // (op 163) 
 // (op 161) Assign SGPR Add(DataFlowTag(414)U32, 1:U32)U32(163) END
// Allocated Offset454: 1 VGPR (Value: UInt32) (op 166): v10
v_mov_b32 v10, v11 // (op 166) 
 // (op 166) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = Offset454: VGPR Value: UInt32 x 1: v10
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
 // 	info.bufDesc = DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 166) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 166): v20, v21, v22, v23, v24, v25
buffer_load_dwordx4 v[20:23], v10, s[40:43], 0 offen // (op 166) Load value
// VMEM: Expected complete at 4436 (current 3956)
buffer_load_dwordx2 v[24:25], v10, s[40:43], 0 offen offset:16 // (op 166) Load value
// VMEM: Expected complete at 4437 (current 3957)
 // (op 206) LoadTiled Value: FP6x16(166) END
 // (op 206) Deallocate{}(1067) BEGIN
 // (op 1067) Deallocate 454
 // Freeing Offset454: 1 VGPR (Value: UInt32) (op 166): v10

 // Deleting tag 454
 // (op 1067) Deallocate 456
 // Deleting tag 456
 // (op 206) Deallocate{}(1067) END
 // (op 206) Assign VGPR Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32(413) BEGIN
 // (op 413) Assign dim(450) = Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32
 // (op 413) Generate Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32 into DataFlowTag450: VGPR Value: UInt32 x 1: v11
 // (op 413) reg expression
 // (op 413) Add({DataFlowTag450: v11:U32}, 96:U32)U32
v_add_u32 v11, 96, v11 // (op 413) 
 // (op 206) Assign VGPR Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32(413) END
 // (op 206) end: generate({166})
 // (op 161) SetCoordinate(206) END
 // (op 161) SetCoordinate(214) BEGIN
 // (op 214) generate({})
 // (op 214) end: generate({})
 // (op 214) generate({179})
 // (op 214) LoadTiled Value: FP6x16(179) BEGIN
 // (op 179) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 179) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 179) Generate 0:U64 into nullptr
 // tag 418: v**UNALLOCATED**
 // (op 179) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 473: v**UNALLOCATED**
 // (op 161) Barrier(990) BEGIN
s_barrier  // (op 990) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 161) Barrier(990) END
// Allocated Offset473: 1 VGPR (Value: UInt32) (op 179): v10
v_mov_b32 v10, v33 // (op 179) 
 // (op 179) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset473: VGPR Value: UInt32 x 1: v10
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 179) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 161) StoreLDSTile Value: FP6x16(167) BEGIN
 // (op 167) GEN: StoreLDSTile
 // (op 167) GEN: storeMacroTileLDS OP 167 LDS 16 MacroTile 416
 // (op 167) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 167) Generate 0:U32 into nullptr
 // (op 167) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[20:25]
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 167) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[20:25]
 // 	info.rowOffsetReg = DataFlowTag536: VGPR Value: UInt32 x 1: v15
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 167) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: FP6x16) (op 179): v26, v27, v28
buffer_load_dwordx3 v[26:28], v10, s[44:47], 0 offen // (op 179) Load value
// VMEM: Expected complete at 4441 (current 3961)
 // (op 214) LoadTiled Value: FP6x16(179) END
 // (op 214) Deallocate{}(1079) BEGIN
 // (op 1079) Deallocate 475
 // Deleting tag 475
 // (op 1079) Deallocate 473
 // Freeing Offset473: 1 VGPR (Value: UInt32) (op 179): v10

 // Deleting tag 473
 // (op 214) Deallocate{}(1079) END
 // (op 214) Assign VGPR Add(DataFlowTag(469)U32, Convert(DataFlowTag(470)U64)U32)U32(482) BEGIN
 // (op 482) Assign dim(469) = Add(DataFlowTag(469)U32, Convert(DataFlowTag(470)U64)U32)U32
 // (op 482) Generate Add(DataFlowTag(469)U32, Convert(DataFlowTag(470)U64)U32)U32 into DataFlowTag469: VGPR Value: UInt32 x 1: v33
 // (op 482) reg expression
 // (op 482) Add({DataFlowTag469: v33:U32}, 96:U32)U32
v_add_u32 v33, 96, v33 // (op 482) 
 // (op 214) Assign VGPR Add(DataFlowTag(469)U32, Convert(DataFlowTag(470)U64)U32)U32(482) END
 // (op 214) end: generate({179})
 // (op 161) SetCoordinate(214) END
s_waitcnt vmcnt(2)
ds_write_b128 v15, v[20:23] offset:1536 // (op 167) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 4013 (current 3918)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 161) SetCoordinate(219) BEGIN
 // (op 219) generate({})
 // (op 219) end: generate({})
 // (op 219) generate({180})
 // (op 219) LoadTiled Value: E8M0(180) BEGIN
 // (op 180) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 180) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 180) Generate 1:U64 into nullptr
 // tag 420: v**UNALLOCATED**
 // (op 180) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 492: v**UNALLOCATED**
// Allocated Offset492: 1 VGPR (Value: UInt32) (op 180): v10
v_mov_b32 v10, v14 // (op 180) 
 // (op 180) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset492: VGPR Value: UInt32 x 1: v10
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 180) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 180): v12
buffer_load_ubyte v12, v10, s[48:51], 0 offen offset: 0 // (op 180) Load value
// VMEM: Expected complete at 4920 (current 4440)
 // (op 219) LoadTiled Value: E8M0(180) END
 // (op 219) Deallocate{}(1081) BEGIN
 // (op 1081) Deallocate 494
 // Deleting tag 494
 // (op 1081) Deallocate 492
 // Freeing Offset492: 1 VGPR (Value: UInt32) (op 180): v10

 // Deleting tag 492
 // (op 219) Deallocate{}(1081) END
 // (op 219) Assign VGPR Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32(546) BEGIN
 // (op 546) Assign dim(488) = Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32
 // (op 546) Generate Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32 into DataFlowTag488: VGPR Value: UInt32 x 1: v14
 // (op 546) reg expression
 // (op 546) Add({DataFlowTag488: v14:U32}, 4:U32)U32
v_add_u32 v14, 4, v14 // (op 546) 
 // (op 219) Assign VGPR Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32(546) END
 // (op 219) end: generate({180})
 // (op 161) SetCoordinate(219) END
s_waitcnt vmcnt(2)
ds_write_b64 v15, v[24:25] offset:1552 // (op 167) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 4018 (current 3923)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 161) StoreLDSTile Value: FP6x16(167) END
 // (op 161) Deallocate{}(1069) BEGIN
 // (op 1069) Deallocate 416
 // Freeing : 6 VGPRs (Value: FP6x16) (op 166): v20, v21, v22, v23, v24, v25

 // Deleting tag 416
 // (op 1069) Deallocate 538
 // Deleting tag 538
 // (op 161) Deallocate{}(1069) END
 // (op 161) Barrier(993) BEGIN
 // (op 161) Barrier(1003) BEGIN
s_barrier  // (op 1003) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 161) Barrier(1003) END
s_waitcnt lgkmcnt(0)
s_barrier  // (op 993) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 161) Barrier(993) END
 // (op 161) StoreLDSTile Value: FP6x16(177) BEGIN
 // (op 177) GEN: StoreLDSTile
 // (op 177) GEN: storeMacroTileLDS OP 177 LDS 24 MacroTile 418
 // (op 177) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 177) Generate 0:U32 into nullptr
 // (op 177) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[26:28]
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

 // (op 177) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[26:28]
 // 	info.rowOffsetReg = DataFlowTag552: VGPR Value: UInt32 x 1: v16
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
 // (op 177) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 161) SetCoordinate(168) BEGIN
 // (op 168) generate({})
 // (op 168) end: generate({})
 // (op 168) generate({169})
 // (op 168) LoadLDSTile{Value: FP6x16}(169) BEGIN
 // (op 169) GEN: loadMacroTileWAVELDS OP 169 LDS 16 WaveTile 99
 // (op 169) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 169) Generate 0:U32 into nullptr
 // tag 422: v**UNALLOCATED**
 // (op 169) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 169) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag544: VGPR Value: UInt32 x 1: v3
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1536
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 169) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 169): v20, v21, v22, v23, v24, v25
ds_read_b128 v[20:23], v3 offset:1536 // (op 169) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 4116 (current 4021)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
ds_read_b64 v[24:25], v3 offset:1552 // (op 169) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 4117 (current 4022)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 1536, Size: 3072)
 // (op 168) LoadLDSTile{Value: FP6x16}(169) END
 // (op 168) end: generate({169})
 // (op 161) SetCoordinate(168) END
s_waitcnt vmcnt(1)
ds_write_b96 v16, v[26:28] // (op 177) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 4119 (current 4024)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 161) StoreLDSTile Value: FP6x16(177) END
 // (op 161) Deallocate{}(1075) BEGIN
 // (op 1075) Deallocate 554
 // Deleting tag 554
 // (op 1075) Deallocate 418
 // Freeing : 3 VGPRs (Value: FP6x16) (op 179): v26, v27, v28

 // Deleting tag 418
 // (op 161) Deallocate{}(1075) END
 // (op 161) Barrier(1007) BEGIN
 // (op 161) Barrier(1017) BEGIN
s_barrier  // (op 1017) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 161) Barrier(1017) END
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1007) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 161) Barrier(1007) END
 // (op 161) StoreLDSTile Value: E8M0(178) BEGIN
 // (op 178) GEN: StoreLDSTile
 // (op 178) GEN: storeMacroTileLDS OP 178 LDS 32 MacroTile 420
 // (op 178) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 178) Generate 1:U32 into nullptr
 // (op 178) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
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

 // (op 178) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = DataFlowTag568: VGPR Value: UInt32 x 1: v18
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 178) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 161) SetCoordinate(170) BEGIN
 // (op 170) generate({})
 // (op 170) end: generate({})
 // (op 170) generate({171})
 // (op 170) LoadLDSTile{Value: FP6x16}(171) BEGIN
 // (op 171) GEN: loadMacroTileWAVELDS OP 171 LDS 24 WaveTile 165
 // (op 171) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 171) Generate 0:U32 into nullptr
 // tag 424: v**UNALLOCATED**
 // (op 171) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
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

 // (op 171) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v4
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
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
 // (op 171) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 171): v26, v27, v28, v29, v30, v31
ds_read_b128 v[26:29], v4 // (op 171) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 4217 (current 4122)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
ds_read_b64 v[30:31], v4 offset:16 // (op 171) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 4218 (current 4123)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 170) LoadLDSTile{Value: FP6x16}(171) END
 // (op 170) end: generate({171})
 // (op 161) SetCoordinate(170) END
s_waitcnt vmcnt(0)
ds_write_b8 v18, v12 offset:4608 // (op 178) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 4220 (current 4125)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 161) StoreLDSTile Value: E8M0(178) END
 // (op 161) Deallocate{}(1077) BEGIN
 // (op 1077) Deallocate 570
 // Deleting tag 570
 // (op 1077) Deallocate 420
 // Freeing : 1 VGPR (Value: E8M0) (op 180): v12

 // Deleting tag 420
 // (op 161) Deallocate{}(1077) END
 // (op 161) Barrier(1021) BEGIN
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1021) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 161) Barrier(1021) END
 // (op 161) SetCoordinate(172) BEGIN
 // (op 172) generate({})
 // (op 172) end: generate({})
 // (op 172) generate({173})
 // (op 172) LoadLDSTile{Value: E8M0}(173) BEGIN
 // (op 173) GEN: loadMacroTileWAVELDS OP 173 LDS 32 WaveTile 231
 // (op 173) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 173) Generate 1:U32 into nullptr
 // tag 426: v**UNALLOCATED**
 // (op 173) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
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

 // (op 173) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag576: VGPR Value: UInt32 x 1: v0
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 173) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 173): v10
ds_read_u8 v10, v0 offset:4608 // (op 173) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 4318 (current 4223)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4608, Size: 128)
 // (op 172) LoadLDSTile{Value: E8M0}(173) END
 // (op 172) end: generate({173})
 // (op 161) SetCoordinate(172) END
 // (op 161) SetCoordinate(174) BEGIN
 // (op 174) generate({})
 // (op 174) end: generate({})
 // (op 174) generate({211})
 // (op 174) SetCoordinate(211) BEGIN
 // (op 211) generate({})
 // (op 211) end: generate({})
 // (op 211) generate({175})
 // (op 211) LoadTiled Value: E8M0(175) BEGIN
 // (op 175) GEN: loadMacroTileWAVE OP175 WaveTile 261
 // (op 175) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 175) Generate 1:U64 into nullptr
 // tag 428: v**UNALLOCATED**
 // (op 175) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 511: v**UNALLOCATED**
// Allocated Offset511: 1 VGPR (Value: UInt32) (op 175): v12
v_mov_b32 v12, v13 // (op 175) 
 // (op 175) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset511: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 175) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 175): v17
buffer_load_ubyte v17, v12, s[52:55], 0 offen offset: 0 // (op 175) Load value
// VMEM: Expected complete at 5407 (current 4927)
 // (op 211) LoadTiled Value: E8M0(175) END
 // (op 211) Deallocate{}(1071) BEGIN
 // (op 1071) Deallocate 511
 // Freeing Offset511: 1 VGPR (Value: UInt32) (op 175): v12

 // Deleting tag 511
 // (op 1071) Deallocate 513
 // Deleting tag 513
 // (op 211) Deallocate{}(1071) END
 // (op 211) Assign VGPR Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32(610) BEGIN
 // (op 610) Assign dim(507) = Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32
 // (op 610) Generate Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32 into DataFlowTag507: VGPR Value: UInt32 x 1: v13
 // (op 610) reg expression
 // (op 610) Add({DataFlowTag507: v13:U32}, 4:U32)U32
v_add_u32 v13, 4, v13 // (op 610) 
 // (op 211) Assign VGPR Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32(610) END
 // (op 211) end: generate({175})
 // (op 174) SetCoordinate(211) END
 // (op 174) end: generate({211})
 // (op 161) SetCoordinate(174) END
 // (op 161) Multiply(176) BEGIN
 // (op 176) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile16xFP6, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v10, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 176) 
 // (op 161) Multiply(176) END
 // (op 161) Deallocate{}(1073) BEGIN
 // (op 1073) Deallocate 422
 // Freeing : 6 VGPRs (Value: FP6x16) (op 169): v20, v21, v22, v23, v24, v25

 // Deleting tag 422
 // (op 1073) Deallocate 424
 // Freeing : 6 VGPRs (Value: FP6x16) (op 171): v26, v27, v28, v29, v30, v31

 // Deleting tag 424
 // (op 1073) Deallocate 426
 // Freeing : 1 VGPR (Value: E8M0) (op 173): v10

 // Deleting tag 426
 // (op 1073) Deallocate 428
 // Freeing : 1 VGPR (Value: E8M0) (op 175): v17

 // Deleting tag 428
 // (op 161) Deallocate{}(1073) END
 // (op 161) end: generate({163, 206})
 // (op 161) For Loop Increment
 // (op 161) generate({})
 // (op 161) end: generate({})
 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 if true)
 // (op 161) Generate LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 161) LessThan({DataFlowTag414: s14:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32)BL
 // (op 161) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 161): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 161) 
// Allocated : 1 SGPR (Value: UInt32) (op 161): s15
s_mov_b32 s15, s0 // (op 161) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 161): s0, s1

s_cmp_lt_u32 s14, s15 // (op 161) 
 // Freeing : 1 SGPR (Value: UInt32) (op 161): s15

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 161) 
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ForLoopBottom_KLoopTail_161:
 // (op 161) 
 // (op 161) Unlock For Loop
 // Unlock None
 // (op 418) ForLoopOp KLoopTail: LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL(161) END
 // (op 418) Deallocate{}(1065) BEGIN
 // (op 1065) Deallocate 450
 // Freeing DataFlowTag450: 1 VGPR (Value: UInt32) (op 423): v11

 // Deleting tag 450
 // (op 1065) Deallocate 451
 // Deleting tag 451
 // (op 1065) Deallocate 455
 // Deleting tag 455
 // (op 1065) Deallocate 507
 // Freeing DataFlowTag507: 1 VGPR (Value: UInt32) (op 618): v13

 // Deleting tag 507
 // (op 1065) Deallocate 457
 // Deleting tag 457
 // (op 1065) Deallocate 495
 // Deleting tag 495
 // (op 1065) Deallocate 568
 // Freeing DataFlowTag568: 1 VGPR (Value: UInt32) (op 817): v18

 // Deleting tag 568
 // (op 1065) Deallocate 470
 // Deleting tag 470
 // (op 1065) Deallocate 469
 // Freeing DataFlowTag469: 1 VGPR (Value: UInt32) (op 490): v33

 // Deleting tag 469
 // (op 1065) Deallocate 569
 // Deleting tag 569
 // (op 1065) Deallocate 553
 // Deleting tag 553
 // (op 1065) Deallocate 414
 // Freeing DataFlowTag414: 1 SGPR (Value: UInt32) (op 162): s14

 // Deleting tag 414
 // (op 1065) Deallocate 571
 // Deleting tag 571
 // (op 1065) Deallocate 536
 // Freeing DataFlowTag536: 1 VGPR (Value: UInt32) (op 711): v15

 // Deleting tag 536
 // (op 1065) Deallocate 476
 // Deleting tag 476
 // (op 1065) Deallocate 474
 // Deleting tag 474
 // (op 1065) Deallocate 488
 // Freeing DataFlowTag488: 1 VGPR (Value: UInt32) (op 554): v14

 // Deleting tag 488
 // (op 1065) Deallocate 489
 // Deleting tag 489
 // (op 1065) Deallocate 508
 // Deleting tag 508
 // (op 1065) Deallocate 514
 // Deleting tag 514
 // (op 1065) Deallocate 537
 // Deleting tag 537
 // (op 1065) Deallocate 539
 // Deleting tag 539
 // (op 1065) Deallocate 493
 // Deleting tag 493
 // (op 1065) Deallocate 552
 // Freeing DataFlowTag552: 1 VGPR (Value: UInt32) (op 764): v16

 // Deleting tag 552
 // (op 1065) Deallocate 555
 // Deleting tag 555
 // (op 1065) Deallocate 512
 // Deleting tag 512
 // (op 418) Deallocate{}(1065) END
 // (op 418) end: generate({412})
 // (op 418) Unlock Scope 418
 // Unlock None
 // (op 202) Scope(418) END
 // (op 202) end: generate({418})
 // (op 201) SetCoordinate(202) END
 // (op 201) end: generate({202})
 // (op 310) SetCoordinate(201) END
 // (op 310) Deallocate{}(1083) BEGIN
 // (op 1083) Deallocate 138
 // Deleting tag 138
 // (op 1083) Deallocate 395
 // Deleting tag 395
 // (op 1083) Deallocate 72
 // Deleting tag 72
 // (op 1083) Deallocate 472
 // Freeing DataFlowTag472: 4 SGPRs (Buffer: None) (op 515): s44, s45, s46, s47

 // Deleting tag 472
 // (op 1083) Deallocate 16
 // Deleting tag 16
 // (op 1083) Deallocate 397
 // Deleting tag 397
 // (op 1083) Deallocate 491
 // Freeing DataFlowTag491: 4 SGPRs (Buffer: None) (op 579): s48, s49, s50, s51

 // Deleting tag 491
 // (op 1083) Deallocate 453
 // Freeing DataFlowTag453: 4 SGPRs (Buffer: None) (op 451): s40, s41, s42, s43

 // Deleting tag 453
 // (op 1083) Deallocate 393
 // Deleting tag 393
 // (op 1083) Deallocate 399
 // Deleting tag 399
 // (op 1083) Deallocate 510
 // Freeing DataFlowTag510: 4 SGPRs (Buffer: None) (op 643): s52, s53, s54, s55

 // Deleting tag 510
 // (op 1083) Deallocate 2
 // Deleting tag 2
 // (op 1083) Deallocate 578
 // Deleting tag 578
 // (op 1083) Deallocate 562
 // Deleting tag 562
 // (op 1083) Deallocate 546
 // Deleting tag 546
 // (op 1083) Deallocate 32
 // Deleting tag 32
 // (op 1083) Deallocate 87
 // Deleting tag 87
 // (op 1083) Deallocate 219
 // Deleting tag 219
 // (op 1083) Deallocate 24
 // Deleting tag 24
 // (op 1083) Deallocate 153
 // Deleting tag 153
 // (op 1083) Deallocate 3
 // Deleting tag 3
 // (op 1083) Deallocate 260
 // Deleting tag 260
 // (op 1083) Deallocate 204
 // Deleting tag 204
 // (op 1083) Deallocate 1
 // Deleting tag 1
 // (op 1083) Deallocate 4
 // Deleting tag 4
 // (op 310) Deallocate{}(1083) END
 // (op 310) Deallocate{Tensor_2_stride_1_3, Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_4_pointer, Tensor_7_pointer, Tensor_0_size_1_14, Tensor_7_stride_1_8, Tensor_2_pointer}(1200) BEGIN
 // (op 1200) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 1200) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1200) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1200) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1200) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1200) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 1200) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1200) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1200) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1200) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 310) Deallocate{Tensor_2_stride_1_3, Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_4_pointer, Tensor_7_pointer, Tensor_0_size_1_14, Tensor_7_stride_1_8, Tensor_2_pointer}(1200) END
 // (op 310) Barrier(1085) BEGIN
s_barrier  // (op 1085) 
 // (op 310) Barrier(1085) END
 // (op 310) end: generate({315})
 // (op 310) Unlock Scope 310
 // Unlock None
 // (op 1) Scope(310) END
 // (op 1) Deallocate{}(1121) BEGIN
 // (op 1121) Deallocate 547
 // Deleting tag 547
 // (op 1121) Deallocate 576
 // Freeing DataFlowTag576: 1 VGPR (Value: UInt32) (op 845): v0

 // Deleting tag 576
 // (op 1121) Deallocate 561
 // Deleting tag 561
 // (op 1121) Deallocate 595
 // Deleting tag 595
 // (op 1121) Deallocate 605
 // Deleting tag 605
 // (op 1121) Deallocate 544
 // Freeing DataFlowTag544: 1 VGPR (Value: UInt32) (op 739): v3

 // Deleting tag 544
 // (op 1121) Deallocate 563
 // Deleting tag 563
 // (op 1121) Deallocate 577
 // Deleting tag 577
 // (op 1121) Deallocate 613
 // Deleting tag 613
 // (op 1121) Deallocate 597
 // Deleting tag 597
 // (op 1121) Deallocate 579
 // Deleting tag 579
 // (op 1121) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 792): v4

 // Deleting tag 560
 // (op 1121) Deallocate 611
 // Deleting tag 611
 // (op 1121) Deallocate 603
 // Deleting tag 603
 // (op 1121) Deallocate 594
 // Freeing DataFlowTag594: 1 VGPR (Value: UInt32) (op 903): v6

 // Deleting tag 594
 // (op 1121) Deallocate 545
 // Deleting tag 545
 // (op 1121) Deallocate 602
 // Freeing DataFlowTag602: 1 VGPR (Value: UInt32) (op 927): v7

 // Deleting tag 602
 // (op 1121) Deallocate 610
 // Freeing DataFlowTag610: 1 VGPR (Value: UInt32) (op 951): v5

 // Deleting tag 610
 // (op 1) Deallocate{}(1121) END
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(35)S)BL(123) BEGIN
 // (op 123) Lock for Conditional Branch
 // Lock Branch
 // (op 123) Generate Equal(0.00000:S, DataFlowTag(35)S)BL into VCC Value: Bool64 x 1: vcc
 // (op 123) reg expression
 // (op 123) Equal(0.00000:S, v8:S)BL64
v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
 // (op 123) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 123) 
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123
 // (op 123) generate({318})
 // (op 123) Scope(318) BEGIN
 // (op 318) Lock Scope 318
 // Lock Branch
 // (op 318) generate({321})
 // (op 318) NOP(321) BEGIN
 // (op 321) generate({})
 // (op 321) end: generate({})
 // (op 318) NOP(321) END
 // (op 318) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(46) BEGIN
 // (op 46) Assign dim(55) = Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA
 // (op 46) Generate Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA into nullptr
 // (op 46) reg expression
 // (op 46) reg expression
 // (op 46) Multiply(v9:S, {DataFlowTag51: a[0:3]:S})S
// Allocated : 4 VGPRs (Value: Float) (op 46): v4, v5, v6, v7
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
// Allocated : 4 VGPRs (Value: Float) (op 46): v10, v11, v12, v13
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 
 // Freeing : 4 VGPRs (Value: Float) (op 46): v4, v5, v6, v7

 // (op 318) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(46) END
 // (op 318) NOP(855) BEGIN
 // (op 855) generate({})
 // (op 855) end: generate({})
 // (op 318) NOP(855) END
 // (op 318) NOP(852) BEGIN
 // (op 852) generate({})
 // (op 852) end: generate({})
 // (op 318) NOP(852) END
 // (op 318) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 4:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(859) BEGIN
 // (op 859) Assign dim(580) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 4:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 580: v**UNALLOCATED**
 // (op 859) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 4:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag580: VGPR Value: UInt32 x 1: (unallocated)
 // (op 859) reg expression
 // (op 859) reg expression
 // (op 859) reg expression
 // (op 859) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 859) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 859): v0
v_lshrrev_b32 v0, 6, v1 // (op 859) 
 // (op 859) convert
 // (op 859) BEGIN: Tile[1]
 // (op 859) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 859): v3
v_and_b32 v3, 63, v1 // (op 859) 
 // (op 859) END: Tile[1]
 // (op 859) LogicalShiftR({Tile[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 859): v4
v_lshrrev_b32 v4, 4, v3 // (op 859) 
 // (op 859) BEGIN: Flatten
 // (op 859) {Flatten: ShiftL(v4:U32, 2:U32)U32}
 // (op 859) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 859)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 859)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 859): v5
v_lshlrev_b32 v5, 2, v4 // (op 859) 
 // (op 859) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 859): v4

 // (op 859) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 859): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 859) 
 // (op 859) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 859): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 859): v0

 // (op 859) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 859): v0
v_lshl_add_u32 v0, s16, 5, v4 // (op 859) 
 // (op 859) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 859): v4

 // (op 859) LogicalShiftR({Tile[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 859): v4
v_lshrrev_b32 v4, 2, v3 // (op 859) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 859): v3

 // (op 859) BEGIN: Tile[1]
 // (op 859) {Tile[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 859): v3
v_and_b32 v3, 3, v4 // (op 859) 
 // (op 859) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 859): v4

 // (op 859) BEGIN: Tile[1]
 // (op 859) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 859): v4
v_and_b32 v4, 3, v1 // (op 859) 
 // (op 859) END: Tile[1]
 // (op 859) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 859): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 859) 
 // (op 859) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 859): v4

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 859): v3

 // (op 859) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 859): v3
v_lshl_add_u32 v3, s17, 4, v5 // (op 859) 
 // (op 859) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 859): v5

 // (op 859) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 859): v4
v_mul_lo_u32 v4, v3, s30 // (op 859) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 859): v3

 // (op 859) BEGIN: Join
// Allocated DataFlowTag580: 1 VGPR (Value: UInt32) (op 859): v3
v_add_lshl_u32 v3, v0, v4, 2 // (op 859) 
 // (op 859) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 859): v4

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 859): v0

 // (op 318) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 4:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(859) END
 // (op 318) Assign VGPR 64:U64(860) BEGIN
 // (op 860) Assign dim(581) = 64:U64
 // (op 860) Assign stride expression
 // (op 318) Assign VGPR 64:U64(860) END
 // (op 318) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(861) BEGIN
 // (op 861) Assign dim(583) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 583: s**UNALLOCATED**
 // (op 861) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag583: SGPR Buffer: None x 1: (unallocated)
 // (op 861) reg expression
 // (op 861) reg expression
 // (op 861) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 861) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 861)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 861)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 861): s14, s15
s_mul_i32 s15, s30, s13 // (op 861) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 861): s0
s_mul_i32 s0, s31, s12 // (op 861) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 861) least significant
// Allocated : 1 SGPR (Value: Int32) (op 861): s1
s_mul_hi_u32 s1, s30, s12 // (op 861) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 861) most significant: sum
s_add_u32 s15, s15, s1 // (op 861) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 861): s1

 // Freeing : 1 SGPR (Value: Int32) (op 861): s0

 // (op 861) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 861) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 861)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 861)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 861): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 861) 
 // Freeing : 2 SGPRs (Value: Int64) (op 861): s14, s15

 // (op 861) reg expression
 // (op 861) reg expression
// Allocated DataFlowTag583: 4 SGPRs (Buffer: None) (op 861): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 861) reg expression
s_mov_b32 s9, s33 // (op 861) reg expression
s_mov_b32 s10, s0 // (op 861) reg expression
s_mov_b32 s11, 131072 // (op 861) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 861): s0, s1

 // (op 318) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(861) END
 // (op 318) NOP(853) BEGIN
 // (op 853) generate({})
 // (op 853) end: generate({})
 // (op 318) NOP(853) END
 // (op 318) Assign VGPR 4:U64(866) BEGIN
 // (op 866) Assign dim(585) = 4:U64
 // (op 866) Assign stride expression
 // (op 318) Assign VGPR 4:U64(866) END
 // (op 318) StoreTiled Value: Float(52) BEGIN
 // (op 52) GEN: storeMacroTileWAVE OP 52 MacroTile 55 WaveTile 331
 // (op 52) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 52) Generate 4:U64 into nullptr
 // (op 52) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag55: VGPR Value: Float x 4: v[10:13]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag583: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 52) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag55: VGPR Value: Float x 4: v[10:13]
 // 	info.rowOffsetReg = DataFlowTag580: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag583: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 52) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx4 v[10:13], v3, s[8:11], 0 offen // (op 52) Store value
// VMEM: Expected complete at 5933 (current 5453)
 // (op 318) StoreTiled Value: Float(52) END
 // (op 318) Deallocate{}(1049) BEGIN
 // (op 1049) Deallocate 580
 // Freeing DataFlowTag580: 1 VGPR (Value: UInt32) (op 859): v3

 // Deleting tag 580
 // (op 1049) Deallocate 584
 // Deleting tag 584
 // (op 1049) Deallocate 581
 // Deleting tag 581
 // (op 1049) Deallocate 585
 // Deleting tag 585
 // (op 1049) Deallocate 55
 // Freeing DataFlowTag55: 4 VGPRs (Value: Float) (op 46): v10, v11, v12, v13

 // Deleting tag 55
 // (op 318) Deallocate{}(1049) END
 // (op 318) end: generate({321})
 // (op 318) Unlock Scope 318
 // Unlock None
 // (op 123) Scope(318) END
 // (op 123) end: generate({318})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 123) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 123) 
s_branch RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalFalse_0_beta_123:
 // (op 123) 
 // (op 123) generate({324})
 // (op 123) Scope(324) BEGIN
 // (op 324) Lock Scope 324
 // Lock Branch
 // (op 324) generate({327})
 // (op 324) NOP(327) BEGIN
 // (op 327) generate({})
 // (op 327) end: generate({})
 // (op 324) NOP(327) END
 // (op 324) NOP(131) BEGIN
 // (op 131) generate({})
 // (op 131) end: generate({})
 // (op 324) NOP(131) END
 // (op 324) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(132) BEGIN
 // (op 132) Assign dim(389) = Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA
 // (op 132) Generate Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA into nullptr
 // (op 132) reg expression
 // (op 132) reg expression
 // (op 132) Multiply(v9:S, {DataFlowTag51: a[0:3]:S})S
 // (op 324) Scope(676) BEGIN
 // (op 676) Lock Scope 676
 // Lock Branch
 // (op 676) generate({673})
 // (op 676) NOP(673) BEGIN
 // (op 673) generate({})
 // (op 673) end: generate({})
 // (op 676) NOP(673) END
 // (op 676) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 4:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(681) BEGIN
 // (op 681) Assign dim(526) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 4:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}
 // tag 526: v**UNALLOCATED**
 // (op 681) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 4:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32} into DataFlowTag526: VGPR Value: UInt32 x 1: (unallocated)
 // (op 681) reg expression
 // (op 681) reg expression
 // (op 681) reg expression
 // (op 681) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 681) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 681): v0
v_lshrrev_b32 v0, 6, v1 // (op 681) 
 // (op 681) convert
 // (op 681) BEGIN: Flatten[1]
 // (op 681) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 681): v3
v_and_b32 v3, 63, v1 // (op 681) 
 // (op 681) END: Flatten[1]
 // (op 681) LogicalShiftR({Flatten[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 681): v4
v_lshrrev_b32 v4, 4, v3 // (op 681) 
 // (op 681) BEGIN: Tile
 // (op 681) {Tile: ShiftL(v4:U32, 2:U32)U32}
 // (op 681) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 681)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 681)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 681): v5
v_lshlrev_b32 v5, 2, v4 // (op 681) 
 // (op 681) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 681): v4

 // (op 681) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 681): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 681) 
 // (op 681) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 681): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 681): v0

 // (op 681) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 681): v0
v_lshl_add_u32 v0, s16, 5, v4 // (op 681) 
 // (op 681) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 681): v4

 // (op 681) LogicalShiftR({Flatten[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 681): v4
v_lshrrev_b32 v4, 2, v3 // (op 681) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 681): v3

 // (op 681) BEGIN: Flatten[1]
 // (op 681) {Flatten[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 681): v3
v_and_b32 v3, 3, v4 // (op 681) 
 // (op 681) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 681): v4

 // (op 681) BEGIN: Flatten[1]
 // (op 681) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 681): v4
v_and_b32 v4, 3, v1 // (op 681) 
 // (op 681) END: Flatten[1]
 // (op 681) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 681): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 681) 
 // (op 681) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 681): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 681): v3

 // (op 681) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 681): v3
v_lshl_add_u32 v3, s17, 4, v5 // (op 681) 
 // (op 681) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 681): v5

 // (op 681) Multiply({Tile: v3:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 681): v4
v_mul_lo_u32 v4, v3, s34 // (op 681) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 681): v3

 // (op 681) BEGIN: Split
// Allocated DataFlowTag526: 1 VGPR (Value: UInt32) (op 681): v3
v_add_lshl_u32 v3, v0, v4, 2 // (op 681) 
 // (op 681) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 681): v4

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 681): v0

 // (op 676) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 4:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(681) END
 // (op 676) Deallocate{}(1139) BEGIN
 // (op 1139) Deallocate 290
 // Deleting tag 290
 // (op 1139) Deallocate 403
 // Deleting tag 403
 // (op 1139) Deallocate 401
 // Deleting tag 401
 // (op 676) Deallocate{}(1139) END
 // (op 676) Assign VGPR 64:U64(682) BEGIN
 // (op 682) Assign dim(527) = 64:U64
 // (op 682) Assign stride expression
 // (op 676) Assign VGPR 64:U64(682) END
 // (op 676) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(683) BEGIN
 // (op 683) Assign dim(529) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 529: s**UNALLOCATED**
 // (op 683) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag529: SGPR Buffer: None x 1: (unallocated)
 // (op 683) reg expression
 // (op 683) reg expression
 // (op 683) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 683) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 683)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 683)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 683): s14, s15
s_mul_i32 s15, s34, s13 // (op 683) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 683): s0
s_mul_i32 s0, s35, s12 // (op 683) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 683) least significant
// Allocated : 1 SGPR (Value: Int32) (op 683): s1
s_mul_hi_u32 s1, s34, s12 // (op 683) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 683) most significant: sum
s_add_u32 s15, s15, s1 // (op 683) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 683): s1

 // Freeing : 1 SGPR (Value: Int32) (op 683): s0

 // (op 683) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 683) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 683)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 683)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 683): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 683) 
 // Freeing : 2 SGPRs (Value: Int64) (op 683): s14, s15

 // (op 683) reg expression
 // (op 683) reg expression
// Allocated DataFlowTag529: 4 SGPRs (Buffer: None) (op 683): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 683) reg expression
s_mov_b32 s5, s37 // (op 683) reg expression
s_mov_b32 s6, s0 // (op 683) reg expression
s_mov_b32 s7, 131072 // (op 683) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 683): s0, s1

 // (op 676) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(683) END
 // (op 676) Deallocate{Tensor_10_pointer}(1204) BEGIN
 // (op 1204) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 676) Deallocate{Tensor_10_pointer}(1204) END
 // (op 676) NOP(674) BEGIN
 // (op 674) generate({})
 // (op 674) end: generate({})
 // (op 676) NOP(674) END
 // (op 676) Assign VGPR 4:U64(688) BEGIN
 // (op 688) Assign dim(531) = 4:U64
 // (op 688) Assign stride expression
 // (op 676) Assign VGPR 4:U64(688) END
 // (op 676) LoadTiled Value: Float(135) BEGIN
 // (op 135) GEN: loadMacroTileWAVECIACCUM OP 135 WaveTile 291
 // (op 135) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 135) Generate 4:U64 into nullptr
 // tag 385: v**UNALLOCATED**
 // (op 135) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 4: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag529: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 135) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag526: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag529: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 135) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 4 VGPRs (Value: Float) (op 135): v4, v5, v6, v7
buffer_load_dwordx4 v[4:7], v3, s[4:7], 0 offen // (op 135) Load value
// VMEM: Expected complete at 6441 (current 5961)
 // (op 676) LoadTiled Value: Float(135) END
 // (op 676) Deallocate{Tensor_10_stride_1_10}(1061) BEGIN
 // (op 1061) Deallocate 5
 // Deleting tag 5
 // (op 1061) Deallocate 530
 // Deleting tag 530
 // (op 1061) Deallocate 529
 // Freeing DataFlowTag529: 4 SGPRs (Buffer: None) (op 683): s4, s5, s6, s7

 // Deleting tag 529
 // (op 1061) Deallocate 531
 // Deleting tag 531
 // (op 1061) Deallocate 526
 // Freeing DataFlowTag526: 1 VGPR (Value: UInt32) (op 681): v3

 // Deleting tag 526
 // (op 1061) Deallocate 527
 // Deleting tag 527
 // (op 1061) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 676) Deallocate{Tensor_10_stride_1_10}(1061) END
 // (op 676) end: generate({673})
 // (op 676) Unlock Scope 676
 // Unlock None
 // (op 324) Scope(676) END
// Allocated : 4 VGPRs (Value: Float) (op 132): v10, v11, v12, v13
v_accvgpr_read v10, a0 // (op 132) 
 // (op 324) Assign VGPR Multiply(DataFlowTag(35)NA, DataFlowTag(385)NA)NA(136) BEGIN
 // (op 136) Assign dim(387) = Multiply(DataFlowTag(35)NA, DataFlowTag(385)NA)NA
 // (op 136) Generate Multiply(DataFlowTag(35)NA, DataFlowTag(385)NA)NA into nullptr
 // (op 136) reg expression
 // (op 136) reg expression
 // (op 136) Multiply(v8:S, v[4:7]:S)S
v_accvgpr_read v11, a1 // (op 132) 
v_accvgpr_read v12, a2 // (op 132) 
v_accvgpr_read v13, a3 // (op 132) 
// Allocated : 4 VGPRs (Value: Float) (op 132): v14, v15, v16, v17
v_mul_f32 v14, v9, v10 // (op 132) 
v_mul_f32 v15, v9, v11 // (op 132) 
v_mul_f32 v16, v9, v12 // (op 132) 
v_mul_f32 v17, v9, v13 // (op 132) 
 // Freeing : 4 VGPRs (Value: Float) (op 132): v10, v11, v12, v13

 // (op 324) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(132) END
// Allocated : 4 VGPRs (Value: Float) (op 136): v10, v11, v12, v13
s_waitcnt vmcnt(0)
v_mul_f32 v10, v8, v4 // (op 136) 
v_mul_f32 v11, v8, v5 // (op 136) 
v_mul_f32 v12, v8, v6 // (op 136) 
v_mul_f32 v13, v8, v7 // (op 136) 
 // (op 324) Assign VGPR Multiply(DataFlowTag(35)NA, DataFlowTag(385)NA)NA(136) END
 // (op 324) Deallocate{}(1063) BEGIN
 // (op 1063) Deallocate 385
 // Freeing : 4 VGPRs (Value: Float) (op 135): v4, v5, v6, v7

 // Deleting tag 385
 // (op 324) Deallocate{}(1063) END
 // (op 324) Assign VGPR Add(DataFlowTag(387)NA, DataFlowTag(389)NA)NA(133) BEGIN
 // (op 133) Assign dim(391) = Add(DataFlowTag(387)NA, DataFlowTag(389)NA)NA
 // (op 133) Generate Add(DataFlowTag(387)NA, DataFlowTag(389)NA)NA into nullptr
 // (op 133) reg expression
 // (op 133) reg expression
 // (op 133) Add({DataFlowTag387: v[10:13]:S}, {DataFlowTag389: v[14:17]:S})S
// Allocated : 4 VGPRs (Value: Float) (op 133): v4, v5, v6, v7
v_add_f32 v4, v10, v14 // (op 133) 
v_add_f32 v5, v11, v15 // (op 133) 
v_add_f32 v6, v12, v16 // (op 133) 
v_add_f32 v7, v13, v17 // (op 133) 
 // (op 324) Assign VGPR Add(DataFlowTag(387)NA, DataFlowTag(389)NA)NA(133) END
 // (op 324) Deallocate{}(1057) BEGIN
 // (op 1057) Deallocate 389
 // Freeing DataFlowTag389: 4 VGPRs (Value: Float) (op 132): v14, v15, v16, v17

 // Deleting tag 389
 // (op 1057) Deallocate 387
 // Freeing DataFlowTag387: 4 VGPRs (Value: Float) (op 136): v10, v11, v12, v13

 // Deleting tag 387
 // (op 324) Deallocate{}(1057) END
 // (op 324) NOP(872) BEGIN
 // (op 872) generate({})
 // (op 872) end: generate({})
 // (op 324) NOP(872) END
 // (op 324) NOP(869) BEGIN
 // (op 869) generate({})
 // (op 869) end: generate({})
 // (op 324) NOP(869) END
 // (op 324) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 4:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(876) BEGIN
 // (op 876) Assign dim(586) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 4:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 586: v**UNALLOCATED**
 // (op 876) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 4:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag586: VGPR Value: UInt32 x 1: (unallocated)
 // (op 876) reg expression
 // (op 876) reg expression
 // (op 876) reg expression
 // (op 876) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 876) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 876): v0
v_lshrrev_b32 v0, 6, v1 // (op 876) 
 // (op 876) convert
 // (op 876) BEGIN: Tile[1]
 // (op 876) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 876): v3
v_and_b32 v3, 63, v1 // (op 876) 
 // (op 876) END: Tile[1]
 // (op 876) LogicalShiftR({Tile[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 876): v10
v_lshrrev_b32 v10, 4, v3 // (op 876) 
 // (op 876) BEGIN: Flatten
 // (op 876) {Flatten: ShiftL(v10:U32, 2:U32)U32}
 // (op 876) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 876)         value (VGPR Value: UInt32 x 1: v10) 
 // (op 876)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 876): v11
v_lshlrev_b32 v11, 2, v10 // (op 876) 
 // (op 876) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 876): v10

 // (op 876) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 876): v10
v_lshl_add_u32 v10, v0, 4, v11 // (op 876) 
 // (op 876) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 876): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 876): v0

 // (op 876) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 876): v0
v_lshl_add_u32 v0, s16, 5, v10 // (op 876) 
 // (op 876) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 876): v10

 // (op 876) LogicalShiftR({Tile[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 876): v10
v_lshrrev_b32 v10, 2, v3 // (op 876) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 876): v3

 // (op 876) BEGIN: Tile[1]
 // (op 876) {Tile[1]: BitwiseAnd(v10:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 876): v3
v_and_b32 v3, 3, v10 // (op 876) 
 // (op 876) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 876): v10

 // (op 876) BEGIN: Tile[1]
 // (op 876) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 876): v10
v_and_b32 v10, 3, v1 // (op 876) 
 // (op 876) END: Tile[1]
 // (op 876) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 876): v11
v_lshl_add_u32 v11, v3, 2, v10 // (op 876) 
 // (op 876) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 876): v10

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 876): v3

 // (op 876) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 876): v3
v_lshl_add_u32 v3, s17, 4, v11 // (op 876) 
 // (op 876) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 876): v11

 // (op 876) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 876): v10
v_mul_lo_u32 v10, v3, s30 // (op 876) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 876): v3

 // (op 876) BEGIN: Join
// Allocated DataFlowTag586: 1 VGPR (Value: UInt32) (op 876): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 876) 
 // (op 876) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 876): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 876): v0

 // (op 324) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 4:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(876) END
 // (op 324) Assign VGPR 64:U64(877) BEGIN
 // (op 877) Assign dim(587) = 64:U64
 // (op 877) Assign stride expression
 // (op 324) Assign VGPR 64:U64(877) END
 // (op 324) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(878) BEGIN
 // (op 878) Assign dim(583) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 878) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag583: SGPR Buffer: None x 1: s[8:11]
 // (op 878) reg expression
 // (op 878) reg expression
 // (op 878) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 878) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 878)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 878)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 878): s14, s15
s_mul_i32 s15, s30, s13 // (op 878) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 878): s0
s_mul_i32 s0, s31, s12 // (op 878) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 878) least significant
// Allocated : 1 SGPR (Value: Int32) (op 878): s1
s_mul_hi_u32 s1, s30, s12 // (op 878) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 878) most significant: sum
s_add_u32 s15, s15, s1 // (op 878) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 878): s1

 // Freeing : 1 SGPR (Value: Int32) (op 878): s0

 // (op 878) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 878) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 878)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 878)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 878): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 878) 
 // Freeing : 2 SGPRs (Value: Int64) (op 878): s14, s15

 // (op 878) reg expression
 // (op 878) reg expression
s_mov_b32 s8, s32 // (op 878) reg expression
s_mov_b32 s9, s33 // (op 878) reg expression
s_mov_b32 s10, s0 // (op 878) reg expression
s_mov_b32 s11, 131072 // (op 878) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 878): s0, s1

 // (op 324) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(878) END
 // (op 324) NOP(870) BEGIN
 // (op 870) generate({})
 // (op 870) end: generate({})
 // (op 324) NOP(870) END
 // (op 324) Assign VGPR 4:U64(883) BEGIN
 // (op 883) Assign dim(589) = 4:U64
 // (op 883) Assign stride expression
 // (op 324) Assign VGPR 4:U64(883) END
 // (op 324) StoreTiled Value: Float(134) BEGIN
 // (op 134) GEN: storeMacroTileWAVE OP 134 MacroTile 391 WaveTile 331
 // (op 134) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 134) Generate 4:U64 into nullptr
 // (op 134) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag391: VGPR Value: Float x 4: v[4:7]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag583: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 134) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag391: VGPR Value: Float x 4: v[4:7]
 // 	info.rowOffsetReg = DataFlowTag586: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag583: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 134) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx4 v[4:7], v3, s[8:11], 0 offen // (op 134) Store value
// VMEM: Expected complete at 6955 (current 6475)
 // (op 324) StoreTiled Value: Float(134) END
 // (op 324) Deallocate{}(1059) BEGIN
 // (op 1059) Deallocate 586
 // Freeing DataFlowTag586: 1 VGPR (Value: UInt32) (op 876): v3

 // Deleting tag 586
 // (op 1059) Deallocate 587
 // Deleting tag 587
 // (op 1059) Deallocate 391
 // Freeing DataFlowTag391: 4 VGPRs (Value: Float) (op 133): v4, v5, v6, v7

 // Deleting tag 391
 // (op 1059) Deallocate 588
 // Deleting tag 588
 // (op 1059) Deallocate 589
 // Deleting tag 589
 // (op 324) Deallocate{}(1059) END
 // (op 324) end: generate({327})
 // (op 324) Unlock Scope 324
 // Unlock None
 // (op 123) Scope(324) END
 // (op 123) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1123) BEGIN
 // (op 1123) Deallocate 405
 // Deleting tag 405
 // (op 1123) Deallocate 407
 // Deleting tag 407
 // (op 1123) Deallocate 330
 // Deleting tag 330
 // (op 1123) Deallocate 583
 // Freeing DataFlowTag583: 4 SGPRs (Buffer: None) (op 861): s8, s9, s10, s11

 // Deleting tag 583
 // (op 1123) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 1123) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 123) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1123) END
 // (op 123) end: generate({324})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 123) 
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_ConditionalBottom_0_beta_123:
 // (op 123) 
 // (op 123) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(35)S)BL(123) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1055) BEGIN
 // (op 1055) Deallocate 51
 // Freeing DataFlowTag51: 4 ACCVGPRs (Value: Float) (op 59): a0, a1, a2, a3

 // Deleting tag 51
 // (op 1055) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 39): v9

 // Deleting tag 44
 // (op 1055) Deallocate 35
 // Freeing : 1 VGPR (Value: Float) (op 2): v8

 // Deleting tag 35
 // (op 1055) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1055) END
 // (op 1) end: generate({735, 788, 841, 899, 923, 947})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.LRR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end:
.size RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2, .LRR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2_end-RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 9472 // lds bytes
  .amdhsa_accum_offset 60
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
    DataType: F6
    DestDataType: s
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
    macrotile: [32, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2
    .symbol: RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 9472
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 59
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

