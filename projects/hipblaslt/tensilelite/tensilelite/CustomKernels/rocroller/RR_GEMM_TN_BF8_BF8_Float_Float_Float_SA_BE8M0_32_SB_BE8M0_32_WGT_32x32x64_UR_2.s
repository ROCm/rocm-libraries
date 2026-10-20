// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.BF8.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.BF8.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
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
 // Tensor_0_pointer: PointerGlobal: BF8(offset: 0, size: 8, read_write)
 // Tensor_0_size_0: Value: Int64(offset: 8, size: 8, read_only)
 // Tensor_0_size_1: Value: Int64(offset: 16, size: 8, read_only)
 // Tensor_0_stride_0: Value: Int64(offset: 24, size: 8, read_only)
 // Tensor_0_stride_1: Value: Int64(offset: 32, size: 8, read_only)
 // Tensor_2_pointer: PointerGlobal: BF8(offset: 40, size: 8, read_write)
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
 // KernelArg{Tensor_0_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
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
.globl RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
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
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:
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
 // KernelArg{Tensor_2_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
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
 // Tensor_0_pointer: SGPR PointerGlobal: BF8 x 1: s[4:5]
 // Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]
 // Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]
 // Tensor_2_pointer: SGPR PointerGlobal: BF8 x 1: s[10:11]
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
 // RemoveImplicitScheduling
 // OrderMultiplyNodes
 // Simplify
 // ScheduleMultiplyAndLDS
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
 // (op 1) generate({823, 875, 928, 981, 1054, 1093, 1117, 1141})
 // (op 1) NOP(823) BEGIN
 // (op 823) generate({})
 // (op 823) end: generate({})
 // (op 1) NOP(823) END
 // (op 1) NOP(875) BEGIN
 // (op 875) generate({})
 // (op 875) end: generate({})
 // (op 1) NOP(875) END
 // (op 1) NOP(928) BEGIN
 // (op 928) generate({})
 // (op 928) end: generate({})
 // (op 1) NOP(928) END
 // (op 1) NOP(981) BEGIN
 // (op 981) generate({})
 // (op 981) end: generate({})
 // (op 1) NOP(981) END
 // (op 1) NOP(1054) BEGIN
 // (op 1054) generate({})
 // (op 1054) end: generate({})
 // (op 1) NOP(1054) END
 // (op 1) NOP(1093) BEGIN
 // (op 1093) generate({})
 // (op 1093) end: generate({})
 // (op 1) NOP(1093) END
 // (op 1) NOP(1117) BEGIN
 // (op 1117) generate({})
 // (op 1117) end: generate({})
 // (op 1) NOP(1117) END
 // (op 1) NOP(1141) BEGIN
 // (op 1141) generate({})
 // (op 1141) end: generate({})
 // (op 1) NOP(1141) END
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(827) BEGIN
 // (op 827) Assign dim(584) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 584: v**UNALLOCATED**
 // (op 827) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag584: VGPR Value: UInt32 x 1: (unallocated)
 // (op 827) reg expression
 // (op 827) BEGIN: Flatten[1]
 // (op 827) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(879) BEGIN
 // (op 879) Assign dim(600) = {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 600: v**UNALLOCATED**
 // (op 879) Generate {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag600: VGPR Value: UInt32 x 1: (unallocated)
 // (op 879) reg expression
 // (op 879) BEGIN: Flatten[1]
 // (op 879) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(932) BEGIN
 // (op 932) Assign dim(616) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 616: v**UNALLOCATED**
 // (op 932) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag616: VGPR Value: UInt32 x 1: (unallocated)
 // (op 932) reg expression
 // (op 932) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(985) BEGIN
 // (op 985) Assign dim(632) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 632: v**UNALLOCATED**
 // (op 985) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag632: VGPR Value: UInt32 x 1: (unallocated)
 // (op 985) reg expression
 // (op 985) BEGIN: Flatten[1]Tile
 // (op 985) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1058) BEGIN
 // (op 1058) Assign dim(654) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 654: v**UNALLOCATED**
 // (op 1058) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag654: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1058) reg expression
 // (op 1058) BEGIN: Flatten[1]
 // (op 1058) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1097) BEGIN
 // (op 1097) Assign dim(666) = {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 666: v**UNALLOCATED**
 // (op 1097) Generate {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag666: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1097) reg expression
 // (op 1097) BEGIN: Flatten[1]
 // (op 1097) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(1121) BEGIN
 // (op 1121) Assign dim(674) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 674: v**UNALLOCATED**
 // (op 1121) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag674: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1121) reg expression
 // (op 1121) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(1145) BEGIN
 // (op 1145) Assign dim(682) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 682: v**UNALLOCATED**
 // (op 1145) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag682: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1145) reg expression
 // (op 1145) BEGIN: Flatten[1]Tile
 // (op 1145) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 827): v0
v_and_b32 v0, 63, v1 // (op 827) 
 // (op 827) END: Flatten[1]
 // (op 827) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 827): v3
v_lshrrev_b32 v3, 4, v0 // (op 827) 
 // (op 827) BEGIN: Flatten[1]
 // (op 827) {Flatten[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 827): v4
v_and_b32 v4, 1, v3 // (op 827) 
 // (op 827) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 827): v3

 // (op 827) BEGIN: Flatten[1]
 // (op 827) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 827): v3
v_and_b32 v3, 15, v1 // (op 827) 
 // (op 827) END: Flatten[1]
 // (op 827) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 827): v5
v_lshl_add_u32 v5, v4, 4, v3 // (op 827) 
 // (op 827) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 827): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 827): v4

 // (op 827) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 827): v3
v_lshrrev_b32 v3, 6, v1 // (op 827) 
 // (op 827) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 827): v4
v_lshl_add_u32 v4, v3, 5, v5 // (op 827) 
 // (op 827) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 827): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 827): v3

 // (op 827) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 827): v3
v_lshrrev_b32 v3, 5, v0 // (op 827) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 827): v0

 // (op 827) BEGIN: TileTile
 // (op 827) {TileTile: ShiftL(v3:U32, 4:U32)U32}
 // (op 827) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 827)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 827)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 827): v0
v_lshlrev_b32 v0, 4, v3 // (op 827) 
 // (op 827) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 827): v3

 // (op 827) BEGIN: Tile
// Allocated DataFlowTag584: 1 VGPR (Value: UInt32) (op 827): v3
v_lshl_add_u32 v3, v4, 6, v0 // (op 827) 
 // (op 827) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 827): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 827): v4

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(827) END
// Allocated : 1 VGPR (Value: UInt32) (op 879): v0
v_and_b32 v0, 63, v1 // (op 879) 
 // (op 879) END: Flatten[1]
 // (op 879) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 879): v4
v_lshrrev_b32 v4, 4, v0 // (op 879) 
 // (op 879) BEGIN: Flatten[1]
 // (op 879) {Flatten[1]: BitwiseAnd(v4:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 879): v5
v_and_b32 v5, 1, v4 // (op 879) 
 // (op 879) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 879): v4

 // (op 879) BEGIN: Flatten[1]
 // (op 879) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 879): v4
v_and_b32 v4, 15, v1 // (op 879) 
 // (op 879) END: Flatten[1]
 // (op 879) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 879): v6
v_lshl_add_u32 v6, v5, 4, v4 // (op 879) 
 // (op 879) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 879): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 879): v5

 // (op 879) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 879): v4
v_lshrrev_b32 v4, 5, v0 // (op 879) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 879): v0

 // (op 879) BEGIN: TileTile
 // (op 879) {TileTile: ShiftL(v4:U32, 4:U32)U32}
 // (op 879) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 879)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 879)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 879): v0
v_lshlrev_b32 v0, 4, v4 // (op 879) 
 // (op 879) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 879): v4

 // (op 879) BEGIN: Tile
// Allocated DataFlowTag600: 1 VGPR (Value: UInt32) (op 879): v4
v_lshl_add_u32 v4, v6, 6, v0 // (op 879) 
 // (op 879) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 879): v0

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 879): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(879) END
// Allocated : 1 VGPR (Value: UInt32) (op 932): v0
v_lshrrev_b32 v0, 6, v1 // (op 932) 
 // (op 932) BEGIN: Flatten[1]
 // (op 932) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 932): v5
v_and_b32 v5, 31, v1 // (op 932) 
 // (op 932) END: Flatten[1]
 // (op 932) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 932): v6
v_lshl_add_u32 v6, v0, 5, v5 // (op 932) 
 // (op 932) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 932): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 932): v0

 // (op 932) BEGIN: Flatten[1]
 // (op 932) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 932): v0
v_and_b32 v0, 63, v1 // (op 932) 
 // (op 932) END: Flatten[1]
 // (op 932) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 932): v5
v_lshrrev_b32 v5, 5, v0 // (op 932) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 932): v0

 // (op 932) BEGIN: Tile
// Allocated DataFlowTag616: 1 VGPR (Value: UInt32) (op 932): v0
v_lshl_add_u32 v0, v6, 1, v5 // (op 932) 
 // (op 932) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 932): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 932): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(932) END
// Allocated : 1 VGPR (Value: UInt32) (op 985): v5
v_and_b32 v5, 31, v1 // (op 985) 
 // (op 985) END: Flatten[1]Tile
 // (op 985) BEGIN: Flatten[1]
 // (op 985) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 985): v6
v_and_b32 v6, 63, v1 // (op 985) 
 // (op 985) END: Flatten[1]
 // (op 985) LogicalShiftR({Flatten[1]: v6:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 985): v7
v_lshrrev_b32 v7, 5, v6 // (op 985) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 985): v6

 // (op 985) BEGIN: Tile
// Allocated DataFlowTag632: 1 VGPR (Value: UInt32) (op 985): v6
v_lshl_add_u32 v6, v5, 1, v7 // (op 985) 
 // (op 985) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 985): v7

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 985): v5

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(985) END
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v5
v_and_b32 v5, 63, v1 // (op 1058) 
 // (op 1058) END: Flatten[1]
 // (op 1058) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v7
v_lshrrev_b32 v7, 4, v5 // (op 1058) 
 // (op 1058) BEGIN: Flatten[1]
 // (op 1058) {Flatten[1]: BitwiseAnd(v7:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v8
v_and_b32 v8, 1, v7 // (op 1058) 
 // (op 1058) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1058): v7

 // (op 1058) BEGIN: Flatten[1]
 // (op 1058) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v7
v_and_b32 v7, 15, v1 // (op 1058) 
 // (op 1058) END: Flatten[1]
 // (op 1058) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v9
v_lshl_add_u32 v9, v8, 4, v7 // (op 1058) 
 // (op 1058) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1058): v7

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1058): v8

 // (op 1058) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v7
v_lshrrev_b32 v7, 6, v1 // (op 1058) 
 // (op 1058) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v8
v_lshl_add_u32 v8, v7, 5, v9 // (op 1058) 
 // (op 1058) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1058): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1058): v7

 // (op 1058) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v7
v_lshrrev_b32 v7, 5, v5 // (op 1058) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1058): v5

 // (op 1058) BEGIN: TileTile
 // (op 1058) {TileTile: ShiftL(v7:U32, 4:U32)U32}
 // (op 1058) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1058)         value (VGPR Value: UInt32 x 1: v7) 
 // (op 1058)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v5
v_lshlrev_b32 v5, 4, v7 // (op 1058) 
 // (op 1058) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 1058): v7

 // (op 1058) BEGIN: Tile
// Allocated DataFlowTag654: 1 VGPR (Value: UInt32) (op 1058): v7
v_lshl_add_u32 v7, v8, 6, v5 // (op 1058) 
 // (op 1058) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1058): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1058): v8

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1058) END
// Allocated : 1 VGPR (Value: UInt32) (op 1097): v5
v_and_b32 v5, 63, v1 // (op 1097) 
 // (op 1097) END: Flatten[1]
 // (op 1097) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
 // (op 1) Deallocate{}(1383) BEGIN
 // (op 1383) Deallocate 102
 // Deleting tag 102
 // (op 1) Deallocate{}(1383) END
// Allocated : 1 VGPR (Value: UInt32) (op 1097): v8
v_lshrrev_b32 v8, 4, v5 // (op 1097) 
 // (op 1097) BEGIN: Flatten[1]
 // (op 1097) {Flatten[1]: BitwiseAnd(v8:U32, 1:U32)U32}
 // (op 1) Assign VGPR 64:U32(828) BEGIN
 // (op 828) Assign dim(585) = 64:U32
 // (op 828) Assign stride expression
 // (op 1) Assign VGPR 64:U32(828) END
 // (op 1) Assign VGPR 64:U32(1059) BEGIN
 // (op 1059) Assign dim(655) = 64:U32
 // (op 1059) Assign stride expression
 // (op 1) Assign VGPR 64:U32(1059) END
 // (op 1) NOP(824) BEGIN
 // (op 824) generate({})
 // (op 824) end: generate({})
 // (op 1) NOP(824) END
 // (op 1) NOP(1055) BEGIN
 // (op 1055) generate({})
 // (op 1055) end: generate({})
 // (op 1) NOP(1055) END
 // (op 1) Assign VGPR 1:U32(832) BEGIN
 // (op 832) Assign dim(587) = 1:U32
 // (op 832) Assign stride expression
 // (op 1) Assign VGPR 1:U32(832) END
 // (op 1) Assign VGPR 1:U32(1063) BEGIN
 // (op 1063) Assign dim(657) = 1:U32
 // (op 1063) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1063) END
// Allocated : 1 VGPR (Value: UInt32) (op 1097): v9
v_and_b32 v9, 1, v8 // (op 1097) 
 // (op 1097) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1097): v8

 // (op 1097) BEGIN: Flatten[1]
 // (op 1097) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1097): v8
v_and_b32 v8, 15, v1 // (op 1097) 
 // (op 1097) END: Flatten[1]
 // (op 1097) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 1097): v10
v_lshl_add_u32 v10, v9, 4, v8 // (op 1097) 
 // (op 1097) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1097): v8

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1097): v9

 // (op 1097) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1097): v8
v_lshrrev_b32 v8, 5, v5 // (op 1097) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1097): v5

 // (op 1097) BEGIN: TileTile
 // (op 1097) {TileTile: ShiftL(v8:U32, 4:U32)U32}
 // (op 1097) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1097)         value (VGPR Value: UInt32 x 1: v8) 
 // (op 1097)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1097): v5
v_lshlrev_b32 v5, 4, v8 // (op 1097) 
 // (op 1097) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 1097): v8

 // (op 1097) BEGIN: Tile
// Allocated DataFlowTag666: 1 VGPR (Value: UInt32) (op 1097): v8
v_lshl_add_u32 v8, v10, 6, v5 // (op 1097) 
 // (op 1097) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1097): v5

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1097): v10

 // (op 1) Assign VGPR {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1097) END
// Allocated : 1 VGPR (Value: UInt32) (op 1121): v5
v_lshrrev_b32 v5, 6, v1 // (op 1121) 
 // (op 1121) BEGIN: Flatten[1]
 // (op 1121) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Deallocate{}(1386) BEGIN
 // (op 1386) Deallocate 168
 // Deleting tag 168
 // (op 1) Deallocate{}(1386) END
// Allocated : 1 VGPR (Value: UInt32) (op 1121): v9
v_and_b32 v9, 31, v1 // (op 1121) 
 // (op 1121) END: Flatten[1]
 // (op 1121) BEGIN: Tile
 // (op 1) Assign VGPR 64:U32(880) BEGIN
 // (op 880) Assign dim(601) = 64:U32
 // (op 880) Assign stride expression
 // (op 1) Assign VGPR 64:U32(880) END
 // (op 1) Assign VGPR 64:U32(1098) BEGIN
 // (op 1098) Assign dim(667) = 64:U32
 // (op 1098) Assign stride expression
 // (op 1) Assign VGPR 64:U32(1098) END
 // (op 1) NOP(876) BEGIN
 // (op 876) generate({})
 // (op 876) end: generate({})
 // (op 1) NOP(876) END
 // (op 1) NOP(1094) BEGIN
 // (op 1094) generate({})
 // (op 1094) end: generate({})
 // (op 1) NOP(1094) END
 // (op 1) Assign VGPR 1:U32(884) BEGIN
 // (op 884) Assign dim(603) = 1:U32
 // (op 884) Assign stride expression
 // (op 1) Assign VGPR 1:U32(884) END
 // (op 1) Assign VGPR 1:U32(1102) BEGIN
 // (op 1102) Assign dim(669) = 1:U32
 // (op 1102) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1102) END
// Allocated : 1 VGPR (Value: UInt32) (op 1121): v10
v_lshl_add_u32 v10, v5, 5, v9 // (op 1121) 
 // (op 1121) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1121): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1121): v5

 // (op 1121) BEGIN: Flatten[1]
 // (op 1121) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1121): v5
v_and_b32 v5, 63, v1 // (op 1121) 
 // (op 1121) END: Flatten[1]
 // (op 1121) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1121): v9
v_lshrrev_b32 v9, 5, v5 // (op 1121) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1121): v5

 // (op 1121) BEGIN: Tile
// Allocated DataFlowTag674: 1 VGPR (Value: UInt32) (op 1121): v5
v_lshl_add_u32 v5, v10, 1, v9 // (op 1121) 
 // (op 1121) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1121): v9

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1121): v10

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(1121) END
// Allocated : 1 VGPR (Value: UInt32) (op 1145): v9
v_and_b32 v9, 31, v1 // (op 1145) 
 // (op 1145) END: Flatten[1]Tile
 // (op 1145) BEGIN: Flatten[1]
 // (op 1145) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Deallocate{}(1389) BEGIN
 // (op 1389) Deallocate 234
 // Deleting tag 234
 // (op 1) Deallocate{}(1389) END
// Allocated : 1 VGPR (Value: UInt32) (op 1145): v10
v_and_b32 v10, 63, v1 // (op 1145) 
 // (op 1145) END: Flatten[1]
 // (op 1145) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
 // (op 1) Assign VGPR 2:U32(933) BEGIN
 // (op 933) Assign dim(617) = 2:U32
 // (op 933) Assign stride expression
 // (op 1) Assign VGPR 2:U32(933) END
 // (op 1) Assign VGPR 2:U32(1122) BEGIN
 // (op 1122) Assign dim(675) = 2:U32
 // (op 1122) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1122) END
 // (op 1) NOP(929) BEGIN
 // (op 929) generate({})
 // (op 929) end: generate({})
 // (op 1) NOP(929) END
 // (op 1) NOP(1118) BEGIN
 // (op 1118) generate({})
 // (op 1118) end: generate({})
 // (op 1) NOP(1118) END
 // (op 1) Assign VGPR 1:U32(937) BEGIN
 // (op 937) Assign dim(619) = 1:U32
 // (op 937) Assign stride expression
 // (op 1) Assign VGPR 1:U32(937) END
 // (op 1) Assign VGPR 1:U32(1126) BEGIN
 // (op 1126) Assign dim(677) = 1:U32
 // (op 1126) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1126) END
// Allocated : 1 VGPR (Value: UInt32) (op 1145): v11
v_lshrrev_b32 v11, 5, v10 // (op 1145) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1145): v10

 // (op 1145) BEGIN: Tile
// Allocated DataFlowTag682: 1 VGPR (Value: UInt32) (op 1145): v10
v_lshl_add_u32 v10, v9, 1, v11 // (op 1145) 
 // (op 1145) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1145): v11

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 1145): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(1145) END
 // (op 1) Deallocate{}(1392) BEGIN
 // (op 1392) Deallocate 294
 // Deleting tag 294
 // (op 1) Deallocate{}(1392) END
 // (op 1) Assign VGPR 2:U32(986) BEGIN
 // (op 986) Assign dim(633) = 2:U32
 // (op 986) Assign stride expression
 // (op 1) Assign VGPR 2:U32(986) END
 // (op 1) Assign VGPR 2:U32(1146) BEGIN
 // (op 1146) Assign dim(683) = 2:U32
 // (op 1146) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1146) END
 // (op 1) NOP(982) BEGIN
 // (op 982) generate({})
 // (op 982) end: generate({})
 // (op 1) NOP(982) END
 // (op 1) NOP(1142) BEGIN
 // (op 1142) generate({})
 // (op 1142) end: generate({})
 // (op 1) NOP(1142) END
 // (op 1) Assign VGPR 1:U32(990) BEGIN
 // (op 990) Assign dim(635) = 1:U32
 // (op 990) Assign stride expression
 // (op 1) Assign VGPR 1:U32(990) END
 // (op 1) Assign VGPR 1:U32(1150) BEGIN
 // (op 1150) Assign dim(685) = 1:U32
 // (op 1150) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1150) END
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
 // (op 1) Deallocate{user_Float_Value_14}(1433) BEGIN
 // (op 1433) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1433) END
 // (op 1) Deallocate{user_Float_Value_12}(1435) BEGIN
 // (op 1435) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1435) END
 // (op 1) Scope(340) BEGIN
 // (op 340) Lock Scope 340
 // Lock Branch
 // (op 340) generate({345})
 // (op 340) NOP(345) BEGIN
 // (op 345) generate({})
 // (op 345) end: generate({})
 // (op 340) NOP(345) END
 // (op 340) Assign ACCVGPR 0.00000:S(63) BEGIN
 // (op 63) Assign dim(55) = 0.00000:S
 // tag 55: a**UNALLOCATED**
 // (op 63) Generate 0.00000:S into DataFlowTag55: ACCVGPR Value: Float x 16: (unallocated)
// Allocated DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 63): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
 // (op 340) Assign ACCVGPR 0.00000:S(63) END
 // (op 340) Scope(530) BEGIN
 // (op 530) Lock Scope 530
 // Lock Branch
 // (op 530) generate({525})
 // (op 530) NOP(525) BEGIN
 // (op 525) generate({})
 // (op 525) end: generate({})
 // (op 530) NOP(525) END
 // (op 530) Assign VGPR 64:U64(535) BEGIN
 // (op 535) Assign dim(498) = 64:U64
 // (op 535) Assign stride expression
 // (op 530) Assign VGPR 64:U64(535) END
 // (op 530) NOP(526) BEGIN
 // (op 526) generate({})
 // (op 526) end: generate({})
 // (op 530) NOP(526) END
 // (op 530) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(538) BEGIN
 // (op 538) Assign dim(499) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}
 // tag 499: v**UNALLOCATED**
 // (op 538) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64} into DataFlowTag499: VGPR Value: UInt64 x 1: (unallocated)
 // (op 538) reg expression
 // (op 538) reg expression
 // (op 538) reg expression
 // (op 538) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 538): v12
v_lshrrev_b32 v12, 2, v1 // (op 538) 
 // (op 538) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 538): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 538) 
 // (op 538) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 538): v12

 // (op 538) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 538): v12
v_and_b32 v12, -4, v1 // (op 538) 
 // (op 538) BEGIN: Flatten[1]
 // (op 538) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 538): v14
v_sub_u32 v14, v1, v12 // (op 538) 
 // (op 538) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 538): v12

 // (op 538) BEGIN: TileTile
 // (op 538) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 538) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 538)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 538)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 538): v12
v_lshlrev_b32 v12, 4, v14 // (op 538) 
 // (op 538) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 538): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 538): v14, v15
v_mov_b32 v15, 0 // (op 538) convert
v_mov_b32 v14, v12 // (op 538) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 538): v12

 // (op 538) BEGIN: Split
 // (op 538) Multiply: dest (DataFlowTag499: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 538)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 538)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag499: 2 VGPRs (Value: UInt64) (op 538): v16, v17
v_mul_lo_u32 v17, v13, s3 // (op 538) most significant: low of low * high
 // (op 538) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 538): v12
v_mul_hi_u32 v12, v13, s2 // (op 538) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 538) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 538) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 538): v12

 // (op 538) Add: dest (DataFlowTag499: VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 538)      lhs (DataFlowTag499: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 538)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 538): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 538) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 538) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 538): s0, s1

 // (op 538) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 538): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 538): v13

 // (op 530) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(538) END
 // (op 530) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(539) BEGIN
 // (op 539) Assign dim(500) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 539) Assign stride expression
 // (op 530) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(539) END
 // (op 530) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(540) BEGIN
 // (op 540) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 493: s**UNALLOCATED**
 // (op 540) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: (unallocated)
 // (op 540) reg expression
 // (op 540) reg expression
 // (op 540) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 540) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 540)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 540)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 540): s0, s1
s_mul_i32 s1, s2, s7 // (op 540) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 540): s14
s_mul_i32 s14, s3, s6 // (op 540) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 540) least significant
// Allocated : 1 SGPR (Value: Int32) (op 540): s15
s_mul_hi_u32 s15, s2, s6 // (op 540) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 540) most significant: sum
s_add_u32 s1, s1, s15 // (op 540) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 540): s15

 // Freeing : 1 SGPR (Value: Int32) (op 540): s14

 // (op 540) reg expression
 // (op 540) reg expression
// Allocated DataFlowTag493: 4 SGPRs (Buffer: None) (op 540): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 540) reg expression
s_mov_b32 s41, s5 // (op 540) reg expression
s_mov_b32 s42, s0 // (op 540) reg expression
s_mov_b32 s43, 131072 // (op 540) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 540): s0, s1

 // (op 530) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(540) END
 // (op 530) NOP(527) BEGIN
 // (op 527) generate({})
 // (op 527) end: generate({})
 // (op 530) NOP(527) END
 // (op 530) Assign VGPR 1:U64(545) BEGIN
 // (op 545) Assign dim(502) = 1:U64
 // (op 545) Assign stride expression
 // (op 530) Assign VGPR 1:U64(545) END
 // (op 530) NOP(548) BEGIN
 // (op 548) generate({})
 // (op 548) end: generate({})
 // (op 530) NOP(548) END
 // (op 530) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(557) BEGIN
 // (op 557) Assign dim(503) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}
 // tag 503: v**UNALLOCATED**
 // (op 557) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64} into DataFlowTag503: VGPR Value: UInt64 x 1: (unallocated)
 // (op 557) reg expression
 // (op 557) reg expression
 // (op 557) reg expression
 // (op 557) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 557): v12
v_lshrrev_b32 v12, 2, v1 // (op 557) 
 // (op 557) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 557): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 557) 
 // (op 557) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 557): v12

 // (op 557) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 557): v12
v_and_b32 v12, -4, v1 // (op 557) 
 // (op 557) BEGIN: Flatten[1]
 // (op 557) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 557): v14
v_sub_u32 v14, v1, v12 // (op 557) 
 // (op 557) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 557): v12

 // (op 557) BEGIN: TileTile
 // (op 557) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 557) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 557)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 557)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 557): v12
v_lshlrev_b32 v12, 4, v14 // (op 557) 
 // (op 557) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 557): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 557): v14, v15
v_mov_b32 v15, 0 // (op 557) convert
v_mov_b32 v14, v12 // (op 557) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 557): v12

 // (op 557) BEGIN: Split
 // (op 557) Multiply: dest (DataFlowTag503: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 557)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 557)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag503: 2 VGPRs (Value: UInt64) (op 557): v18, v19
v_mul_lo_u32 v19, v13, s3 // (op 557) most significant: low of low * high
 // (op 557) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 557): v12
v_mul_hi_u32 v12, v13, s2 // (op 557) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 557) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 557) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 557): v12

 // (op 557) Add: dest (DataFlowTag503: VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 557)      lhs (DataFlowTag503: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 557)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 557): s0, s1
v_add_co_u32 v18, s[0:1], v18, v14 // (op 557) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 557) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 557): s0, s1

 // (op 557) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 557): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 557): v13

 // (op 530) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(557) END
 // (op 530) Assign VGPR 64:U64(558) BEGIN
 // (op 558) Assign dim(504) = 64:U64
 // (op 558) Assign stride expression
 // (op 530) Assign VGPR 64:U64(558) END
 // (op 530) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(559) BEGIN
 // (op 559) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 559) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // (op 559) reg expression
 // (op 559) reg expression
 // (op 559) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 559) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 559)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 559)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 559): s0, s1
s_mul_i32 s1, s2, s7 // (op 559) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 559): s14
s_mul_i32 s14, s3, s6 // (op 559) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 559) least significant
// Allocated : 1 SGPR (Value: Int32) (op 559): s15
s_mul_hi_u32 s15, s2, s6 // (op 559) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 559) most significant: sum
s_add_u32 s1, s1, s15 // (op 559) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 559): s15

 // Freeing : 1 SGPR (Value: Int32) (op 559): s14

 // (op 559) reg expression
 // (op 559) reg expression
s_mov_b32 s40, s4 // (op 559) reg expression
s_mov_b32 s41, s5 // (op 559) reg expression
s_mov_b32 s42, s0 // (op 559) reg expression
s_mov_b32 s43, 131072 // (op 559) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 559): s0, s1

 // (op 530) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(559) END
 // (op 530) NOP(550) BEGIN
 // (op 550) generate({})
 // (op 550) end: generate({})
 // (op 530) NOP(550) END
 // (op 530) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(564) BEGIN
 // (op 564) Assign dim(506) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 564) Assign stride expression
 // (op 530) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(564) END
 // (op 530) NOP(551) BEGIN
 // (op 551) generate({})
 // (op 551) end: generate({})
 // (op 530) NOP(551) END
 // (op 530) Assign VGPR 1:U64(567) BEGIN
 // (op 567) Assign dim(508) = 1:U64
 // (op 567) Assign stride expression
 // (op 530) Assign VGPR 1:U64(567) END
 // (op 530) NOP(592) BEGIN
 // (op 592) generate({})
 // (op 592) end: generate({})
 // (op 530) NOP(592) END
 // (op 530) Assign VGPR 64:U64(599) BEGIN
 // (op 599) Assign dim(517) = 64:U64
 // (op 599) Assign stride expression
 // (op 530) Assign VGPR 64:U64(599) END
 // (op 530) NOP(593) BEGIN
 // (op 593) generate({})
 // (op 593) end: generate({})
 // (op 530) NOP(593) END
 // (op 530) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(602) BEGIN
 // (op 602) Assign dim(518) = {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 518: v**UNALLOCATED**
 // (op 602) Generate {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag518: VGPR Value: UInt64 x 1: (unallocated)
 // (op 602) reg expression
 // (op 602) reg expression
 // (op 602) reg expression
 // (op 602) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 602): v12
v_and_b32 v12, -4, v1 // (op 602) 
 // (op 602) BEGIN: Flatten[1]
 // (op 602) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 602): v13
v_sub_u32 v13, v1, v12 // (op 602) 
 // (op 602) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 602): v12

 // (op 602) BEGIN: TileTile
 // (op 602) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 602) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 602)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 602)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 602): v12
v_lshlrev_b32 v12, 4, v13 // (op 602) 
 // (op 602) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 602): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 602): v14, v15
v_mov_b32 v15, 0 // (op 602) convert
v_mov_b32 v14, v12 // (op 602) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 602): v12

 // (op 602) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 602): v12
v_lshrrev_b32 v12, 2, v1 // (op 602) 
 // (op 602) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 602): v13
v_lshl_add_u32 v13, s17, 5, v12 // (op 602) 
 // (op 602) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 602): v12

 // (op 602) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 602) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 602)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 602)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 602): v20, v21
v_mul_lo_u32 v21, v13, s9 // (op 602) most significant: low of low * high
 // (op 602) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 602): v12
v_mul_hi_u32 v12, v13, s8 // (op 602) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 602) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 602) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 602): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 602): v13

 // (op 602) BEGIN: Split
 // (op 602) {Split: Add({TileTile: v[14:15]:U64}, v[20:21]:I64)U64}
 // (op 602) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 602)      lhs (TileTile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 602)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated DataFlowTag518: 2 VGPRs (Value: UInt64) (op 602): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 602): s0, s1
v_add_co_u32 v12, s[0:1], v14, v20 // (op 602) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 602) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 602): s0, s1

 // (op 602) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 602): v20, v21

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 602): v14, v15

 // (op 530) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(602) END
 // (op 530) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(603) BEGIN
 // (op 603) Assign dim(519) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 603) Assign stride expression
 // (op 530) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(603) END
 // (op 530) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(604) BEGIN
 // (op 604) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 512: s**UNALLOCATED**
 // (op 604) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: (unallocated)
 // (op 604) reg expression
 // (op 604) reg expression
 // (op 604) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 604) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 604)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 604)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 604): s0, s1
s_mul_i32 s1, s8, s13 // (op 604) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 604): s14
s_mul_i32 s14, s9, s12 // (op 604) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 604) least significant
// Allocated : 1 SGPR (Value: Int32) (op 604): s15
s_mul_hi_u32 s15, s8, s12 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 604) most significant: sum
s_add_u32 s1, s1, s15 // (op 604) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 604): s15

 // Freeing : 1 SGPR (Value: Int32) (op 604): s14

 // (op 604) reg expression
 // (op 604) reg expression
// Allocated DataFlowTag512: 4 SGPRs (Buffer: None) (op 604): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 604) reg expression
s_mov_b32 s45, s11 // (op 604) reg expression
s_mov_b32 s46, s0 // (op 604) reg expression
s_mov_b32 s47, 131072 // (op 604) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 604): s0, s1

 // (op 530) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(604) END
 // (op 530) NOP(594) BEGIN
 // (op 594) generate({})
 // (op 594) end: generate({})
 // (op 530) NOP(594) END
 // (op 530) Assign VGPR 1:U64(609) BEGIN
 // (op 609) Assign dim(521) = 1:U64
 // (op 609) Assign stride expression
 // (op 530) Assign VGPR 1:U64(609) END
 // (op 530) NOP(612) BEGIN
 // (op 612) generate({})
 // (op 612) end: generate({})
 // (op 530) NOP(612) END
 // (op 530) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(621) BEGIN
 // (op 621) Assign dim(522) = {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 522: v**UNALLOCATED**
 // (op 621) Generate {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag522: VGPR Value: UInt64 x 1: (unallocated)
 // (op 621) reg expression
 // (op 621) reg expression
 // (op 621) reg expression
 // (op 621) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 621): v14
v_and_b32 v14, -4, v1 // (op 621) 
 // (op 621) BEGIN: Flatten[1]
 // (op 621) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 621): v15
v_sub_u32 v15, v1, v14 // (op 621) 
 // (op 621) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 621): v14

 // (op 621) BEGIN: TileTile
 // (op 621) {TileTile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 621) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 621)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 621)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 621): v14
v_lshlrev_b32 v14, 4, v15 // (op 621) 
 // (op 621) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 621): v15

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 621): v20, v21
v_mov_b32 v21, 0 // (op 621) convert
v_mov_b32 v20, v14 // (op 621) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 621): v14

 // (op 621) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 621): v14
v_lshrrev_b32 v14, 2, v1 // (op 621) 
 // (op 621) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 621): v15
v_lshl_add_u32 v15, s17, 5, v14 // (op 621) 
 // (op 621) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 621): v14

 // (op 621) Multiply({Tile: v15:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 621) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 621)           lhs (Tile: VGPR Value: UInt32 x 1: v15) 
 // (op 621)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 621): v22, v23
v_mul_lo_u32 v23, v15, s9 // (op 621) most significant: low of low * high
 // (op 621) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 621): v14
v_mul_hi_u32 v14, v15, s8 // (op 621) most significant: high of low * low
v_mul_lo_u32 v22, v15, s8 // (op 621) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 621) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 621): v14

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 621): v15

 // (op 621) BEGIN: Split
 // (op 621) {Split: Add({TileTile: v[20:21]:U64}, v[22:23]:I64)U64}
 // (op 621) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 621)      lhs (TileTile: VGPR Value: UInt64 x 1: v[20:21]) 
 // (op 621)      rhs (VGPR Value: Int64 x 1: v[22:23])
// Allocated DataFlowTag522: 2 VGPRs (Value: UInt64) (op 621): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 621): s0, s1
v_add_co_u32 v14, s[0:1], v20, v22 // (op 621) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 621) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 621): s0, s1

 // (op 621) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 621): v22, v23

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 621): v20, v21

 // (op 530) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(621) END
 // (op 530) Assign VGPR 64:U64(622) BEGIN
 // (op 622) Assign dim(523) = 64:U64
 // (op 622) Assign stride expression
 // (op 530) Assign VGPR 64:U64(622) END
 // (op 530) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(623) BEGIN
 // (op 623) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 623) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // (op 623) reg expression
 // (op 623) reg expression
 // (op 623) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 623) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 623)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 623)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 623): s0, s1
s_mul_i32 s1, s8, s13 // (op 623) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 623): s14
s_mul_i32 s14, s9, s12 // (op 623) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 623) least significant
// Allocated : 1 SGPR (Value: Int32) (op 623): s15
s_mul_hi_u32 s15, s8, s12 // (op 623) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 623) most significant: sum
s_add_u32 s1, s1, s15 // (op 623) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 623): s15

 // Freeing : 1 SGPR (Value: Int32) (op 623): s14

 // (op 623) reg expression
 // (op 623) reg expression
s_mov_b32 s44, s10 // (op 623) reg expression
s_mov_b32 s45, s11 // (op 623) reg expression
s_mov_b32 s46, s0 // (op 623) reg expression
s_mov_b32 s47, 131072 // (op 623) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 623): s0, s1

 // (op 530) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(623) END
 // (op 530) NOP(614) BEGIN
 // (op 614) generate({})
 // (op 614) end: generate({})
 // (op 530) NOP(614) END
 // (op 530) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(628) BEGIN
 // (op 628) Assign dim(525) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 628) Assign stride expression
 // (op 530) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(628) END
 // (op 530) NOP(615) BEGIN
 // (op 615) generate({})
 // (op 615) end: generate({})
 // (op 530) NOP(615) END
 // (op 530) Assign VGPR 1:U64(631) BEGIN
 // (op 631) Assign dim(527) = 1:U64
 // (op 631) Assign stride expression
 // (op 530) Assign VGPR 1:U64(631) END
 // (op 530) NOP(656) BEGIN
 // (op 656) generate({})
 // (op 656) end: generate({})
 // (op 530) NOP(656) END
 // (op 530) Assign VGPR 2:U64(663) BEGIN
 // (op 663) Assign dim(536) = 2:U64
 // (op 663) Assign stride expression
 // (op 530) Assign VGPR 2:U64(663) END
 // (op 530) NOP(657) BEGIN
 // (op 657) generate({})
 // (op 657) end: generate({})
 // (op 530) NOP(657) END
 // (op 530) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(666) BEGIN
 // (op 666) Assign dim(537) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}
 // tag 537: v**UNALLOCATED**
 // (op 666) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32} into DataFlowTag537: VGPR Value: UInt32 x 1: (unallocated)
 // (op 666) reg expression
 // (op 666) reg expression
 // (op 666) reg expression
 // (op 666) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 666) convert
// Allocated : 1 VGPR (Value: UInt32) (op 666): v20
v_lshrrev_b32 v20, 1, v1 // (op 666) 
 // (op 666) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 666): v21
v_lshl_add_u32 v21, s16, 5, v20 // (op 666) 
 // (op 666) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 666): v20

 // (op 666) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 666): v20
v_and_b32 v20, -2, v1 // (op 666) 
 // (op 666) BEGIN: Flatten[1]TileTile
 // (op 666) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 666): v22
v_sub_u32 v22, v1, v20 // (op 666) 
 // (op 666) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 666): v20

 // (op 666) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag537: 1 VGPR (Value: UInt32) (op 666): v20
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v20, v21, s20 // (op 666) 
v_add_u32 v20, v20, v22 // (op 666) 
 // (op 666) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 666): v22

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 666): v21

 // (op 530) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(666) END
 // (op 530) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(667) BEGIN
 // (op 667) Assign dim(538) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 667) Assign stride expression
 // (op 530) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(667) END
 // (op 530) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(668) BEGIN
 // (op 668) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 531: s**UNALLOCATED**
 // (op 668) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: (unallocated)
 // (op 668) reg expression
 // (op 668) reg expression
 // (op 668) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 668) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 668)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 668)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 668): s0, s1
s_mul_i32 s1, s20, s7 // (op 668) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 668): s14
s_mul_i32 s14, s21, s6 // (op 668) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 668) least significant
// Allocated : 1 SGPR (Value: Int32) (op 668): s15
s_mul_hi_u32 s15, s20, s6 // (op 668) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 668) most significant: sum
s_add_u32 s1, s1, s15 // (op 668) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 668): s15

 // Freeing : 1 SGPR (Value: Int32) (op 668): s14

 // (op 668) reg expression
 // (op 668) reg expression
// Allocated DataFlowTag531: 4 SGPRs (Buffer: None) (op 668): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 668) reg expression
s_mov_b32 s49, s23 // (op 668) reg expression
s_mov_b32 s50, s0 // (op 668) reg expression
s_mov_b32 s51, 131072 // (op 668) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 668): s0, s1

 // (op 530) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(668) END
 // (op 530) NOP(658) BEGIN
 // (op 658) generate({})
 // (op 658) end: generate({})
 // (op 530) NOP(658) END
 // (op 530) Assign VGPR 1:U64(673) BEGIN
 // (op 673) Assign dim(540) = 1:U64
 // (op 673) Assign stride expression
 // (op 530) Assign VGPR 1:U64(673) END
 // (op 530) NOP(676) BEGIN
 // (op 676) generate({})
 // (op 676) end: generate({})
 // (op 530) NOP(676) END
 // (op 530) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(685) BEGIN
 // (op 685) Assign dim(541) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}
 // tag 541: v**UNALLOCATED**
 // (op 685) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32} into DataFlowTag541: VGPR Value: UInt32 x 1: (unallocated)
 // (op 685) reg expression
 // (op 685) reg expression
 // (op 685) reg expression
 // (op 685) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 685) convert
// Allocated : 1 VGPR (Value: UInt32) (op 685): v21
v_lshrrev_b32 v21, 1, v1 // (op 685) 
 // (op 685) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 685): v22
v_lshl_add_u32 v22, s16, 5, v21 // (op 685) 
 // (op 685) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 685): v21

 // (op 685) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 685): v21
v_and_b32 v21, -2, v1 // (op 685) 
 // (op 685) BEGIN: Flatten[1]TileTile
 // (op 685) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v21:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 685): v23
v_sub_u32 v23, v1, v21 // (op 685) 
 // (op 685) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 685): v21

 // (op 685) BEGIN: Split
// Allocated DataFlowTag541: 1 VGPR (Value: UInt32) (op 685): v21
v_mul_lo_u32 v21, v22, s20 // (op 685) 
v_add_u32 v21, v21, v23 // (op 685) 
 // (op 685) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 685): v23

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 685): v22

 // (op 530) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(685) END
 // (op 530) Assign VGPR 2:U64(686) BEGIN
 // (op 686) Assign dim(542) = 2:U64
 // (op 686) Assign stride expression
 // (op 530) Assign VGPR 2:U64(686) END
 // (op 530) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(687) BEGIN
 // (op 687) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 687) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // (op 687) reg expression
 // (op 687) reg expression
 // (op 687) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 687) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 687)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 687)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 687): s0, s1
s_mul_i32 s1, s20, s7 // (op 687) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 687): s14
s_mul_i32 s14, s21, s6 // (op 687) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 687) least significant
// Allocated : 1 SGPR (Value: Int32) (op 687): s15
s_mul_hi_u32 s15, s20, s6 // (op 687) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 687) most significant: sum
s_add_u32 s1, s1, s15 // (op 687) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 687): s15

 // Freeing : 1 SGPR (Value: Int32) (op 687): s14

 // (op 687) reg expression
 // (op 687) reg expression
s_mov_b32 s48, s22 // (op 687) reg expression
s_mov_b32 s49, s23 // (op 687) reg expression
s_mov_b32 s50, s0 // (op 687) reg expression
s_mov_b32 s51, 131072 // (op 687) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 687): s0, s1

 // (op 530) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(687) END
 // (op 530) NOP(678) BEGIN
 // (op 678) generate({})
 // (op 678) end: generate({})
 // (op 530) NOP(678) END
 // (op 530) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(692) BEGIN
 // (op 692) Assign dim(544) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 692) Assign stride expression
 // (op 530) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(692) END
 // (op 530) NOP(679) BEGIN
 // (op 679) generate({})
 // (op 679) end: generate({})
 // (op 530) NOP(679) END
 // (op 530) Assign VGPR 1:U64(695) BEGIN
 // (op 695) Assign dim(546) = 1:U64
 // (op 695) Assign stride expression
 // (op 530) Assign VGPR 1:U64(695) END
 // (op 530) NOP(720) BEGIN
 // (op 720) generate({})
 // (op 720) end: generate({})
 // (op 530) NOP(720) END
 // (op 530) Assign VGPR 2:U64(727) BEGIN
 // (op 727) Assign dim(555) = 2:U64
 // (op 727) Assign stride expression
 // (op 530) Assign VGPR 2:U64(727) END
 // (op 530) NOP(721) BEGIN
 // (op 721) generate({})
 // (op 721) end: generate({})
 // (op 530) NOP(721) END
 // (op 530) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(730) BEGIN
 // (op 730) Assign dim(556) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 556: v**UNALLOCATED**
 // (op 730) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag556: VGPR Value: UInt32 x 1: (unallocated)
 // (op 730) reg expression
 // (op 730) reg expression
 // (op 730) reg expression
 // (op 730) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 730) convert
// Allocated : 1 VGPR (Value: UInt32) (op 730): v22
v_and_b32 v22, -2, v1 // (op 730) 
 // (op 730) BEGIN: Flatten[1]TileTile
 // (op 730) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v22:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 730): v23
v_sub_u32 v23, v1, v22 // (op 730) 
 // (op 730) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 730): v22

 // (op 730) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 730): v22
v_lshrrev_b32 v22, 1, v1 // (op 730) 
 // (op 730) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 730): v24
v_lshl_add_u32 v24, s17, 5, v22 // (op 730) 
 // (op 730) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 730): v22

 // (op 730) Multiply({Tile: v24:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 730): v22
v_mul_lo_u32 v22, v24, s24 // (op 730) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 730): v24

 // (op 730) BEGIN: Split
 // (op 730) {Split: Add({Flatten[1]TileTile: v23:U32}, v22:U32)U32}
// Allocated DataFlowTag556: 1 VGPR (Value: UInt32) (op 730): v24
v_add_u32 v24, v23, v22 // (op 730) 
 // (op 730) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 730): v22

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 730): v23

 // (op 530) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(730) END
 // (op 530) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(731) BEGIN
 // (op 731) Assign dim(557) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 731) Assign stride expression
 // (op 530) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(731) END
 // (op 530) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(732) BEGIN
 // (op 732) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 550: s**UNALLOCATED**
 // (op 732) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: (unallocated)
 // (op 732) reg expression
 // (op 732) reg expression
 // (op 732) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 732) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 732)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 732)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 732): s0, s1
s_mul_i32 s1, s24, s13 // (op 732) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 732): s14
s_mul_i32 s14, s25, s12 // (op 732) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 732) least significant
// Allocated : 1 SGPR (Value: Int32) (op 732): s15
s_mul_hi_u32 s15, s24, s12 // (op 732) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 732) most significant: sum
s_add_u32 s1, s1, s15 // (op 732) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 732): s15

 // Freeing : 1 SGPR (Value: Int32) (op 732): s14

 // (op 732) reg expression
 // (op 732) reg expression
// Allocated DataFlowTag550: 4 SGPRs (Buffer: None) (op 732): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 732) reg expression
s_mov_b32 s53, s27 // (op 732) reg expression
s_mov_b32 s54, s0 // (op 732) reg expression
s_mov_b32 s55, 131072 // (op 732) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 732): s0, s1

 // (op 530) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(732) END
 // (op 530) NOP(722) BEGIN
 // (op 722) generate({})
 // (op 722) end: generate({})
 // (op 530) NOP(722) END
 // (op 530) Assign VGPR 1:U64(737) BEGIN
 // (op 737) Assign dim(559) = 1:U64
 // (op 737) Assign stride expression
 // (op 530) Assign VGPR 1:U64(737) END
 // (op 530) NOP(740) BEGIN
 // (op 740) generate({})
 // (op 740) end: generate({})
 // (op 530) NOP(740) END
 // (op 530) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(749) BEGIN
 // (op 749) Assign dim(560) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 560: v**UNALLOCATED**
 // (op 749) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 749) reg expression
 // (op 749) reg expression
 // (op 749) reg expression
 // (op 749) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 749) convert
// Allocated : 1 VGPR (Value: UInt32) (op 749): v22
v_and_b32 v22, -2, v1 // (op 749) 
 // (op 749) BEGIN: Flatten[1]TileTile
 // (op 749) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v22:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 749): v23
v_sub_u32 v23, v1, v22 // (op 749) 
 // (op 749) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 749): v22

 // (op 749) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 749): v22
v_lshrrev_b32 v22, 1, v1 // (op 749) 
 // (op 749) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 749): v25
v_lshl_add_u32 v25, s17, 5, v22 // (op 749) 
 // (op 749) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 749): v22

 // (op 749) Multiply({Tile: v25:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 749): v22
v_mul_lo_u32 v22, v25, s24 // (op 749) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 749): v25

 // (op 749) BEGIN: Split
 // (op 749) {Split: Add({Flatten[1]TileTile: v23:U32}, v22:U32)U32}
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 749): v25
v_add_u32 v25, v23, v22 // (op 749) 
 // (op 749) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 749): v22

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 749): v23

 // (op 530) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(749) END
 // (op 530) Assign VGPR 2:U64(750) BEGIN
 // (op 750) Assign dim(561) = 2:U64
 // (op 750) Assign stride expression
 // (op 530) Assign VGPR 2:U64(750) END
 // (op 530) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(751) BEGIN
 // (op 751) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 751) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // (op 751) reg expression
 // (op 751) reg expression
 // (op 751) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 751) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 751)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 751)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 751): s0, s1
s_mul_i32 s1, s24, s13 // (op 751) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 751): s14
s_mul_i32 s14, s25, s12 // (op 751) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 751) least significant
// Allocated : 1 SGPR (Value: Int32) (op 751): s15
s_mul_hi_u32 s15, s24, s12 // (op 751) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 751) most significant: sum
s_add_u32 s1, s1, s15 // (op 751) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 751): s15

 // Freeing : 1 SGPR (Value: Int32) (op 751): s14

 // (op 751) reg expression
 // (op 751) reg expression
s_mov_b32 s52, s26 // (op 751) reg expression
s_mov_b32 s53, s27 // (op 751) reg expression
s_mov_b32 s54, s0 // (op 751) reg expression
s_mov_b32 s55, 131072 // (op 751) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 751): s0, s1

 // (op 530) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(751) END
 // (op 530) NOP(742) BEGIN
 // (op 742) generate({})
 // (op 742) end: generate({})
 // (op 530) NOP(742) END
 // (op 530) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(756) BEGIN
 // (op 756) Assign dim(563) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 756) Assign stride expression
 // (op 530) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(756) END
 // (op 530) NOP(743) BEGIN
 // (op 743) generate({})
 // (op 743) end: generate({})
 // (op 530) NOP(743) END
 // (op 530) Assign VGPR 1:U64(759) BEGIN
 // (op 759) Assign dim(565) = 1:U64
 // (op 759) Assign stride expression
 // (op 530) Assign VGPR 1:U64(759) END
 // (op 530) NOP(360) BEGIN
 // (op 360) generate({})
 // (op 360) end: generate({})
 // (op 530) NOP(360) END
 // (op 530) SetCoordinate(365) BEGIN
 // (op 365) generate({})
 // (op 365) end: generate({})
 // (op 365) generate({863})
 // (op 365) Scope(863) BEGIN
 // (op 863) Lock Scope 863
 // Lock Branch
 // (op 863) generate({860})
 // (op 863) NOP(860) BEGIN
 // (op 860) generate({})
 // (op 860) end: generate({})
 // (op 863) NOP(860) END
 // (op 863) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(867) BEGIN
 // (op 867) Assign dim(596) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 596: s**UNALLOCATED**
 // (op 867) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag596: SGPR Value: UInt32 x 1: (unallocated)
 // (op 867) reg expression
 // (op 867) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 867): v22
v_and_b32 v22, -4, v1 // (op 867) 
 // (op 867) BEGIN: Tile[1]
 // (op 867) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v22:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 867): v23
v_sub_u32 v23, v1, v22 // (op 867) 
 // (op 867) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 867): v22

 // (op 867) BEGIN: Flatten
 // (op 867) {Flatten: ShiftL({Tile[1]: v23:U32}, 4:U32)U32}
 // (op 867) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 867)         value (Tile[1]: VGPR Value: UInt32 x 1: v23) 
 // (op 867)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 867): v22
v_lshlrev_b32 v22, 4, v23 // (op 867) 
 // (op 867) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 867): v23

 // (op 867) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 867): v23
v_lshrrev_b32 v23, 2, v1 // (op 867) 
 // (op 867) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 867): v26
v_lshl_add_u32 v26, v23, 6, v22 // (op 867) 
 // (op 867) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 867): v22

 // Freeing : 1 VGPR (Value: UInt32) (op 867): v23

// Allocated DataFlowTag596: 1 SGPR (Value: UInt32) (op 867): s0
s_nop 0
v_readfirstlane_b32 s0, v26 // (op 867) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 867): v26

 // (op 863) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(867) END
 // (op 863) Assign VGPR 1024:U32(868) BEGIN
 // (op 868) Assign dim(597) = 1024:U32
 // (op 868) Assign stride expression
 // (op 863) Assign VGPR 1024:U32(868) END
 // (op 863) NOP(861) BEGIN
 // (op 861) generate({})
 // (op 861) end: generate({})
 // (op 863) NOP(861) END
 // (op 863) Assign VGPR 1:U32(872) BEGIN
 // (op 872) Assign dim(599) = 1:U32
 // (op 872) Assign stride expression
 // (op 863) Assign VGPR 1:U32(872) END
 // (op 863) LoadTileDirect2LDS Value: BF8x4(487) BEGIN
 // (op 487) GEN: LoadTileDirect2LDS
 // (op 487) GEN: loadMacroTileDirect2LDS OP 487 LDS 25 MacroTile 9
 // (op 487) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 487) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride519: VGPR Value: UInt64 x 1: (unallocated)
 // (op 487) reg expression
 // (op 487) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 487) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 487)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 487)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 487): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 487) 
// Allocated Stride519: 2 VGPRs (Value: UInt64) (op 487): v22, v23
v_mov_b64 v[22:23], s[14:15] // (op 487) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 487): s14, s15

 // (op 487) Generate 1:U64 into nullptr
 // (op 487) Generate 1024:U32 into nullptr
 // (op 487) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride519: VGPR Value: UInt64 x 1: v[22:23]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 487): v26, v27
v_mov_b64 v[26:27], v[12:13] // (op 487) 
 // (op 487) Generate Add(0:U32, {DataFlowTag596Flatten: s0:U32})U32 into nullptr
 // (op 487)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride519: v[22:23]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 487) Generate {DataFlowTag596Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 487) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 487) Load value direct to lds
// VMEM: Expected complete at 798 (current 318)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 487) Generate Add(v[26:27]:U64, v22:R)U64 into VGPR Value: UInt64 x 1: v[26:27]
 // (op 487) reg expression
 // (op 487) reg expression
 // (op 487) Add(v[26:27]:U64, v22:R)U64
 // (op 487) Add: dest (VGPR Value: UInt64 x 1: v[26:27]) = 
 // (op 487)      lhs (VGPR Value: UInt64 x 1: v[26:27]) 
 // (op 487)      rhs (VGPR Value: Raw32 x 1: v22)
// Allocated : 2 SGPRs (Value: Bool64) (op 487): s14, s15
v_add_co_u32 v26, s[14:15], v26, v22 // (op 487) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 487) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 487): s14, s15

 // (op 487) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 487) reg expression
 // (op 487) Add(m0:U32, 1024:U32)U32
 // (op 487) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 487) 
 // (op 487) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 487) Load value direct to lds
// VMEM: Expected complete at 802 (current 322)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing Stride519: 2 VGPRs (Value: UInt64) (op 487): v22, v23

 // Freeing : 2 VGPRs (Value: UInt64) (op 487): v26, v27

 // (op 863) LoadTileDirect2LDS Value: BF8x4(487) END
 // (op 863) Deallocate{}(1373) BEGIN
 // (op 1373) Deallocate 596
 // Freeing DataFlowTag596: 1 SGPR (Value: UInt32) (op 867): s0

 // Deleting tag 596
 // (op 1373) Deallocate 597
 // Deleting tag 597
 // (op 1373) Deallocate 599
 // Deleting tag 599
 // (op 1373) Deallocate 598
 // Deleting tag 598
 // (op 863) Deallocate{}(1373) END
 // (op 863) end: generate({860})
 // (op 863) Unlock Scope 863
 // Unlock None
 // (op 365) Scope(863) END
 // (op 365) end: generate({863})
 // (op 530) SetCoordinate(365) END
 // (op 530) SetCoordinate(368) BEGIN
 // (op 368) generate({})
 // (op 368) end: generate({})
 // (op 368) generate({369})
 // (op 368) LoadTiled Value: E8M0(369) BEGIN
 // (op 369) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 369) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 369) Generate 1:U64 into nullptr
 // tag 11: v**UNALLOCATED**
 // (op 369) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 369) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag556: VGPR Value: UInt32 x 1: v24
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 369) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 369): v22
buffer_load_ubyte v22, v24, s[52:55], 0 offen offset: 0 // (op 369) Load value
// VMEM: Expected complete at 803 (current 323)
 // (op 368) LoadTiled Value: E8M0(369) END
 // (op 368) end: generate({369})
 // (op 530) SetCoordinate(368) END
 // (op 530) SetCoordinate(371) BEGIN
 // (op 371) generate({})
 // (op 371) end: generate({})
 // (op 371) generate({811})
 // (op 371) Scope(811) BEGIN
 // (op 811) Lock Scope 811
 // Lock Branch
 // (op 811) generate({808})
 // (op 811) NOP(808) BEGIN
 // (op 808) generate({})
 // (op 808) end: generate({})
 // (op 811) NOP(808) END
 // (op 811) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(815) BEGIN
 // (op 815) Assign dim(580) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 580: s**UNALLOCATED**
 // (op 815) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag580: SGPR Value: UInt32 x 1: (unallocated)
 // (op 815) reg expression
 // (op 815) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 815): v23
v_and_b32 v23, -4, v1 // (op 815) 
 // (op 815) BEGIN: Tile[1]
 // (op 815) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v23:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 815): v26
v_sub_u32 v26, v1, v23 // (op 815) 
 // (op 815) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 815): v23

 // (op 815) BEGIN: Flatten
 // (op 815) {Flatten: ShiftL({Tile[1]: v26:U32}, 4:U32)U32}
 // (op 815) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 815)         value (Tile[1]: VGPR Value: UInt32 x 1: v26) 
 // (op 815)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 815): v23
v_lshlrev_b32 v23, 4, v26 // (op 815) 
 // (op 815) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 815): v26

 // (op 815) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 815): v26
v_lshrrev_b32 v26, 2, v1 // (op 815) 
 // (op 815) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 815): v27
v_lshl_add_u32 v27, v26, 6, v23 // (op 815) 
 // (op 815) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 815): v23

 // Freeing : 1 VGPR (Value: UInt32) (op 815): v26

// Allocated DataFlowTag580: 1 SGPR (Value: UInt32) (op 815): s0
s_nop 0
v_readfirstlane_b32 s0, v27 // (op 815) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 815): v27

 // (op 811) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(815) END
 // (op 811) Assign VGPR 1024:U32(816) BEGIN
 // (op 816) Assign dim(581) = 1024:U32
 // (op 816) Assign stride expression
 // (op 811) Assign VGPR 1024:U32(816) END
 // (op 811) NOP(809) BEGIN
 // (op 809) generate({})
 // (op 809) end: generate({})
 // (op 811) NOP(809) END
 // (op 811) Assign VGPR 1:U32(820) BEGIN
 // (op 820) Assign dim(583) = 1:U32
 // (op 820) Assign stride expression
 // (op 811) Assign VGPR 1:U32(820) END
 // (op 811) LoadTileDirect2LDS Value: BF8x4(492) BEGIN
 // (op 492) GEN: LoadTileDirect2LDS
 // (op 492) GEN: loadMacroTileDirect2LDS OP 492 LDS 17 MacroTile 8
 // (op 492) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 492) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride500: VGPR Value: UInt64 x 1: (unallocated)
 // (op 492) reg expression
 // (op 492) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 492) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 492)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 492)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 492): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 492) 
// Allocated Stride500: 2 VGPRs (Value: UInt64) (op 492): v26, v27
v_mov_b64 v[26:27], s[14:15] // (op 492) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 492): s14, s15

 // (op 492) Generate 1:U64 into nullptr
 // (op 492) Generate 1024:U32 into nullptr
 // (op 492) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride500: VGPR Value: UInt64 x 1: v[26:27]
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
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 492): v28, v29
v_mov_b64 v[28:29], v[16:17] // (op 492) 
 // (op 492) Generate Add(2048:U32, {DataFlowTag580Flatten: s0:U32})U32 into nullptr
 // (op 492) reg expression
 // (op 492) Add(2048:U32, {DataFlowTag580Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 492): s1
s_add_u32 s1, 2048, s0 // (op 492) 
 // (op 492)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride500: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 492) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 492) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 492) Load value direct to lds
// VMEM: Expected stall of 463. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1279 (current 799)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 492) Generate Add(v[28:29]:U64, v26:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 492) reg expression
 // (op 492) reg expression
 // (op 492) Add(v[28:29]:U64, v26:R)U64
 // (op 492) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 492)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 492)      rhs (VGPR Value: Raw32 x 1: v26)
// Allocated : 2 SGPRs (Value: Bool64) (op 492): s14, s15
v_add_co_u32 v28, s[14:15], v28, v26 // (op 492) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 492) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 492): s14, s15

 // (op 492) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 492) reg expression
 // (op 492) Add(m0:U32, 1024:U32)U32
 // (op 492) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 492) 
 // (op 492) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 492) Load value direct to lds
// VMEM: Expected complete at 1283 (current 803)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing Stride500: 2 VGPRs (Value: UInt64) (op 492): v26, v27

 // Freeing : 2 VGPRs (Value: UInt64) (op 492): v28, v29

 // Freeing : 1 SGPR (Value: UInt32) (op 492): s1

 // (op 811) LoadTileDirect2LDS Value: BF8x4(492) END
 // (op 811) Deallocate{}(1375) BEGIN
 // (op 1375) Deallocate 580
 // Freeing DataFlowTag580: 1 SGPR (Value: UInt32) (op 815): s0

 // Deleting tag 580
 // (op 1375) Deallocate 581
 // Deleting tag 581
 // (op 1375) Deallocate 583
 // Deleting tag 583
 // (op 1375) Deallocate 582
 // Deleting tag 582
 // (op 811) Deallocate{}(1375) END
 // (op 811) end: generate({808})
 // (op 811) Unlock Scope 811
 // Unlock None
 // (op 371) Scope(811) END
 // (op 371) end: generate({811})
 // (op 530) SetCoordinate(371) END
 // (op 530) SetCoordinate(374) BEGIN
 // (op 374) generate({})
 // (op 374) end: generate({})
 // (op 374) generate({375})
 // (op 374) LoadTiled Value: E8M0(375) BEGIN
 // (op 375) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 375) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 375) Generate 1:U64 into nullptr
 // tag 10: v**UNALLOCATED**
 // (op 375) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 375) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag537: VGPR Value: UInt32 x 1: v20
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
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 375) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 375): v23
buffer_load_ubyte v23, v20, s[48:51], 0 offen offset: 0 // (op 375) Load value
// VMEM: Expected complete at 1284 (current 804)
 // (op 374) LoadTiled Value: E8M0(375) END
 // (op 374) end: generate({375})
 // (op 530) SetCoordinate(374) END
 // (op 530) SetCoordinate(377) BEGIN
 // (op 377) generate({})
 // (op 377) end: generate({})
 // (op 377) generate({1081})
 // (op 377) Scope(1081) BEGIN
 // (op 1081) Lock Scope 1081
 // Lock Branch
 // (op 1081) generate({1078})
 // (op 1081) NOP(1078) BEGIN
 // (op 1078) generate({})
 // (op 1078) end: generate({})
 // (op 1081) NOP(1078) END
 // (op 1081) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1085) BEGIN
 // (op 1085) Assign dim(662) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 662: s**UNALLOCATED**
 // (op 1085) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag662: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1085) reg expression
 // (op 1085) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1085): v26
v_and_b32 v26, -4, v1 // (op 1085) 
 // (op 1085) BEGIN: Tile[1]
 // (op 1085) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v26:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1085): v27
v_sub_u32 v27, v1, v26 // (op 1085) 
 // (op 1085) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1085): v26

 // (op 1085) BEGIN: Flatten
 // (op 1085) {Flatten: ShiftL({Tile[1]: v27:U32}, 4:U32)U32}
 // (op 1085) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1085)         value (Tile[1]: VGPR Value: UInt32 x 1: v27) 
 // (op 1085)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1085): v26
v_lshlrev_b32 v26, 4, v27 // (op 1085) 
 // (op 1085) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1085): v27

 // (op 1085) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1085): v27
v_lshrrev_b32 v27, 2, v1 // (op 1085) 
 // (op 1085) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1085): v28
v_lshl_add_u32 v28, v27, 6, v26 // (op 1085) 
 // (op 1085) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1085): v26

 // Freeing : 1 VGPR (Value: UInt32) (op 1085): v27

// Allocated DataFlowTag662: 1 SGPR (Value: UInt32) (op 1085): s0
s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1085) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1085): v28

 // (op 1081) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1085) END
 // (op 1081) Assign VGPR 1024:U32(1086) BEGIN
 // (op 1086) Assign dim(663) = 1024:U32
 // (op 1086) Assign stride expression
 // (op 1081) Assign VGPR 1024:U32(1086) END
 // (op 1081) NOP(1079) BEGIN
 // (op 1079) generate({})
 // (op 1079) end: generate({})
 // (op 1081) NOP(1079) END
 // (op 1081) Assign VGPR 1:U32(1090) BEGIN
 // (op 1090) Assign dim(665) = 1:U32
 // (op 1090) Assign stride expression
 // (op 1081) Assign VGPR 1:U32(1090) END
 // (op 1081) LoadTileDirect2LDS Value: BF8x4(497) BEGIN
 // (op 497) GEN: LoadTileDirect2LDS
 // (op 497) GEN: loadMacroTileDirect2LDS OP 497 LDS 472 MacroTile 470
 // (op 497) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 497) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride519: VGPR Value: UInt64 x 1: (unallocated)
 // (op 497) reg expression
 // (op 497) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 497) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 497)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 497)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 497): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 497) 
// Allocated Stride519: 2 VGPRs (Value: UInt64) (op 497): v26, v27
v_mov_b64 v[26:27], s[14:15] // (op 497) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 497): s14, s15

 // (op 497) Generate 1:U64 into nullptr
 // (op 497) Generate 1024:U32 into nullptr
 // (op 497) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 4096
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride519: VGPR Value: UInt64 x 1: v[26:27]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 497): v28, v29
v_mov_b64 v[28:29], v[12:13] // (op 497) 
 // (op 497) Generate Add(4096:U32, {DataFlowTag662Flatten: s0:U32})U32 into nullptr
 // (op 497) reg expression
 // (op 497) Add(4096:U32, {DataFlowTag662Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 497): s1
s_add_u32 s1, 4096, s0 // (op 497) 
 // (op 497)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride519: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 497) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 497) reg expression
buffer_load_dwordx4 v28, s[44:47], 64 offen offset: 0 lds // (op 497) Load value direct to lds
// VMEM: Expected stall of 463. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1760 (current 1280)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 497) Generate Add(v[28:29]:U64, v26:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 497) reg expression
 // (op 497) reg expression
 // (op 497) Add(v[28:29]:U64, v26:R)U64
 // (op 497) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 497)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 497)      rhs (VGPR Value: Raw32 x 1: v26)
// Allocated : 2 SGPRs (Value: Bool64) (op 497): s14, s15
v_add_co_u32 v28, s[14:15], v28, v26 // (op 497) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 497) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 497): s14, s15

 // (op 497) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 497) reg expression
 // (op 497) Add(m0:U32, 1024:U32)U32
 // (op 497) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 497) 
 // (op 497) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[44:47], 64 offen offset: 0 lds // (op 497) Load value direct to lds
// VMEM: Expected complete at 1764 (current 1284)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // Freeing Stride519: 2 VGPRs (Value: UInt64) (op 497): v26, v27

 // Freeing : 2 VGPRs (Value: UInt64) (op 497): v28, v29

 // Freeing : 1 SGPR (Value: UInt32) (op 497): s1

 // (op 1081) LoadTileDirect2LDS Value: BF8x4(497) END
 // (op 1081) Deallocate{}(1377) BEGIN
 // (op 1377) Deallocate 665
 // Deleting tag 665
 // (op 1377) Deallocate 663
 // Deleting tag 663
 // (op 1377) Deallocate 664
 // Deleting tag 664
 // (op 1377) Deallocate 662
 // Freeing DataFlowTag662: 1 SGPR (Value: UInt32) (op 1085): s0

 // Deleting tag 662
 // (op 1081) Deallocate{}(1377) END
 // (op 1081) end: generate({1078})
 // (op 1081) Unlock Scope 1081
 // Unlock None
 // (op 377) Scope(1081) END
 // (op 377) end: generate({1081})
 // (op 530) SetCoordinate(377) END
 // (op 530) Deallocate{}(1349) BEGIN
 // (op 1349) Deallocate 519
 // Deleting tag 519
 // (op 1349) Deallocate 521
 // Deleting tag 521
 // (op 1349) Deallocate 518
 // Freeing DataFlowTag518: 2 VGPRs (Value: UInt64) (op 602): v12, v13

 // Deleting tag 518
 // (op 1349) Deallocate 520
 // Deleting tag 520
 // (op 1349) Deallocate 517
 // Deleting tag 517
 // (op 530) Deallocate{}(1349) END
 // (op 530) SetCoordinate(380) BEGIN
 // (op 380) generate({})
 // (op 380) end: generate({})
 // (op 380) generate({381})
 // (op 380) LoadTiled Value: E8M0(381) BEGIN
 // (op 381) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 381) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 381) Generate 1:U64 into nullptr
 // tag 478: v**UNALLOCATED**
 // (op 381) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 381) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag556: VGPR Value: UInt32 x 1: v24
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 381) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 381): v12
buffer_load_ubyte v12, v24, s[52:55], 0 offen offset:2 // (op 381) Load value
// VMEM: Expected complete at 1765 (current 1285)
 // (op 380) LoadTiled Value: E8M0(381) END
 // (op 380) end: generate({381})
 // (op 530) SetCoordinate(380) END
 // (op 530) Deallocate{}(1351) BEGIN
 // (op 1351) Deallocate 556
 // Freeing DataFlowTag556: 1 VGPR (Value: UInt32) (op 730): v24

 // Deleting tag 556
 // (op 1351) Deallocate 555
 // Deleting tag 555
 // (op 1351) Deallocate 558
 // Deleting tag 558
 // (op 1351) Deallocate 557
 // Deleting tag 557
 // (op 1351) Deallocate 559
 // Deleting tag 559
 // (op 530) Deallocate{}(1351) END
 // (op 530) SetCoordinate(383) BEGIN
 // (op 383) generate({})
 // (op 383) end: generate({})
 // (op 383) generate({1042})
 // (op 383) Scope(1042) BEGIN
 // (op 1042) Lock Scope 1042
 // Lock Branch
 // (op 1042) generate({1039})
 // (op 1042) NOP(1039) BEGIN
 // (op 1039) generate({})
 // (op 1039) end: generate({})
 // (op 1042) NOP(1039) END
 // (op 1042) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1046) BEGIN
 // (op 1046) Assign dim(650) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 650: s**UNALLOCATED**
 // (op 1046) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag650: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1046) reg expression
 // (op 1046) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1046): v13
v_and_b32 v13, -4, v1 // (op 1046) 
 // (op 1046) BEGIN: Tile[1]
 // (op 1046) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1046): v24
v_sub_u32 v24, v1, v13 // (op 1046) 
 // (op 1046) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1046): v13

 // (op 1046) BEGIN: Flatten
 // (op 1046) {Flatten: ShiftL({Tile[1]: v24:U32}, 4:U32)U32}
 // (op 1046) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1046)         value (Tile[1]: VGPR Value: UInt32 x 1: v24) 
 // (op 1046)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1046): v13
v_lshlrev_b32 v13, 4, v24 // (op 1046) 
 // (op 1046) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1046): v24

 // (op 1046) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1046): v24
v_lshrrev_b32 v24, 2, v1 // (op 1046) 
 // (op 1046) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1046): v26
v_lshl_add_u32 v26, v24, 6, v13 // (op 1046) 
 // (op 1046) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1046): v13

 // Freeing : 1 VGPR (Value: UInt32) (op 1046): v24

// Allocated DataFlowTag650: 1 SGPR (Value: UInt32) (op 1046): s0
s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1046) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1046): v26

 // (op 1042) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1046) END
 // (op 1042) Assign VGPR 1024:U32(1047) BEGIN
 // (op 1047) Assign dim(651) = 1024:U32
 // (op 1047) Assign stride expression
 // (op 1042) Assign VGPR 1024:U32(1047) END
 // (op 1042) NOP(1040) BEGIN
 // (op 1040) generate({})
 // (op 1040) end: generate({})
 // (op 1042) NOP(1040) END
 // (op 1042) Assign VGPR 1:U32(1051) BEGIN
 // (op 1051) Assign dim(653) = 1:U32
 // (op 1051) Assign stride expression
 // (op 1042) Assign VGPR 1:U32(1051) END
 // (op 1042) LoadTileDirect2LDS Value: BF8x4(499) BEGIN
 // (op 499) GEN: LoadTileDirect2LDS
 // (op 499) GEN: loadMacroTileDirect2LDS OP 499 LDS 468 MacroTile 466
 // (op 499) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 499) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride500: VGPR Value: UInt64 x 1: (unallocated)
 // (op 499) reg expression
 // (op 499) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 499) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 499)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 499)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 499): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 499) 
// Allocated Stride500: 2 VGPRs (Value: UInt64) (op 499): v26, v27
v_mov_b64 v[26:27], s[14:15] // (op 499) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 499): s14, s15

 // (op 499) Generate 1:U64 into nullptr
 // (op 499) Generate 1024:U32 into nullptr
 // (op 499) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 6144
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride500: VGPR Value: UInt64 x 1: v[26:27]
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
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 499): v28, v29
v_mov_b64 v[28:29], v[16:17] // (op 499) 
 // (op 499) Generate Add(6144:U32, {DataFlowTag650Flatten: s0:U32})U32 into nullptr
 // (op 499) reg expression
 // (op 499) Add(6144:U32, {DataFlowTag650Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 499): s1
s_add_u32 s1, 6144, s0 // (op 499) 
 // (op 499)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride500: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 499) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 499) reg expression
buffer_load_dwordx4 v28, s[40:43], 64 offen offset: 0 lds // (op 499) Load value direct to lds
// VMEM: Expected stall of 463. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2241 (current 1761)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 499) Generate Add(v[28:29]:U64, v26:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 499) reg expression
 // (op 499) reg expression
 // (op 499) Add(v[28:29]:U64, v26:R)U64
 // (op 499) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 499)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 499)      rhs (VGPR Value: Raw32 x 1: v26)
// Allocated : 2 SGPRs (Value: Bool64) (op 499): s14, s15
v_add_co_u32 v28, s[14:15], v28, v26 // (op 499) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 499) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 499): s14, s15

 // (op 499) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 499) reg expression
 // (op 499) Add(m0:U32, 1024:U32)U32
 // (op 499) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 499) 
 // (op 499) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[40:43], 64 offen offset: 0 lds // (op 499) Load value direct to lds
// VMEM: Expected complete at 2245 (current 1765)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // Freeing Stride500: 2 VGPRs (Value: UInt64) (op 499): v26, v27

 // Freeing : 2 VGPRs (Value: UInt64) (op 499): v28, v29

 // Freeing : 1 SGPR (Value: UInt32) (op 499): s1

 // (op 1042) LoadTileDirect2LDS Value: BF8x4(499) END
 // (op 1042) Deallocate{}(1379) BEGIN
 // (op 1379) Deallocate 650
 // Freeing DataFlowTag650: 1 SGPR (Value: UInt32) (op 1046): s0

 // Deleting tag 650
 // (op 1379) Deallocate 652
 // Deleting tag 652
 // (op 1379) Deallocate 651
 // Deleting tag 651
 // (op 1379) Deallocate 653
 // Deleting tag 653
 // (op 1042) Deallocate{}(1379) END
 // (op 1042) end: generate({1039})
 // (op 1042) Unlock Scope 1042
 // Unlock None
 // (op 383) Scope(1042) END
 // (op 383) end: generate({1042})
 // (op 530) SetCoordinate(383) END
 // (op 530) Deallocate{}(1353) BEGIN
 // (op 1353) Deallocate 502
 // Deleting tag 502
 // (op 1353) Deallocate 499
 // Freeing DataFlowTag499: 2 VGPRs (Value: UInt64) (op 538): v16, v17

 // Deleting tag 499
 // (op 1353) Deallocate 498
 // Deleting tag 498
 // (op 1353) Deallocate 500
 // Deleting tag 500
 // (op 1353) Deallocate 501
 // Deleting tag 501
 // (op 530) Deallocate{}(1353) END
 // (op 530) SetCoordinate(386) BEGIN
 // (op 386) generate({})
 // (op 386) end: generate({})
 // (op 386) generate({387})
 // (op 386) LoadTiled Value: E8M0(387) BEGIN
 // (op 387) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 387) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 387) Generate 1:U64 into nullptr
 // tag 474: v**UNALLOCATED**
 // (op 387) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 387) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag537: VGPR Value: UInt32 x 1: v20
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
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 387) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 387): v13
buffer_load_ubyte v13, v20, s[48:51], 0 offen offset:2 // (op 387) Load value
// VMEM: Expected complete at 2246 (current 1766)
 // (op 386) LoadTiled Value: E8M0(387) END
 // (op 386) end: generate({387})
 // (op 530) SetCoordinate(386) END
 // (op 530) Deallocate{}(1355) BEGIN
 // (op 1355) Deallocate 538
 // Deleting tag 538
 // (op 1355) Deallocate 536
 // Deleting tag 536
 // (op 1355) Deallocate 537
 // Freeing DataFlowTag537: 1 VGPR (Value: UInt32) (op 666): v20

 // Deleting tag 537
 // (op 1355) Deallocate 540
 // Deleting tag 540
 // (op 1355) Deallocate 539
 // Deleting tag 539
 // (op 530) Deallocate{}(1355) END
 // (op 530) NOP(489) BEGIN
 // (op 489) generate({})
 // (op 489) end: generate({})
 // (op 530) NOP(489) END
 // (op 530) Scope(968) BEGIN
 // (op 968) Lock Scope 968
 // Lock Branch
 // (op 968) generate({965})
 // (op 968) NOP(965) BEGIN
 // (op 965) generate({})
 // (op 965) end: generate({})
 // (op 968) NOP(965) END
 // (op 968) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(973) BEGIN
 // (op 973) Assign dim(628) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 628: v**UNALLOCATED**
 // (op 973) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag628: VGPR Value: UInt32 x 1: (unallocated)
 // (op 973) reg expression
 // (op 973) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 973): v20
v_and_b32 v20, -2, v1 // (op 973) 
 // (op 973) BEGIN: Tile[1]Flatten
 // (op 973) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 973): v24
v_sub_u32 v24, v1, v20 // (op 973) 
 // (op 973) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 973): v20

 // (op 973) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 973): v20
v_and_b32 v20, 4294967294, v1 // (op 973) 
 // (op 973) BEGIN: Flatten
 // (op 973) {Flatten: Add(v20:U32, {Tile[1]Flatten: v24:U32})U32}
// Allocated DataFlowTag628: 1 VGPR (Value: UInt32) (op 973): v16
v_add_u32 v16, v20, v24 // (op 973) 
 // (op 973) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 973): v24

 // Freeing : 1 VGPR (Value: UInt32) (op 973): v20

 // (op 968) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(973) END
 // (op 968) Assign VGPR 2:U32(974) BEGIN
 // (op 974) Assign dim(629) = 2:U32
 // (op 974) Assign stride expression
 // (op 968) Assign VGPR 2:U32(974) END
 // (op 968) NOP(966) BEGIN
 // (op 966) generate({})
 // (op 966) end: generate({})
 // (op 968) NOP(966) END
 // (op 968) Assign VGPR 1:U32(978) BEGIN
 // (op 978) Assign dim(631) = 1:U32
 // (op 978) Assign stride expression
 // (op 968) Assign VGPR 1:U32(978) END
 // (op 968) StoreLDSTile Value: E8M0(390) BEGIN
 // (op 390) GEN: StoreLDSTile
 // (op 390) GEN: storeMacroTileLDS OP 390 LDS 41 MacroTile 11
 // (op 390) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 390) Generate 1:U32 into nullptr
 // (op 390) 	Dir = Store
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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

 // (op 390) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = DataFlowTag628: VGPR Value: UInt32 x 1: v16
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 390) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v16, v22 offset:8192 // (op 390) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1867 (current 1772)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 968) StoreLDSTile Value: E8M0(390) END
 // (op 968) Deallocate{}(1357) BEGIN
 // (op 1357) Deallocate 631
 // Deleting tag 631
 // (op 1357) Deallocate 630
 // Deleting tag 630
 // (op 1357) Deallocate 629
 // Deleting tag 629
 // (op 1357) Deallocate 628
 // Freeing DataFlowTag628: 1 VGPR (Value: UInt32) (op 973): v16

 // Deleting tag 628
 // (op 968) Deallocate{}(1357) END
 // (op 968) end: generate({965})
 // (op 968) Unlock Scope 968
 // Unlock None
 // (op 530) Scope(968) END
 // (op 530) NOP(494) BEGIN
 // (op 494) generate({})
 // (op 494) end: generate({})
 // (op 530) NOP(494) END
 // (op 530) Scope(915) BEGIN
 // (op 915) Lock Scope 915
 // Lock Branch
 // (op 915) generate({912})
 // (op 915) NOP(912) BEGIN
 // (op 912) generate({})
 // (op 912) end: generate({})
 // (op 915) NOP(912) END
 // (op 915) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(920) BEGIN
 // (op 920) Assign dim(612) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 612: v**UNALLOCATED**
 // (op 920) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag612: VGPR Value: UInt32 x 1: (unallocated)
 // (op 920) reg expression
 // (op 920) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 920): v20
v_and_b32 v20, -2, v1 // (op 920) 
 // (op 920) BEGIN: Tile[1]Flatten
 // (op 920) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 920): v24
v_sub_u32 v24, v1, v20 // (op 920) 
 // (op 920) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 920): v20

 // (op 920) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 920): v20
v_and_b32 v20, 4294967294, v1 // (op 920) 
 // (op 920) BEGIN: Flatten
 // (op 920) {Flatten: Add(v20:U32, {Tile[1]Flatten: v24:U32})U32}
// Allocated DataFlowTag612: 1 VGPR (Value: UInt32) (op 920): v16
v_add_u32 v16, v20, v24 // (op 920) 
 // (op 920) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 920): v24

 // Freeing : 1 VGPR (Value: UInt32) (op 920): v20

 // (op 915) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(920) END
 // (op 915) Assign VGPR 2:U32(921) BEGIN
 // (op 921) Assign dim(613) = 2:U32
 // (op 921) Assign stride expression
 // (op 915) Assign VGPR 2:U32(921) END
 // (op 915) NOP(913) BEGIN
 // (op 913) generate({})
 // (op 913) end: generate({})
 // (op 915) NOP(913) END
 // (op 915) Assign VGPR 1:U32(925) BEGIN
 // (op 925) Assign dim(615) = 1:U32
 // (op 925) Assign stride expression
 // (op 915) Assign VGPR 1:U32(925) END
 // (op 915) StoreLDSTile Value: E8M0(392) BEGIN
 // (op 392) GEN: StoreLDSTile
 // (op 392) GEN: storeMacroTileLDS OP 392 LDS 33 MacroTile 10
 // (op 392) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 392) Generate 1:U32 into nullptr
 // (op 392) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
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
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

 // (op 392) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v23
 // 	info.rowOffsetReg = DataFlowTag612: VGPR Value: UInt32 x 1: v16
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 392) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v16, v23 offset:8256 // (op 392) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1873 (current 1778)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 915) StoreLDSTile Value: E8M0(392) END
 // (op 915) Deallocate{}(1359) BEGIN
 // (op 1359) Deallocate 615
 // Deleting tag 615
 // (op 1359) Deallocate 612
 // Freeing DataFlowTag612: 1 VGPR (Value: UInt32) (op 920): v16

 // Deleting tag 612
 // (op 1359) Deallocate 613
 // Deleting tag 613
 // (op 1359) Deallocate 614
 // Deleting tag 614
 // (op 915) Deallocate{}(1359) END
 // (op 915) end: generate({912})
 // (op 915) Unlock Scope 915
 // Unlock None
 // (op 530) Scope(915) END
 // (op 530) Barrier(1203) BEGIN
s_barrier  // (op 1203) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 530) Barrier(1203) END
 // (op 530) SetCoordinate(408) BEGIN
 // (op 408) generate({})
 // (op 408) end: generate({})
 // (op 408) generate({409})
 // (op 408) LoadLDSTile{Value: BF8x4}(409) BEGIN
 // (op 409) GEN: loadMacroTileWAVELDS OP 409 LDS 25 WaveTile 169
 // (op 409) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 409) Generate 1:U32 into nullptr
 // tag 23: v**UNALLOCATED**
 // (op 409) 	Dir = Load
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

 // (op 409) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag600: VGPR Value: UInt32 x 1: v4
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
 // (op 409) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated : 8 VGPRs (Value: BF8x4) (op 409): v26, v27, v28, v29, v30, v31, v32, v33
ds_read_b128 v[26:29], v4 // (op 409) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 1875 (current 1780)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
ds_read_b128 v[30:33], v4 offset:32 // (op 409) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 87. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1963 (current 1868)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 408) LoadLDSTile{Value: BF8x4}(409) END
 // (op 408) end: generate({409})
 // (op 530) SetCoordinate(408) END
 // (op 530) Barrier(1237) BEGIN
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1237) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 530) Barrier(1237) END
 // (op 530) SetCoordinate(414) BEGIN
 // (op 414) generate({})
 // (op 414) end: generate({})
 // (op 414) generate({415})
 // (op 414) LoadLDSTile{Value: E8M0}(415) BEGIN
 // (op 415) GEN: loadMacroTileWAVELDS OP 415 LDS 41 WaveTile 295
 // (op 415) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 415) Generate 1:U32 into nullptr
 // tag 39: v**UNALLOCATED**
 // (op 415) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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

 // (op 415) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag632: VGPR Value: UInt32 x 1: v6
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 415) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 415): v20
ds_read_u8 v20, v6 offset:8192 // (op 415) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected stall of 3. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1969 (current 1874)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 414) LoadLDSTile{Value: E8M0}(415) END
 // (op 414) end: generate({415})
 // (op 530) SetCoordinate(414) END
 // (op 530) Barrier(1187) BEGIN
s_barrier  // (op 1187) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 530) Barrier(1187) END
 // (op 530) SetCoordinate(405) BEGIN
 // (op 405) generate({})
 // (op 405) end: generate({})
 // (op 405) generate({406})
 // (op 405) LoadLDSTile{Value: BF8x4}(406) BEGIN
 // (op 406) GEN: loadMacroTileWAVELDS OP 406 LDS 17 WaveTile 103
 // (op 406) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 406) Generate 1:U32 into nullptr
 // tag 15: v**UNALLOCATED**
 // (op 406) 	Dir = Load
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

 // (op 406) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag584: VGPR Value: UInt32 x 1: v3
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
 // (op 406) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated : 8 VGPRs (Value: BF8x4) (op 406): v34, v35, v36, v37, v38, v39, v40, v41
ds_read_b128 v[34:37], v3 offset:2048 // (op 406) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 1971 (current 1876)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
ds_read_b128 v[38:41], v3 offset:2080 // (op 406) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 87. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 2059 (current 1964)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 405) LoadLDSTile{Value: BF8x4}(406) END
 // (op 405) end: generate({406})
 // (op 530) SetCoordinate(405) END
 // (op 530) Barrier(1220) BEGIN
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1220) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 530) Barrier(1220) END
 // (op 530) SetCoordinate(411) BEGIN
 // (op 411) generate({})
 // (op 411) end: generate({})
 // (op 411) generate({412})
 // (op 411) LoadLDSTile{Value: E8M0}(412) BEGIN
 // (op 412) GEN: loadMacroTileWAVELDS OP 412 LDS 33 WaveTile 235
 // (op 412) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 412) Generate 1:U32 into nullptr
 // tag 31: v**UNALLOCATED**
 // (op 412) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

 // (op 412) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag616: VGPR Value: UInt32 x 1: v0
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 412) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 412): v24
ds_read_u8 v24, v0 offset:8256 // (op 412) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected stall of 3. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 2065 (current 1970)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 411) LoadLDSTile{Value: E8M0}(412) END
 // (op 411) end: generate({412})
 // (op 530) SetCoordinate(411) END
 // (op 530) NOP(363) BEGIN
 // (op 363) generate({})
 // (op 363) end: generate({})
 // (op 530) NOP(363) END
 // (op 530) NOP(783) BEGIN
 // (op 783) generate({})
 // (op 783) end: generate({})
 // (op 530) NOP(783) END
 // (op 530) NOP(780) BEGIN
 // (op 780) generate({})
 // (op 780) end: generate({})
 // (op 530) NOP(780) END
 // (op 530) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(787) BEGIN
 // (op 787) Assign dim(572) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 572: s**UNALLOCATED**
 // (op 787) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag572: SGPR Value: UInt32 x 1: (unallocated)
 // (op 787) reg expression
 // (op 787) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 787): v16
v_and_b32 v16, -4, v1 // (op 787) 
 // (op 787) BEGIN: Tile[1]
 // (op 787) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 787): v17
v_sub_u32 v17, v1, v16 // (op 787) 
 // (op 787) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 787): v16

 // (op 787) BEGIN: Flatten
 // (op 787) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 787) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 787)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 787)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 787): v16
v_lshlrev_b32 v16, 4, v17 // (op 787) 
 // (op 787) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 787): v17

 // (op 787) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 787): v17
v_lshrrev_b32 v17, 2, v1 // (op 787) 
 // (op 787) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 787): v42
v_lshl_add_u32 v42, v17, 6, v16 // (op 787) 
 // (op 787) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 787): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 787): v17

// Allocated DataFlowTag572: 1 SGPR (Value: UInt32) (op 787): s0
s_nop 0
v_readfirstlane_b32 s0, v42 // (op 787) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 787): v42

 // (op 530) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(787) END
 // (op 530) Assign VGPR 1024:U32(788) BEGIN
 // (op 788) Assign dim(573) = 1024:U32
 // (op 788) Assign stride expression
 // (op 530) Assign VGPR 1024:U32(788) END
 // (op 530) NOP(781) BEGIN
 // (op 781) generate({})
 // (op 781) end: generate({})
 // (op 530) NOP(781) END
 // (op 530) Assign VGPR 1:U32(792) BEGIN
 // (op 792) Assign dim(575) = 1:U32
 // (op 792) Assign stride expression
 // (op 530) Assign VGPR 1:U32(792) END
 // (op 530) NOP(834) BEGIN
 // (op 834) generate({})
 // (op 834) end: generate({})
 // (op 530) NOP(834) END
 // (op 530) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(839) BEGIN
 // (op 839) Assign dim(588) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 588: s**UNALLOCATED**
 // (op 839) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag588: SGPR Value: UInt32 x 1: (unallocated)
 // (op 839) reg expression
 // (op 839) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 839): v16
v_and_b32 v16, -4, v1 // (op 839) 
 // (op 839) BEGIN: Tile[1]
 // (op 839) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 839): v17
v_sub_u32 v17, v1, v16 // (op 839) 
 // (op 839) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 839): v16

 // (op 839) BEGIN: Flatten
 // (op 839) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 839) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 839)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 839)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 839): v16
v_lshlrev_b32 v16, 4, v17 // (op 839) 
 // (op 839) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 839): v17

 // (op 839) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 839): v17
v_lshrrev_b32 v17, 2, v1 // (op 839) 
 // (op 839) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 839): v42
v_lshl_add_u32 v42, v17, 6, v16 // (op 839) 
 // (op 839) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 839): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 839): v17

// Allocated DataFlowTag588: 1 SGPR (Value: UInt32) (op 839): s1
s_nop 0
v_readfirstlane_b32 s1, v42 // (op 839) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 839): v42

 // (op 530) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(839) END
 // (op 530) Assign VGPR 1024:U32(840) BEGIN
 // (op 840) Assign dim(589) = 1024:U32
 // (op 840) Assign stride expression
 // (op 530) Assign VGPR 1024:U32(840) END
 // (op 530) NOP(835) BEGIN
 // (op 835) generate({})
 // (op 835) end: generate({})
 // (op 530) NOP(835) END
 // (op 530) Assign VGPR 1:U32(844) BEGIN
 // (op 844) Assign dim(591) = 1:U32
 // (op 844) Assign stride expression
 // (op 530) Assign VGPR 1:U32(844) END
 // (op 530) NOP(886) BEGIN
 // (op 886) generate({})
 // (op 886) end: generate({})
 // (op 530) NOP(886) END
 // (op 530) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(891) BEGIN
 // (op 891) Assign dim(604) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 604: v**UNALLOCATED**
 // (op 891) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag604: VGPR Value: UInt32 x 1: (unallocated)
 // (op 891) reg expression
 // (op 891) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 891): v16
v_and_b32 v16, -2, v1 // (op 891) 
 // (op 891) BEGIN: Tile[1]Flatten
 // (op 891) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 891): v17
v_sub_u32 v17, v1, v16 // (op 891) 
 // (op 891) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 891): v16

 // (op 891) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 891): v16
v_and_b32 v16, 4294967294, v1 // (op 891) 
 // (op 891) BEGIN: Flatten
 // (op 891) {Flatten: Add(v16:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag604: 1 VGPR (Value: UInt32) (op 891): v42
v_add_u32 v42, v16, v17 // (op 891) 
 // (op 891) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 891): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 891): v16

 // (op 530) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(891) END
 // (op 530) Assign VGPR 2:U32(892) BEGIN
 // (op 892) Assign dim(605) = 2:U32
 // (op 892) Assign stride expression
 // (op 530) Assign VGPR 2:U32(892) END
 // (op 530) NOP(887) BEGIN
 // (op 887) generate({})
 // (op 887) end: generate({})
 // (op 530) NOP(887) END
 // (op 530) Assign VGPR 1:U32(896) BEGIN
 // (op 896) Assign dim(607) = 1:U32
 // (op 896) Assign stride expression
 // (op 530) Assign VGPR 1:U32(896) END
 // (op 530) NOP(939) BEGIN
 // (op 939) generate({})
 // (op 939) end: generate({})
 // (op 530) NOP(939) END
 // (op 530) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(944) BEGIN
 // (op 944) Assign dim(620) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 620: v**UNALLOCATED**
 // (op 944) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag620: VGPR Value: UInt32 x 1: (unallocated)
 // (op 944) reg expression
 // (op 944) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 944): v16
v_and_b32 v16, -2, v1 // (op 944) 
 // (op 944) BEGIN: Tile[1]Flatten
 // (op 944) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 944): v17
v_sub_u32 v17, v1, v16 // (op 944) 
 // (op 944) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 944): v16

 // (op 944) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 944): v16
v_and_b32 v16, 4294967294, v1 // (op 944) 
 // (op 944) BEGIN: Flatten
 // (op 944) {Flatten: Add(v16:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag620: 1 VGPR (Value: UInt32) (op 944): v43
v_add_u32 v43, v16, v17 // (op 944) 
 // (op 944) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 944): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 944): v16

 // (op 530) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(944) END
 // (op 530) Assign VGPR 2:U32(945) BEGIN
 // (op 945) Assign dim(621) = 2:U32
 // (op 945) Assign stride expression
 // (op 530) Assign VGPR 2:U32(945) END
 // (op 530) NOP(940) BEGIN
 // (op 940) generate({})
 // (op 940) end: generate({})
 // (op 530) NOP(940) END
 // (op 530) Assign VGPR 1:U32(949) BEGIN
 // (op 949) Assign dim(623) = 1:U32
 // (op 949) Assign stride expression
 // (op 530) Assign VGPR 1:U32(949) END
 // (op 530) NOP(1026) BEGIN
 // (op 1026) generate({})
 // (op 1026) end: generate({})
 // (op 530) NOP(1026) END
 // (op 530) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1031) BEGIN
 // (op 1031) Assign dim(646) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 646: s**UNALLOCATED**
 // (op 1031) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag646: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1031) reg expression
 // (op 1031) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v16
v_and_b32 v16, -4, v1 // (op 1031) 
 // (op 1031) BEGIN: Tile[1]
 // (op 1031) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v17
v_sub_u32 v17, v1, v16 // (op 1031) 
 // (op 1031) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1031): v16

 // (op 1031) BEGIN: Flatten
 // (op 1031) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 1031) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1031)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 1031)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v16
v_lshlrev_b32 v16, 4, v17 // (op 1031) 
 // (op 1031) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1031): v17

 // (op 1031) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v17
v_lshrrev_b32 v17, 2, v1 // (op 1031) 
 // (op 1031) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v44
v_lshl_add_u32 v44, v17, 6, v16 // (op 1031) 
 // (op 1031) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1031): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 1031): v17

// Allocated DataFlowTag646: 1 SGPR (Value: UInt32) (op 1031): s14
s_nop 0
v_readfirstlane_b32 s14, v44 // (op 1031) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1031): v44

 // (op 530) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1031) END
 // (op 530) Assign VGPR 1024:U32(1032) BEGIN
 // (op 1032) Assign dim(647) = 1024:U32
 // (op 1032) Assign stride expression
 // (op 530) Assign VGPR 1024:U32(1032) END
 // (op 530) NOP(1027) BEGIN
 // (op 1027) generate({})
 // (op 1027) end: generate({})
 // (op 530) NOP(1027) END
 // (op 530) Assign VGPR 1:U32(1036) BEGIN
 // (op 1036) Assign dim(649) = 1:U32
 // (op 1036) Assign stride expression
 // (op 530) Assign VGPR 1:U32(1036) END
 // (op 530) NOP(1065) BEGIN
 // (op 1065) generate({})
 // (op 1065) end: generate({})
 // (op 530) NOP(1065) END
 // (op 530) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1070) BEGIN
 // (op 1070) Assign dim(658) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 658: s**UNALLOCATED**
 // (op 1070) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag658: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1070) reg expression
 // (op 1070) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1070): v16
v_and_b32 v16, -4, v1 // (op 1070) 
 // (op 1070) BEGIN: Tile[1]
 // (op 1070) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1070): v17
v_sub_u32 v17, v1, v16 // (op 1070) 
 // (op 1070) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1070): v16

 // (op 1070) BEGIN: Flatten
 // (op 1070) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 1070) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1070)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 1070)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1070): v16
v_lshlrev_b32 v16, 4, v17 // (op 1070) 
 // (op 1070) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1070): v17

 // (op 1070) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1070): v17
v_lshrrev_b32 v17, 2, v1 // (op 1070) 
 // (op 1070) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1070): v44
v_lshl_add_u32 v44, v17, 6, v16 // (op 1070) 
 // (op 1070) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1070): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 1070): v17

// Allocated DataFlowTag658: 1 SGPR (Value: UInt32) (op 1070): s15
s_nop 0
v_readfirstlane_b32 s15, v44 // (op 1070) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1070): v44

 // (op 530) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1070) END
 // (op 530) Assign VGPR 1024:U32(1071) BEGIN
 // (op 1071) Assign dim(659) = 1024:U32
 // (op 1071) Assign stride expression
 // (op 530) Assign VGPR 1024:U32(1071) END
 // (op 530) NOP(1066) BEGIN
 // (op 1066) generate({})
 // (op 1066) end: generate({})
 // (op 530) NOP(1066) END
 // (op 530) Assign VGPR 1:U32(1075) BEGIN
 // (op 1075) Assign dim(661) = 1:U32
 // (op 1075) Assign stride expression
 // (op 530) Assign VGPR 1:U32(1075) END
 // (op 530) NOP(1104) BEGIN
 // (op 1104) generate({})
 // (op 1104) end: generate({})
 // (op 530) NOP(1104) END
 // (op 530) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(1109) BEGIN
 // (op 1109) Assign dim(670) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 670: v**UNALLOCATED**
 // (op 1109) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag670: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1109) reg expression
 // (op 1109) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1109): v16
v_and_b32 v16, -2, v1 // (op 1109) 
 // (op 1109) BEGIN: Tile[1]Flatten
 // (op 1109) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1109): v17
v_sub_u32 v17, v1, v16 // (op 1109) 
 // (op 1109) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1109): v16

 // (op 1109) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1109): v16
v_and_b32 v16, 4294967294, v1 // (op 1109) 
 // (op 1109) BEGIN: Flatten
 // (op 1109) {Flatten: Add(v16:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag670: 1 VGPR (Value: UInt32) (op 1109): v44
v_add_u32 v44, v16, v17 // (op 1109) 
 // (op 1109) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1109): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 1109): v16

 // (op 530) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(1109) END
 // (op 530) Assign VGPR 2:U32(1110) BEGIN
 // (op 1110) Assign dim(671) = 2:U32
 // (op 1110) Assign stride expression
 // (op 530) Assign VGPR 2:U32(1110) END
 // (op 530) NOP(1105) BEGIN
 // (op 1105) generate({})
 // (op 1105) end: generate({})
 // (op 530) NOP(1105) END
 // (op 530) Assign VGPR 1:U32(1114) BEGIN
 // (op 1114) Assign dim(673) = 1:U32
 // (op 1114) Assign stride expression
 // (op 530) Assign VGPR 1:U32(1114) END
 // (op 530) NOP(1128) BEGIN
 // (op 1128) generate({})
 // (op 1128) end: generate({})
 // (op 530) NOP(1128) END
 // (op 530) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(1133) BEGIN
 // (op 1133) Assign dim(678) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 678: v**UNALLOCATED**
 // (op 1133) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag678: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1133) reg expression
 // (op 1133) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1133): v16
v_and_b32 v16, -2, v1 // (op 1133) 
 // (op 1133) BEGIN: Tile[1]Flatten
 // (op 1133) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1133): v17
v_sub_u32 v17, v1, v16 // (op 1133) 
 // (op 1133) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1133): v16

 // (op 1133) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1133): v16
v_and_b32 v16, 4294967294, v1 // (op 1133) 
 // (op 1133) BEGIN: Flatten
 // (op 1133) {Flatten: Add(v16:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag678: 1 VGPR (Value: UInt32) (op 1133): v45
v_add_u32 v45, v16, v17 // (op 1133) 
 // (op 1133) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1133): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 1133): v16

 // (op 530) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(1133) END
 // (op 530) Assign VGPR 2:U32(1134) BEGIN
 // (op 1134) Assign dim(679) = 2:U32
 // (op 1134) Assign stride expression
 // (op 530) Assign VGPR 2:U32(1134) END
 // (op 530) NOP(1129) BEGIN
 // (op 1129) generate({})
 // (op 1129) end: generate({})
 // (op 530) NOP(1129) END
 // (op 530) Assign VGPR 1:U32(1138) BEGIN
 // (op 1138) Assign dim(681) = 1:U32
 // (op 1138) Assign stride expression
 // (op 530) Assign VGPR 1:U32(1138) END
 // (op 530) ForLoopOp KLoop: LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(58) BEGIN
 // (op 58) Initialize For Loop
 // (op 58) generate({59})
 // (op 58) Assign SGPR 0:U32(59) BEGIN
 // (op 59) Assign dim(395) = 0:U32
 // tag 395: s**UNALLOCATED**
 // (op 59) Generate 0:U32 into DataFlowTag395: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag395: 1 SGPR (Value: UInt32) (op 59): s18
s_mov_b32 s18, 0 // (op 59) call()
 // (op 58) Assign SGPR 0:U32(59) END
 // (op 58) end: generate({59})
 // (op 58) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 58) 
 // (op 58) Generate LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 58) LessThan({DataFlowTag395: s18:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 58) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 58) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 58): s38, s39
s_ashr_i64 s[38:39], s[28:29], 6 // (op 58) 
// Allocated : 1 SGPR (Value: UInt32) (op 58): s19
s_mov_b32 s19, s38 // (op 58) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 58): s38, s39

// Allocated : 1 SGPR (Value: UInt32) (op 58): s38
s_and_b32 s38, s19, 4294967294 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s19

s_cmp_lt_u32 s18, s38 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s38

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 58) 
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58:
 // (op 58) 
 // (op 58) generate({60, 88, 1206})
 // (op 58) Assign SGPR Add(DataFlowTag(395)U32, 2:U32)U32(60) BEGIN
 // (op 60) Assign dim(395) = Add(DataFlowTag(395)U32, 2:U32)U32
 // (op 60) Generate Add(DataFlowTag(395)U32, 2:U32)U32 into DataFlowTag395: SGPR Value: UInt32 x 1: s18
 // (op 60) reg expression
 // (op 60) Add({DataFlowTag395: s18:U32}, 2:U32)U32
 // (op 58) Multiply(88) BEGIN
 // (op 88) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile4xBF8, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 58) Barrier(1206) BEGIN
s_add_u32 s18, s18, 2 // (op 60) 
 // (op 58) Assign SGPR Add(DataFlowTag(395)U32, 2:U32)U32(60) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[34:41], v[26:33], a[0:15], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 88) 
 // (op 58) Multiply(88) END
s_barrier  // (op 1206) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 58) Barrier(1206) END
 // (op 58) SetCoordinate(288) BEGIN
 // (op 288) generate({})
 // (op 288) end: generate({})
 // (op 288) generate({459})
 // (op 288) LoadTileDirect2LDS Value: BF8x4(459) BEGIN
 // (op 459) GEN: LoadTileDirect2LDS
 // (op 459) GEN: loadMacroTileDirect2LDS OP 459 LDS 25 MacroTile 9
 // (op 459) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 459) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride525: VGPR Value: UInt64 x 1: (unallocated)
 // (op 459) reg expression
 // (op 459) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 459) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 459)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 459)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 459): s38, s39
s_lshl_b64 s[38:39], s[8:9], 4 // (op 459) 
// Allocated Stride525: 2 VGPRs (Value: UInt64) (op 459): v16, v17
v_mov_b64 v[16:17], s[38:39] // (op 459) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 459): s38, s39

 // (op 459) Generate 1:U64 into nullptr
 // (op 459) Generate 1024:U32 into nullptr
 // (op 459) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride525: VGPR Value: UInt64 x 1: v[16:17]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 524: v**UNALLOCATED**
// Allocated Offset524: 2 VGPRs (Value: UInt64) (op 459): v46, v47
v_mov_b64 v[46:47], v[14:15] // (op 459) 
// Allocated : 2 VGPRs (Value: UInt64) (op 459): v48, v49
v_mov_b64 v[48:49], v[46:47] // (op 459) 
 // (op 459) Generate Add(0:U32, {DataFlowTag588Flatten: s1:U32})U32 into nullptr
 // (op 459)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride525: v[16:17]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 459) Generate {DataFlowTag588Flatten: s1:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 459) reg expression
 // (op 459) Generate 128:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 128 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 459): s19
s_mov_b32 s19, 128 // (op 459) call()
// 128 is not a supported value!
buffer_load_dwordx4 v48, s[44:47], s19 offen offset: 0 lds // (op 459) Load value direct to lds
// VMEM: Expected stall of 210. CBNW: 3, Inc: 3
// VMEM: Expected complete at 2722 (current 2242)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 459) Generate Add(v[48:49]:U64, v16:R)U64 into VGPR Value: UInt64 x 1: v[48:49]
 // (op 459) reg expression
 // (op 459) reg expression
 // (op 459) Add(v[48:49]:U64, v16:R)U64
 // (op 459) Add: dest (VGPR Value: UInt64 x 1: v[48:49]) = 
 // (op 459)      lhs (VGPR Value: UInt64 x 1: v[48:49]) 
 // (op 459)      rhs (VGPR Value: Raw32 x 1: v16)
// Allocated : 2 SGPRs (Value: Bool64) (op 459): s38, s39
v_add_co_u32 v48, s[38:39], v48, v16 // (op 459) least significant half
v_addc_co_u32 v49, s[38:39], 0, v49, s[38:39] // (op 459) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 459): s38, s39

 // (op 459) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 459) reg expression
 // (op 459) Add(m0:U32, 1024:U32)U32
 // (op 459) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 459) 
 // (op 459) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v48, s[44:47], s19 offen offset: 0 lds // (op 459) Load value direct to lds
// VMEM: Expected complete at 2726 (current 2246)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 459): s19

 // Freeing Stride525: 2 VGPRs (Value: UInt64) (op 459): v16, v17

 // Freeing : 2 VGPRs (Value: UInt64) (op 459): v48, v49

 // (op 288) LoadTileDirect2LDS Value: BF8x4(459) END
 // (op 288) Deallocate{}(1363) BEGIN
 // (op 1363) Deallocate 590
 // Deleting tag 590
 // (op 288) Deallocate{}(1363) END
 // (op 288) end: generate({459})
 // (op 58) SetCoordinate(288) END
 // (op 58) SetCoordinate(298) BEGIN
 // (op 298) generate({})
 // (op 298) end: generate({})
 // (op 298) generate({25})
 // (op 298) LoadTiled Value: E8M0(25) BEGIN
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 562: v**UNALLOCATED**
// Allocated Offset562: 1 VGPR (Value: UInt32) (op 25): v16
v_mov_b32 v16, v25 // (op 25) 
 // (op 25) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = Offset562: VGPR Value: UInt32 x 1: v16
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
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

buffer_load_ubyte v22, v16, s[52:55], 0 offen offset:4 // (op 25) Load value
// VMEM: Expected complete at 2728 (current 2248)
 // (op 298) LoadTiled Value: E8M0(25) END
 // (op 298) end: generate({25})
 // (op 58) SetCoordinate(298) END
 // (op 58) Barrier(1190) BEGIN
s_barrier  // (op 1190) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 58) Barrier(1190) END
 // (op 58) SetCoordinate(283) BEGIN
 // (op 283) generate({})
 // (op 283) end: generate({})
 // (op 283) generate({454})
 // (op 283) LoadTileDirect2LDS Value: BF8x4(454) BEGIN
 // (op 454) GEN: LoadTileDirect2LDS
 // (op 454) GEN: loadMacroTileDirect2LDS OP 454 LDS 17 MacroTile 8
 // (op 454) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 454) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride506: VGPR Value: UInt64 x 1: (unallocated)
 // (op 454) reg expression
 // (op 454) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 454) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 454)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 454)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 454): s38, s39
s_lshl_b64 s[38:39], s[2:3], 4 // (op 454) 
// Allocated Stride506: 2 VGPRs (Value: UInt64) (op 454): v48, v49
v_mov_b64 v[48:49], s[38:39] // (op 454) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 454): s38, s39

 // (op 454) Generate 1:U64 into nullptr
 // (op 454) Generate 1024:U32 into nullptr
 // (op 454) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride506: VGPR Value: UInt64 x 1: v[48:49]
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
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 505: v**UNALLOCATED**
// Allocated Offset505: 2 VGPRs (Value: UInt64) (op 454): v50, v51
v_mov_b64 v[50:51], v[18:19] // (op 454) 
// Allocated : 2 VGPRs (Value: UInt64) (op 454): v52, v53
v_mov_b64 v[52:53], v[50:51] // (op 454) 
 // (op 454) Generate Add(2048:U32, {DataFlowTag572Flatten: s0:U32})U32 into nullptr
 // (op 454) reg expression
 // (op 454) Add(2048:U32, {DataFlowTag572Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 454): s19
s_add_u32 s19, 2048, s0 // (op 454) 
 // (op 454)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride506: v[48:49]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 454) Generate s19:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s19 // (op 454) reg expression
 // (op 454) Generate 128:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 128 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 454): s38
s_mov_b32 s38, 128 // (op 454) call()
// 128 is not a supported value!
buffer_load_dwordx4 v52, s[40:43], s38 offen offset: 0 lds // (op 454) Load value direct to lds
// VMEM: Expected stall of 466. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3203 (current 2723)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 454) Generate Add(v[52:53]:U64, v48:R)U64 into VGPR Value: UInt64 x 1: v[52:53]
 // (op 454) reg expression
 // (op 454) reg expression
 // (op 454) Add(v[52:53]:U64, v48:R)U64
 // (op 454) Add: dest (VGPR Value: UInt64 x 1: v[52:53]) = 
 // (op 454)      lhs (VGPR Value: UInt64 x 1: v[52:53]) 
 // (op 454)      rhs (VGPR Value: Raw32 x 1: v48)
// Allocated : 2 SGPRs (Value: Bool64) (op 454): s56, s57
v_add_co_u32 v52, s[56:57], v52, v48 // (op 454) least significant half
v_addc_co_u32 v53, s[56:57], 0, v53, s[56:57] // (op 454) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 454): s56, s57

 // (op 454) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 454) reg expression
 // (op 454) Add(m0:U32, 1024:U32)U32
 // (op 454) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 454) 
 // (op 454) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v52, s[40:43], s38 offen offset: 0 lds // (op 454) Load value direct to lds
// VMEM: Expected complete at 3207 (current 2727)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 454): s38

 // Freeing Stride506: 2 VGPRs (Value: UInt64) (op 454): v48, v49

 // Freeing : 2 VGPRs (Value: UInt64) (op 454): v52, v53

 // Freeing : 1 SGPR (Value: UInt32) (op 454): s19

 // (op 283) LoadTileDirect2LDS Value: BF8x4(454) END
 // (op 283) Deallocate{}(1361) BEGIN
 // (op 1361) Deallocate 574
 // Deleting tag 574
 // (op 283) Deallocate{}(1361) END
 // (op 283) end: generate({454})
 // (op 58) SetCoordinate(283) END
 // (op 58) SetCoordinate(293) BEGIN
 // (op 293) generate({})
 // (op 293) end: generate({})
 // (op 293) generate({17})
 // (op 293) LoadTiled Value: E8M0(17) BEGIN
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
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 543: v**UNALLOCATED**
// Allocated Offset543: 1 VGPR (Value: UInt32) (op 17): v17
v_mov_b32 v17, v21 // (op 17) 
 // (op 17) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v23
 // 	info.rowOffsetReg = Offset543: VGPR Value: UInt32 x 1: v17
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
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
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

buffer_load_ubyte v23, v17, s[48:51], 0 offen offset:4 // (op 17) Load value
// VMEM: Expected complete at 3209 (current 2729)
 // (op 293) LoadTiled Value: E8M0(17) END
 // (op 293) end: generate({17})
 // (op 58) SetCoordinate(293) END
 // (op 58) NOP(484) BEGIN
 // (op 484) generate({})
 // (op 484) end: generate({})
 // (op 58) NOP(484) END
 // (op 58) Barrier(1276) BEGIN
s_barrier  // (op 1276) 
 // (op 58) Barrier(1276) END
 // (op 58) StoreLDSTile Value: E8M0(255) BEGIN
 // (op 255) GEN: StoreLDSTile
 // (op 255) GEN: storeMacroTileLDS OP 255 LDS 480 MacroTile 478
 // (op 255) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 255) Generate 1:U32 into nullptr
 // (op 255) 	Dir = Store
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
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = DataFlowTag678: VGPR Value: UInt32 x 1: v45
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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

s_waitcnt vmcnt(9)
ds_write_b8 v45, v12 offset:8320 // (op 255) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2870 (current 2775)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8320, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(255) END
 // (op 58) Deallocate{}(1335) BEGIN
 // (op 1335) Deallocate 680
 // Deleting tag 680
 // (op 58) Deallocate{}(1335) END
 // (op 58) NOP(479) BEGIN
 // (op 479) generate({})
 // (op 479) end: generate({})
 // (op 58) NOP(479) END
 // (op 58) Barrier(1270) BEGIN
s_barrier  // (op 1270) 
 // (op 58) Barrier(1270) END
 // (op 58) StoreLDSTile Value: E8M0(254) BEGIN
 // (op 254) GEN: StoreLDSTile
 // (op 254) GEN: storeMacroTileLDS OP 254 LDS 476 MacroTile 474
 // (op 254) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 254) Generate 1:U32 into nullptr
 // (op 254) 	Dir = Store
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
 // 	info.offset = Literal Value: UInt32 x 0: 8384
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

 // (op 254) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = DataFlowTag670: VGPR Value: UInt32 x 1: v44
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
 // 	info.offset = Literal Value: UInt32 x 0: 8384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 254) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v44, v13 offset:8384 // (op 254) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2873 (current 2778)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8384, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(254) END
 // (op 58) Deallocate{}(1333) BEGIN
 // (op 1333) Deallocate 672
 // Deleting tag 672
 // (op 58) Deallocate{}(1333) END
 // (op 58) Barrier(1260) BEGIN
s_barrier  // (op 1260) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 58) Barrier(1260) END
 // (op 58) SetCoordinate(246) BEGIN
 // (op 246) generate({})
 // (op 246) end: generate({})
 // (op 246) generate({247})
 // (op 246) LoadLDSTile{Value: BF8x4}(247) BEGIN
 // (op 247) GEN: loadMacroTileWAVELDS OP 247 LDS 472 WaveTile 169
 // (op 247) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 247) Generate 1:U32 into nullptr
 // tag 484: v[26:33] alias 484 -> 23
 // (op 247) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[26:33]
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

 // (op 247) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[26:33]
 // 	info.rowOffsetReg = DataFlowTag666: VGPR Value: UInt32 x 1: v8
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
 // (op 247) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

ds_read_b128 v[26:29], v8 offset:4096 // (op 247) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2875 (current 2780)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 2048)
ds_read_b128 v[30:33], v8 offset:4128 // (op 247) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 90. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2966 (current 2871)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 246) LoadLDSTile{Value: BF8x4}(247) END
 // (op 246) Deallocate{}(1317) BEGIN
 // (op 1317) Deallocate 668
 // Deleting tag 668
 // (op 246) Deallocate{}(1317) END
 // (op 246) end: generate({247})
 // (op 58) SetCoordinate(246) END
 // (op 58) Barrier(1273) BEGIN
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1273) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8320, Size: 64)
 // (op 58) Barrier(1273) END
 // (op 58) SetCoordinate(250) BEGIN
 // (op 250) generate({})
 // (op 250) end: generate({})
 // (op 250) generate({251})
 // (op 250) LoadLDSTile{Value: E8M0}(251) BEGIN
 // (op 251) GEN: loadMacroTileWAVELDS OP 251 LDS 480 WaveTile 295
 // (op 251) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 251) Generate 1:U32 into nullptr
 // tag 488: v20 alias 488 -> 39
 // (op 251) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_u8 v20, v10 offset:8320 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 2969 (current 2874)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8320, Size: 64)
 // (op 250) LoadLDSTile{Value: E8M0}(251) END
 // (op 250) Deallocate{}(1329) BEGIN
 // (op 1329) Deallocate 684
 // Deleting tag 684
 // (op 250) Deallocate{}(1329) END
 // (op 250) end: generate({251})
 // (op 58) SetCoordinate(250) END
 // (op 58) Deallocate{}(1325) BEGIN
 // (op 1325) Deallocate 480
 // Deleting tag 480
 // (op 58) Deallocate{}(1325) END
 // (op 58) Barrier(1254) BEGIN
 // (op 58) Barrier(1327) BEGIN
s_barrier  // (op 1254) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 58) Barrier(1254) END
s_barrier  // (op 1327) 
 // (op 58) Barrier(1327) END
 // (op 58) SetCoordinate(244) BEGIN
 // (op 244) generate({})
 // (op 244) end: generate({})
 // (op 244) generate({245})
 // (op 244) LoadLDSTile{Value: BF8x4}(245) BEGIN
 // (op 245) GEN: loadMacroTileWAVELDS OP 245 LDS 468 WaveTile 103
 // (op 245) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 245) Generate 1:U32 into nullptr
 // tag 482: v[34:41] alias 482 -> 15
 // (op 245) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[34:41]
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
 // 	info.offset = Literal Value: UInt32 x 0: 6144
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

 // (op 245) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[34:41]
 // 	info.rowOffsetReg = DataFlowTag654: VGPR Value: UInt32 x 1: v7
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
 // 	info.offset = Literal Value: UInt32 x 0: 6144
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 245) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

ds_read_b128 v[34:37], v7 offset:6144 // (op 245) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2972 (current 2877)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
ds_read_b128 v[38:41], v7 offset:6176 // (op 245) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 89. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 3062 (current 2967)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 244) LoadLDSTile{Value: BF8x4}(245) END
 // (op 244) Deallocate{}(1315) BEGIN
 // (op 1315) Deallocate 656
 // Deleting tag 656
 // (op 244) Deallocate{}(1315) END
 // (op 244) end: generate({245})
 // (op 58) SetCoordinate(244) END
 // (op 58) Barrier(1267) BEGIN
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1267) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8384, Size: 64)
 // (op 58) Barrier(1267) END
 // (op 58) SetCoordinate(248) BEGIN
 // (op 248) generate({})
 // (op 248) end: generate({})
 // (op 248) generate({249})
 // (op 248) LoadLDSTile{Value: E8M0}(249) BEGIN
 // (op 249) GEN: loadMacroTileWAVELDS OP 249 LDS 476 WaveTile 235
 // (op 249) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 249) Generate 1:U32 into nullptr
 // tag 486: v24 alias 486 -> 31
 // (op 249) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8384
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
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8384
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_u8 v24, v5 offset:8384 // (op 249) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3065 (current 2970)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8384, Size: 64)
 // (op 248) LoadLDSTile{Value: E8M0}(249) END
 // (op 248) Deallocate{}(1323) BEGIN
 // (op 1323) Deallocate 676
 // Deleting tag 676
 // (op 248) Deallocate{}(1323) END
 // (op 248) end: generate({249})
 // (op 58) SetCoordinate(248) END
 // (op 58) Deallocate{}(1319) BEGIN
 // (op 1319) Deallocate 476
 // Deleting tag 476
 // (op 58) Deallocate{}(1319) END
 // (op 58) NOP(422) BEGIN
 // (op 422) generate({})
 // (op 422) end: generate({})
 // (op 58) NOP(422) END
 // (op 58) Barrier(1321) BEGIN
 // (op 58) Multiply(252) BEGIN
 // (op 252) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile4xBF8, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 58) Barrier(1264) BEGIN
s_barrier  // (op 1321) 
 // (op 58) Barrier(1321) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[34:41], v[26:33], a[0:15], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 252) 
 // (op 58) Multiply(252) END
s_barrier  // (op 1264) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 58) Barrier(1264) END
 // (op 58) Deallocate{}(1331) BEGIN
 // (op 1331) Deallocate 488
 // Deleting tag 488alias (-> 39)
 // (op 1331) Deallocate 486
 // Deleting tag 486alias (-> 31)
 // (op 1331) Deallocate 484
 // Deleting tag 484alias (-> 23)
 // (op 1331) Deallocate 482
 // Deleting tag 482alias (-> 15)
 // (op 58) Deallocate{}(1331) END
 // (op 58) SetCoordinate(309) BEGIN
 // (op 309) generate({})
 // (op 309) end: generate({})
 // (op 309) generate({482})
 // (op 309) LoadTileDirect2LDS Value: BF8x4(482) BEGIN
 // (op 482) GEN: LoadTileDirect2LDS
 // (op 482) GEN: loadMacroTileDirect2LDS OP 482 LDS 472 MacroTile 470
 // (op 482) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 482) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride525: VGPR Value: UInt64 x 1: (unallocated)
 // (op 482) reg expression
 // (op 482) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 482) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 482)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 482)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 482): s38, s39
s_lshl_b64 s[38:39], s[8:9], 4 // (op 482) 
// Allocated Stride525: 2 VGPRs (Value: UInt64) (op 482): v48, v49
v_mov_b64 v[48:49], s[38:39] // (op 482) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 482): s38, s39

 // (op 482) Generate 1:U64 into nullptr
 // (op 482) Generate 1024:U32 into nullptr
 // (op 482) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 4096
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride525: VGPR Value: UInt64 x 1: v[48:49]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 482): v52, v53
v_mov_b64 v[52:53], v[46:47] // (op 482) 
 // (op 482) Generate Add(4096:U32, {DataFlowTag658Flatten: s15:U32})U32 into nullptr
 // (op 482) reg expression
 // (op 482) Add(4096:U32, {DataFlowTag658Flatten: s15:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 482): s19
s_add_u32 s19, 4096, s15 // (op 482) 
 // (op 482)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride525: v[48:49]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 482) Generate s19:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s19 // (op 482) reg expression
 // (op 482) Generate 192:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 192 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 482): s38
s_mov_b32 s38, 192 // (op 482) call()
// 192 is not a supported value!
buffer_load_dwordx4 v52, s[44:47], s38 offen offset: 0 lds // (op 482) Load value direct to lds
// VMEM: Expected stall of 266. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3684 (current 3204)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 482) Generate Add(v[52:53]:U64, v48:R)U64 into VGPR Value: UInt64 x 1: v[52:53]
 // (op 482) reg expression
 // (op 482) reg expression
 // (op 482) Add(v[52:53]:U64, v48:R)U64
 // (op 482) Add: dest (VGPR Value: UInt64 x 1: v[52:53]) = 
 // (op 482)      lhs (VGPR Value: UInt64 x 1: v[52:53]) 
 // (op 482)      rhs (VGPR Value: Raw32 x 1: v48)
// Allocated : 2 SGPRs (Value: Bool64) (op 482): s56, s57
v_add_co_u32 v52, s[56:57], v52, v48 // (op 482) least significant half
v_addc_co_u32 v53, s[56:57], 0, v53, s[56:57] // (op 482) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 482): s56, s57

 // (op 482) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 482) reg expression
 // (op 482) Add(m0:U32, 1024:U32)U32
 // (op 482) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 482) 
 // (op 482) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v52, s[44:47], s38 offen offset: 0 lds // (op 482) Load value direct to lds
// VMEM: Expected complete at 3688 (current 3208)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 482): s38

 // Freeing Stride525: 2 VGPRs (Value: UInt64) (op 482): v48, v49

 // Freeing : 2 VGPRs (Value: UInt64) (op 482): v52, v53

 // Freeing : 1 SGPR (Value: UInt32) (op 482): s19

 // (op 309) LoadTileDirect2LDS Value: BF8x4(482) END
 // (op 309) Deallocate{}(1371) BEGIN
 // (op 1371) Deallocate 660
 // Deleting tag 660
 // (op 309) Deallocate{}(1371) END
 // (op 309) end: generate({482})
 // (op 58) SetCoordinate(309) END
 // (op 58) Deallocate{}(1339) BEGIN
 // (op 1339) Deallocate 524
 // Freeing Offset524: 2 VGPRs (Value: UInt64) (op 459): v46, v47

 // Deleting tag 524
 // (op 1339) Deallocate 526
 // Deleting tag 526
 // (op 58) Deallocate{}(1339) END
 // (op 58) SetCoordinate(319) BEGIN
 // (op 319) generate({})
 // (op 319) end: generate({})
 // (op 319) generate({258})
 // (op 319) LoadTiled Value: E8M0(258) BEGIN
 // (op 258) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 258) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 258) Generate 1:U64 into nullptr
 // (op 258) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
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
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
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
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = Offset562: VGPR Value: UInt32 x 1: v16
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
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

 // (op 58) Assign VGPR Add(DataFlowTag(522)U64, ShiftL(DataFlowTag(523)U64, 1:U32)U64)U64(613) BEGIN
 // (op 613) Assign dim(522) = Add(DataFlowTag(522)U64, ShiftL(DataFlowTag(523)U64, 1:U32)U64)U64
 // (op 613) Generate Add(DataFlowTag(522)U64, ShiftL(DataFlowTag(523)U64, 1:U32)U64)U64 into DataFlowTag522: VGPR Value: UInt64 x 1: v[14:15]
 // (op 613) reg expression
 // (op 613) Add({DataFlowTag522: v[14:15]:U64}, 128:U64)U64
 // (op 613) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 613)      lhs (DataFlowTag522: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 613)      rhs (Literal Value: UInt64 x 0: 128)
v_add_co_u32 v14, vcc, 128, v14 // (op 613) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
buffer_load_ubyte v12, v16, s[52:55], 0 offen offset:6 // (op 258) Load value
// VMEM: Expected complete at 3690 (current 3210)
 // (op 319) LoadTiled Value: E8M0(258) END
 // (op 319) end: generate({258})
 // (op 58) SetCoordinate(319) END
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 613) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 58) Assign VGPR Add(DataFlowTag(522)U64, ShiftL(DataFlowTag(523)U64, 1:U32)U64)U64(613) END
 // (op 58) Deallocate{}(1343) BEGIN
 // (op 1343) Deallocate 562
 // Freeing Offset562: 1 VGPR (Value: UInt32) (op 25): v16

 // Deleting tag 562
 // (op 1343) Deallocate 564
 // Deleting tag 564
 // (op 58) Deallocate{}(1343) END
 // (op 58) Assign VGPR Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32(741) BEGIN
 // (op 741) Assign dim(560) = Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32
 // (op 741) Generate Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32 into DataFlowTag560: VGPR Value: UInt32 x 1: v25
 // (op 741) reg expression
 // (op 741) Add({DataFlowTag560: v25:U32}, 4:U32)U32
 // (op 58) Barrier(1257) BEGIN
s_barrier  // (op 1257) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 58) Barrier(1257) END
v_add_u32 v25, 4, v25 // (op 741) 
 // (op 58) Assign VGPR Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32(741) END
 // (op 58) SetCoordinate(304) BEGIN
 // (op 304) generate({})
 // (op 304) end: generate({})
 // (op 304) generate({477})
 // (op 304) LoadTileDirect2LDS Value: BF8x4(477) BEGIN
 // (op 477) GEN: LoadTileDirect2LDS
 // (op 477) GEN: loadMacroTileDirect2LDS OP 477 LDS 468 MacroTile 466
 // (op 477) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 477) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride506: VGPR Value: UInt64 x 1: (unallocated)
 // (op 477) reg expression
 // (op 477) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 477) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 477)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 477)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 477): s38, s39
s_lshl_b64 s[38:39], s[2:3], 4 // (op 477) 
// Allocated Stride506: 2 VGPRs (Value: UInt64) (op 477): v46, v47
v_mov_b64 v[46:47], s[38:39] // (op 477) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 477): s38, s39

 // (op 477) Generate 1:U64 into nullptr
 // (op 477) Generate 1024:U32 into nullptr
 // (op 477) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 6144
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride506: VGPR Value: UInt64 x 1: v[46:47]
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
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 477): v48, v49
v_mov_b64 v[48:49], v[50:51] // (op 477) 
 // (op 477) Generate Add(6144:U32, {DataFlowTag646Flatten: s14:U32})U32 into nullptr
 // (op 477) reg expression
 // (op 477) Add(6144:U32, {DataFlowTag646Flatten: s14:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 477): s19
s_add_u32 s19, 6144, s14 // (op 477) 
 // (op 477)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride506: v[46:47]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 477) Generate s19:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s19 // (op 477) reg expression
 // (op 477) Generate 192:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 192 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 477): s38
s_mov_b32 s38, 192 // (op 477) call()
// 192 is not a supported value!
buffer_load_dwordx4 v48, s[40:43], s38 offen offset: 0 lds // (op 477) Load value direct to lds
// VMEM: Expected stall of 465. CBNW: 6, Inc: 3
// VMEM: Expected complete at 4165 (current 3685)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 477) Generate Add(v[48:49]:U64, v46:R)U64 into VGPR Value: UInt64 x 1: v[48:49]
 // (op 477) reg expression
 // (op 477) reg expression
 // (op 477) Add(v[48:49]:U64, v46:R)U64
 // (op 477) Add: dest (VGPR Value: UInt64 x 1: v[48:49]) = 
 // (op 477)      lhs (VGPR Value: UInt64 x 1: v[48:49]) 
 // (op 477)      rhs (VGPR Value: Raw32 x 1: v46)
// Allocated : 2 SGPRs (Value: Bool64) (op 477): s56, s57
v_add_co_u32 v48, s[56:57], v48, v46 // (op 477) least significant half
v_addc_co_u32 v49, s[56:57], 0, v49, s[56:57] // (op 477) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 477): s56, s57

 // (op 477) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 477) reg expression
 // (op 477) Add(m0:U32, 1024:U32)U32
 // (op 477) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 477) 
 // (op 477) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v48, s[40:43], s38 offen offset: 0 lds // (op 477) Load value direct to lds
// VMEM: Expected complete at 4169 (current 3689)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 477): s38

 // Freeing Stride506: 2 VGPRs (Value: UInt64) (op 477): v46, v47

 // Freeing : 2 VGPRs (Value: UInt64) (op 477): v48, v49

 // Freeing : 1 SGPR (Value: UInt32) (op 477): s19

 // (op 304) LoadTileDirect2LDS Value: BF8x4(477) END
 // (op 304) Deallocate{}(1369) BEGIN
 // (op 1369) Deallocate 648
 // Deleting tag 648
 // (op 304) Deallocate{}(1369) END
 // (op 304) end: generate({477})
 // (op 58) SetCoordinate(304) END
 // (op 58) Deallocate{}(1337) BEGIN
 // (op 1337) Deallocate 505
 // Freeing Offset505: 2 VGPRs (Value: UInt64) (op 454): v50, v51

 // Deleting tag 505
 // (op 1337) Deallocate 507
 // Deleting tag 507
 // (op 58) Deallocate{}(1337) END
 // (op 58) SetCoordinate(314) BEGIN
 // (op 314) generate({})
 // (op 314) end: generate({})
 // (op 314) generate({257})
 // (op 314) LoadTiled Value: E8M0(257) BEGIN
 // (op 257) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 257) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 257) Generate 1:U64 into nullptr
 // (op 257) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
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
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
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
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = Offset543: VGPR Value: UInt32 x 1: v17
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
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 257) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 58) Assign VGPR Add(DataFlowTag(503)U64, ShiftL(DataFlowTag(504)U64, 1:U32)U64)U64(549) BEGIN
 // (op 549) Assign dim(503) = Add(DataFlowTag(503)U64, ShiftL(DataFlowTag(504)U64, 1:U32)U64)U64
 // (op 549) Generate Add(DataFlowTag(503)U64, ShiftL(DataFlowTag(504)U64, 1:U32)U64)U64 into DataFlowTag503: VGPR Value: UInt64 x 1: v[18:19]
 // (op 549) reg expression
 // (op 549) Add({DataFlowTag503: v[18:19]:U64}, 128:U64)U64
 // (op 549) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 549)      lhs (DataFlowTag503: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 549)      rhs (Literal Value: UInt64 x 0: 128)
v_add_co_u32 v18, vcc, 128, v18 // (op 549) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
buffer_load_ubyte v13, v17, s[48:51], 0 offen offset:6 // (op 257) Load value
// VMEM: Expected complete at 4171 (current 3691)
 // (op 314) LoadTiled Value: E8M0(257) END
 // (op 314) end: generate({257})
 // (op 58) SetCoordinate(314) END
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 549) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 58) Assign VGPR Add(DataFlowTag(503)U64, ShiftL(DataFlowTag(504)U64, 1:U32)U64)U64(549) END
 // (op 58) Deallocate{}(1341) BEGIN
 // (op 1341) Deallocate 545
 // Deleting tag 545
 // (op 1341) Deallocate 543
 // Freeing Offset543: 1 VGPR (Value: UInt32) (op 17): v17

 // Deleting tag 543
 // (op 58) Deallocate{}(1341) END
 // (op 58) NOP(461) BEGIN
 // (op 461) generate({})
 // (op 461) end: generate({})
 // (op 58) NOP(461) END
 // (op 58) Assign VGPR Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32(677) BEGIN
 // (op 677) Assign dim(541) = Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32
 // (op 677) Generate Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32 into DataFlowTag541: VGPR Value: UInt32 x 1: v21
 // (op 677) reg expression
 // (op 677) Add({DataFlowTag541: v21:U32}, 4:U32)U32
 // (op 58) Barrier(1240) BEGIN
s_barrier  // (op 1240) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 58) Barrier(1240) END
v_add_u32 v21, 4, v21 // (op 677) 
 // (op 58) Assign VGPR Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32(677) END
 // (op 58) StoreLDSTile Value: E8M0(30) BEGIN
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = DataFlowTag620: VGPR Value: UInt32 x 1: v43
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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
ds_write_b8 v43, v22 offset:8192 // (op 30) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3928 (current 3833)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(30) END
 // (op 58) Deallocate{}(1281) BEGIN
 // (op 1281) Deallocate 622
 // Deleting tag 622
 // (op 58) Deallocate{}(1281) END
 // (op 58) NOP(456) BEGIN
 // (op 456) generate({})
 // (op 456) end: generate({})
 // (op 58) NOP(456) END
 // (op 58) Barrier(1223) BEGIN
s_barrier  // (op 1223) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 58) Barrier(1223) END
 // (op 58) StoreLDSTile Value: E8M0(21) BEGIN
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
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
 // 	info.data = VGPR Value: E8M0 x 1: v23
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v42
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
ds_write_b8 v42, v23 offset:8256 // (op 21) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3931 (current 3836)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(21) END
 // (op 58) Deallocate{}(1279) BEGIN
 // (op 1279) Deallocate 606
 // Deleting tag 606
 // (op 58) Deallocate{}(1279) END
 // (op 58) Barrier(1212) BEGIN
s_barrier  // (op 1212) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 58) Barrier(1212) END
 // (op 58) SetCoordinate(73) BEGIN
 // (op 73) generate({})
 // (op 73) end: generate({})
 // (op 73) generate({75})
 // (op 73) LoadLDSTile{Value: BF8x4}(75) BEGIN
 // (op 75) GEN: loadMacroTileWAVELDS OP 75 LDS 25 WaveTile 169
 // (op 75) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 75) Generate 1:U32 into nullptr
 // (op 75) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[26:33]
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

 // (op 75) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[26:33]
 // 	info.rowOffsetReg = DataFlowTag600: VGPR Value: UInt32 x 1: v4
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
 // (op 75) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

ds_read_b128 v[26:29], v4 // (op 75) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 3933 (current 3838)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
ds_read_b128 v[30:33], v4 offset:32 // (op 75) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 90. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4024 (current 3929)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 73) LoadLDSTile{Value: BF8x4}(75) END
 // (op 73) end: generate({75})
 // (op 58) SetCoordinate(73) END
 // (op 58) Barrier(1247) BEGIN
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1247) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 58) Barrier(1247) END
 // (op 58) SetCoordinate(83) BEGIN
 // (op 83) generate({})
 // (op 83) end: generate({})
 // (op 83) generate({85})
 // (op 83) LoadLDSTile{Value: E8M0}(85) BEGIN
 // (op 85) GEN: loadMacroTileWAVELDS OP 85 LDS 41 WaveTile 295
 // (op 85) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 85) Generate 1:U32 into nullptr
 // (op 85) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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

 // (op 85) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = DataFlowTag632: VGPR Value: UInt32 x 1: v6
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 85) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_u8 v20, v6 offset:8192 // (op 85) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4027 (current 3932)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 83) LoadLDSTile{Value: E8M0}(85) END
 // (op 83) end: generate({85})
 // (op 58) SetCoordinate(83) END
 // (op 58) Barrier(1196) BEGIN
s_barrier  // (op 1196) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 58) Barrier(1196) END
 // (op 58) SetCoordinate(68) BEGIN
 // (op 68) generate({})
 // (op 68) end: generate({})
 // (op 68) generate({70})
 // (op 68) LoadLDSTile{Value: BF8x4}(70) BEGIN
 // (op 70) GEN: loadMacroTileWAVELDS OP 70 LDS 17 WaveTile 103
 // (op 70) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 70) Generate 1:U32 into nullptr
 // (op 70) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[34:41]
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

 // (op 70) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[34:41]
 // 	info.rowOffsetReg = DataFlowTag584: VGPR Value: UInt32 x 1: v3
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
 // (op 70) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

ds_read_b128 v[34:37], v3 offset:2048 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4029 (current 3934)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
ds_read_b128 v[38:41], v3 offset:2080 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 90. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 4120 (current 4025)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 68) LoadLDSTile{Value: BF8x4}(70) END
 // (op 68) end: generate({70})
 // (op 58) SetCoordinate(68) END
 // (op 58) Barrier(1230) BEGIN
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1230) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 58) Barrier(1230) END
 // (op 58) SetCoordinate(78) BEGIN
 // (op 78) generate({})
 // (op 78) end: generate({})
 // (op 78) generate({80})
 // (op 78) LoadLDSTile{Value: E8M0}(80) BEGIN
 // (op 80) GEN: loadMacroTileWAVELDS OP 80 LDS 33 WaveTile 235
 // (op 80) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 80) Generate 1:U32 into nullptr
 // (op 80) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

 // (op 80) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v24
 // 	info.rowOffsetReg = DataFlowTag616: VGPR Value: UInt32 x 1: v0
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
 // 	info.bufDesc = (nullptr)
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
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_u8 v24, v0 offset:8256 // (op 80) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4123 (current 4028)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 78) LoadLDSTile{Value: E8M0}(80) END
 // (op 78) end: generate({80})
 // (op 58) SetCoordinate(78) END
 // (op 58) NOP(423) BEGIN
 // (op 423) generate({})
 // (op 423) end: generate({})
 // (op 58) NOP(423) END
 // (op 58) end: generate({60, 88, 1206})
 // (op 58) For Loop Increment
 // (op 58) generate({})
 // (op 58) end: generate({})
 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
 // (op 58) Generate LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 58) LessThan({DataFlowTag395: s18:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 58) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 58) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 58): s38, s39
s_ashr_i64 s[38:39], s[28:29], 6 // (op 58) 
// Allocated : 1 SGPR (Value: UInt32) (op 58): s19
s_mov_b32 s19, s38 // (op 58) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 58): s38, s39

// Allocated : 1 SGPR (Value: UInt32) (op 58): s38
s_and_b32 s38, s19, 4294967294 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s19

s_cmp_lt_u32 s18, s38 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s38

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 58) 
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58:
 // (op 58) 
 // (op 58) Unlock For Loop
 // Unlock None
 // (op 530) ForLoopOp KLoop: LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(58) END
 // (op 530) Deallocate{}(1285) BEGIN
 // (op 1285) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 375): v23

 // Deleting tag 10
 // (op 1285) Deallocate 671
 // Deleting tag 671
 // (op 1285) Deallocate 15
 // Deleting tag 15
 // (op 1285) Deallocate 31
 // Deleting tag 31
 // (op 1285) Deallocate 572
 // Freeing DataFlowTag572: 1 SGPR (Value: UInt32) (op 787): s0

 // Deleting tag 572
 // (op 1285) Deallocate 573
 // Deleting tag 573
 // (op 1285) Deallocate 588
 // Freeing DataFlowTag588: 1 SGPR (Value: UInt32) (op 839): s1

 // Deleting tag 588
 // (op 1285) Deallocate 604
 // Freeing DataFlowTag604: 1 VGPR (Value: UInt32) (op 891): v42

 // Deleting tag 604
 // (op 1285) Deallocate 607
 // Deleting tag 607
 // (op 1285) Deallocate 620
 // Freeing DataFlowTag620: 1 VGPR (Value: UInt32) (op 944): v43

 // Deleting tag 620
 // (op 1285) Deallocate 621
 // Deleting tag 621
 // (op 1285) Deallocate 646
 // Freeing DataFlowTag646: 1 SGPR (Value: UInt32) (op 1031): s14

 // Deleting tag 646
 // (op 1285) Deallocate 647
 // Deleting tag 647
 // (op 1285) Deallocate 649
 // Deleting tag 649
 // (op 1285) Deallocate 658
 // Freeing DataFlowTag658: 1 SGPR (Value: UInt32) (op 1070): s15

 // Deleting tag 658
 // (op 1285) Deallocate 678
 // Freeing DataFlowTag678: 1 VGPR (Value: UInt32) (op 1133): v45

 // Deleting tag 678
 // (op 1285) Deallocate 670
 // Freeing DataFlowTag670: 1 VGPR (Value: UInt32) (op 1109): v44

 // Deleting tag 670
 // (op 1285) Deallocate 673
 // Deleting tag 673
 // (op 1285) Deallocate 659
 // Deleting tag 659
 // (op 1285) Deallocate 661
 // Deleting tag 661
 // (op 1285) Deallocate 589
 // Deleting tag 589
 // (op 1285) Deallocate 575
 // Deleting tag 575
 // (op 1285) Deallocate 591
 // Deleting tag 591
 // (op 1285) Deallocate 623
 // Deleting tag 623
 // (op 1285) Deallocate 679
 // Deleting tag 679
 // (op 1285) Deallocate 681
 // Deleting tag 681
 // (op 1285) Deallocate 541
 // Freeing DataFlowTag541: 1 VGPR (Value: UInt32) (op 685): v21

 // Deleting tag 541
 // (op 1285) Deallocate 544
 // Deleting tag 544
 // (op 1285) Deallocate 546
 // Deleting tag 546
 // (op 1285) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 749): v25

 // Deleting tag 560
 // (op 1285) Deallocate 565
 // Deleting tag 565
 // (op 1285) Deallocate 470
 // Deleting tag 470
 // (op 1285) Deallocate 605
 // Deleting tag 605
 // (op 1285) Deallocate 472
 // Deleting tag 472
 // (op 1285) Deallocate 522
 // Freeing DataFlowTag522: 2 VGPRs (Value: UInt64) (op 621): v14, v15

 // Deleting tag 522
 // (op 1285) Deallocate 8
 // Deleting tag 8
 // (op 1285) Deallocate 478
 // Deleting tag 478
 // (op 1285) Deallocate 466
 // Deleting tag 466
 // (op 1285) Deallocate 468
 // Deleting tag 468
 // (op 1285) Deallocate 508
 // Deleting tag 508
 // (op 1285) Deallocate 9
 // Deleting tag 9
 // (op 1285) Deallocate 523
 // Deleting tag 523
 // (op 1285) Deallocate 23
 // Deleting tag 23
 // (op 1285) Deallocate 474
 // Deleting tag 474
 // (op 1285) Deallocate 542
 // Deleting tag 542
 // (op 1285) Deallocate 39
 // Deleting tag 39
 // (op 1285) Deallocate 561
 // Deleting tag 561
 // (op 1285) Deallocate 395
 // Freeing DataFlowTag395: 1 SGPR (Value: UInt32) (op 59): s18

 // Deleting tag 395
 // (op 1285) Deallocate 525
 // Deleting tag 525
 // (op 1285) Deallocate 11
 // Freeing : 1 VGPR (Value: E8M0) (op 369): v22

 // Deleting tag 11
 // (op 1285) Deallocate 506
 // Deleting tag 506
 // (op 1285) Deallocate 563
 // Deleting tag 563
 // (op 1285) Deallocate 527
 // Deleting tag 527
 // (op 1285) Deallocate 503
 // Freeing DataFlowTag503: 2 VGPRs (Value: UInt64) (op 557): v18, v19

 // Deleting tag 503
 // (op 1285) Deallocate 504
 // Deleting tag 504
 // (op 530) Deallocate{}(1285) END
 // (op 530) Barrier(1287) BEGIN
s_barrier  // (op 1287) 
 // (op 530) Barrier(1287) END
 // (op 530) end: generate({525})
 // (op 530) Unlock Scope 530
 // Unlock None
 // (op 340) Scope(530) END
 // (op 340) SetCoordinate(217) BEGIN
 // (op 217) generate({})
 // (op 217) end: generate({})
 // (op 217) generate({218})
 // (op 217) SetCoordinate(218) BEGIN
 // (op 218) generate({})
 // (op 218) end: generate({})
 // (op 218) generate({507})
 // (op 218) Scope(507) BEGIN
 // (op 507) Lock Scope 507
 // Lock Branch
 // (op 507) generate({501})
 // (op 507) NOP(501) BEGIN
 // (op 501) generate({})
 // (op 501) end: generate({})
 // (op 507) NOP(501) END
 // (op 507) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}(512) BEGIN
 // (op 512) Assign dim(490) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}
 // tag 490: v**UNALLOCATED**
 // (op 512) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64} into DataFlowTag490: VGPR Value: UInt64 x 1: (unallocated)
 // (op 512) reg expression
 // (op 512) reg expression
 // (op 512) reg expression
 // (op 512) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 512) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 512): v25
v_lshrrev_b32 v25, 2, v1 // (op 512) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 512): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 512) 
 // (op 512) BEGIN: Tile
 // (op 512) convert
// Allocated : 1 VGPR (Value: UInt32) (op 512): v14
v_lshl_add_u32 v14, s16, 5, v25 // (op 512) 
 // (op 512) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 512): v25

 // (op 512) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 512) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 512): s14
s_lshr_b32 s14, s0, 1 // (op 512) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 512): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 512): v25
v_and_b32 v25, -4, v1 // (op 512) 
 // (op 512) reg expression
 // (op 512) BEGIN: Flatten[1]
 // (op 512) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v25:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 512): v15
v_sub_u32 v15, v1, v25 // (op 512) 
 // (op 512) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 512): v25

 // (op 512) BEGIN: Tile
 // (op 512) {Tile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 512) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 512)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 512)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 512): v25
v_lshlrev_b32 v25, 4, v15 // (op 512) 
 // (op 512) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 512): v15

 // (op 512) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 512): v15
v_lshl_add_u32 v15, s14, 7, v25 // (op 512) 
 // (op 512) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 512): v25

 // Freeing : 1 SGPR (Value: UInt32) (op 512): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 512): v22, v23
v_mov_b32 v23, 0 // (op 512) convert
v_mov_b32 v22, v15 // (op 512) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 512): v15

 // (op 512) BEGIN: Split
 // (op 512) Multiply: dest (DataFlowTag490: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 512)           lhs (Tile: VGPR Value: UInt32 x 1: v14) 
 // (op 512)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag490: 2 VGPRs (Value: UInt64) (op 512): v18, v19
v_mul_lo_u32 v19, v14, s3 // (op 512) most significant: low of low * high
 // (op 512) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 512): v21
v_mul_hi_u32 v21, v14, s2 // (op 512) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 512) least significant: low of low * low
v_add_u32 v19, v19, v21 // (op 512) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 512): v21

 // (op 512) Add: dest (DataFlowTag490: VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 512)      lhs (DataFlowTag490: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 512)      rhs (Tile: VGPR Value: UInt64 x 1: v[22:23])
// Allocated : 2 SGPRs (Value: Bool64) (op 512): s0, s1
v_add_co_u32 v18, s[0:1], v18, v22 // (op 512) least significant half
v_addc_co_u32 v19, s[0:1], v19, v23, s[0:1] // (op 512) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 512): s0, s1

 // (op 512) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 512): v22, v23

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 512): v14

 // (op 507) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}(512) END
 // (op 507) Assign VGPR 64:U64(513) BEGIN
 // (op 513) Assign dim(491) = 64:U64
 // (op 513) Assign stride expression
 // (op 507) Assign VGPR 64:U64(513) END
 // (op 507) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(514) BEGIN
 // (op 514) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 514) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // (op 514) reg expression
 // (op 514) reg expression
 // (op 514) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 514) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 514)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 514)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 514): s0, s1
s_mul_i32 s1, s2, s7 // (op 514) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 514): s14
s_mul_i32 s14, s3, s6 // (op 514) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 514) least significant
// Allocated : 1 SGPR (Value: Int32) (op 514): s15
s_mul_hi_u32 s15, s2, s6 // (op 514) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 514) most significant: sum
s_add_u32 s1, s1, s15 // (op 514) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 514): s15

 // Freeing : 1 SGPR (Value: Int32) (op 514): s14

 // (op 514) reg expression
 // (op 514) reg expression
s_mov_b32 s40, s4 // (op 514) reg expression
s_mov_b32 s41, s5 // (op 514) reg expression
s_mov_b32 s42, s0 // (op 514) reg expression
s_mov_b32 s43, 131072 // (op 514) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 514): s0, s1

 // (op 507) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(514) END
 // (op 507) NOP(503) BEGIN
 // (op 503) generate({})
 // (op 503) end: generate({})
 // (op 507) NOP(503) END
 // (op 507) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(519) BEGIN
 // (op 519) Assign dim(495) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 519) Assign stride expression
 // (op 507) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(519) END
 // (op 507) NOP(504) BEGIN
 // (op 504) generate({})
 // (op 504) end: generate({})
 // (op 507) NOP(504) END
 // (op 507) Assign VGPR 1:U64(522) BEGIN
 // (op 522) Assign dim(497) = 1:U64
 // (op 522) Assign stride expression
 // (op 507) Assign VGPR 1:U64(522) END
 // (op 507) NOP(570) BEGIN
 // (op 570) generate({})
 // (op 570) end: generate({})
 // (op 507) NOP(570) END
 // (op 507) Assign VGPR {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(579) BEGIN
 // (op 579) Assign dim(509) = {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 509: v**UNALLOCATED**
 // (op 579) Generate {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag509: VGPR Value: UInt64 x 1: (unallocated)
 // (op 579) reg expression
 // (op 579) reg expression
 // (op 579) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 579) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 579): v25
v_and_b32 v25, -4, v1 // (op 579) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 579): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 579) 
 // (op 579) BEGIN: Flatten[1]
 // (op 579) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v25:U32)U32}
 // (op 579) convert
// Allocated : 1 VGPR (Value: UInt32) (op 579): v14
v_sub_u32 v14, v1, v25 // (op 579) 
 // (op 579) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 579): v25

 // (op 579) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 579) BEGIN: Tile
 // (op 579) {Tile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 579) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 579)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 579)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 SGPR (Value: UInt32) (op 579): s14
s_lshr_b32 s14, s0, 1 // (op 579) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 579): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 579): v25
v_lshlrev_b32 v25, 4, v14 // (op 579) 
 // (op 579) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 579): v14

 // (op 579) reg expression
 // (op 579) BEGIN: Tile
 // (op 579) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 579): v14
v_lshl_add_u32 v14, s14, 7, v25 // (op 579) 
 // (op 579) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 579): v25

 // Freeing : 1 SGPR (Value: UInt32) (op 579): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 579): v16, v17
v_mov_b32 v17, 0 // (op 579) convert
v_mov_b32 v16, v14 // (op 579) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 579): v14

 // (op 579) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 579): v25
v_lshrrev_b32 v25, 2, v1 // (op 579) 
 // (op 579) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 579): v14
v_lshl_add_u32 v14, s17, 5, v25 // (op 579) 
 // (op 579) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 579): v25

 // (op 579) Multiply({Tile: v14:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 579) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 579)           lhs (Tile: VGPR Value: UInt32 x 1: v14) 
 // (op 579)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 579): v22, v23
v_mul_lo_u32 v23, v14, s9 // (op 579) most significant: low of low * high
 // (op 579) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 579): v15
v_mul_hi_u32 v15, v14, s8 // (op 579) most significant: high of low * low
v_mul_lo_u32 v22, v14, s8 // (op 579) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 579) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 579): v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 579): v14

 // (op 579) BEGIN: Split
 // (op 579) {Split: Add({Tile: v[16:17]:U64}, v[22:23]:I64)U64}
 // (op 579) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 579)      lhs (Tile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 579)      rhs (VGPR Value: Int64 x 1: v[22:23])
// Allocated DataFlowTag509: 2 VGPRs (Value: UInt64) (op 579): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 579): s0, s1
v_add_co_u32 v14, s[0:1], v16, v22 // (op 579) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 579) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 579): s0, s1

 // (op 579) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 579): v22, v23

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 579): v16, v17

 // (op 507) Assign VGPR {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(579) END
 // (op 507) Assign VGPR 64:U64(580) BEGIN
 // (op 580) Assign dim(510) = 64:U64
 // (op 580) Assign stride expression
 // (op 507) Assign VGPR 64:U64(580) END
 // (op 507) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(581) BEGIN
 // (op 581) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 581) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // (op 581) reg expression
 // (op 581) reg expression
 // (op 581) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 581) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 581)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 581)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 581): s0, s1
s_mul_i32 s1, s8, s13 // (op 581) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 581): s14
s_mul_i32 s14, s9, s12 // (op 581) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 581) least significant
// Allocated : 1 SGPR (Value: Int32) (op 581): s15
s_mul_hi_u32 s15, s8, s12 // (op 581) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 581) most significant: sum
s_add_u32 s1, s1, s15 // (op 581) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 581): s15

 // Freeing : 1 SGPR (Value: Int32) (op 581): s14

 // (op 581) reg expression
 // (op 581) reg expression
s_mov_b32 s44, s10 // (op 581) reg expression
s_mov_b32 s45, s11 // (op 581) reg expression
s_mov_b32 s46, s0 // (op 581) reg expression
s_mov_b32 s47, 131072 // (op 581) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 581): s0, s1

 // (op 507) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(581) END
 // (op 507) NOP(572) BEGIN
 // (op 572) generate({})
 // (op 572) end: generate({})
 // (op 507) NOP(572) END
 // (op 507) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(586) BEGIN
 // (op 586) Assign dim(514) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 586) Assign stride expression
 // (op 507) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(586) END
 // (op 507) NOP(573) BEGIN
 // (op 573) generate({})
 // (op 573) end: generate({})
 // (op 507) NOP(573) END
 // (op 507) Assign VGPR 1:U64(589) BEGIN
 // (op 589) Assign dim(516) = 1:U64
 // (op 589) Assign stride expression
 // (op 507) Assign VGPR 1:U64(589) END
 // (op 507) NOP(634) BEGIN
 // (op 634) generate({})
 // (op 634) end: generate({})
 // (op 507) NOP(634) END
 // (op 507) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}(643) BEGIN
 // (op 643) Assign dim(528) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}
 // tag 528: v**UNALLOCATED**
 // (op 643) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32} into DataFlowTag528: VGPR Value: UInt32 x 1: (unallocated)
 // (op 643) reg expression
 // (op 643) reg expression
 // (op 643) reg expression
 // (op 643) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 643) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 643): v25
v_lshrrev_b32 v25, 1, v1 // (op 643) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 643): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 643) 
 // (op 643) BEGIN: Tile
 // (op 643) convert
// Allocated : 1 VGPR (Value: UInt32) (op 643): v16
v_lshl_add_u32 v16, s16, 5, v25 // (op 643) 
 // (op 643) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 643): v25

 // (op 643) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 643) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 643): s14
s_lshr_b32 s14, s0, 1 // (op 643) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 643): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 643): v17
v_and_b32 v17, -2, v1 // (op 643) 
 // (op 643) reg expression
 // (op 643) BEGIN: Flatten[1]Tile
 // (op 643) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v17:U32)U32}
 // (op 643) convert
// Allocated : 1 VGPR (Value: UInt32) (op 643): v25
v_sub_u32 v25, v1, v17 // (op 643) 
 // (op 643) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 643): v17

 // (op 643) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 643): v17
v_lshl_add_u32 v17, s14, 2, v25 // (op 643) 
 // (op 643) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 643): v25

 // Freeing : 1 SGPR (Value: UInt32) (op 643): s14

 // (op 643) BEGIN: Split
// Allocated DataFlowTag528: 1 VGPR (Value: UInt32) (op 643): v25
v_mul_lo_u32 v25, v16, s20 // (op 643) 
v_add_u32 v25, v25, v17 // (op 643) 
 // (op 643) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 643): v17

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 643): v16

 // (op 507) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}(643) END
 // (op 507) Assign VGPR 2:U64(644) BEGIN
 // (op 644) Assign dim(529) = 2:U64
 // (op 644) Assign stride expression
 // (op 507) Assign VGPR 2:U64(644) END
 // (op 507) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(645) BEGIN
 // (op 645) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 645) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // (op 645) reg expression
 // (op 645) reg expression
 // (op 645) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 645) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 645)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 645)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 645): s0, s1
s_mul_i32 s1, s20, s7 // (op 645) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 645): s14
s_mul_i32 s14, s21, s6 // (op 645) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 645) least significant
// Allocated : 1 SGPR (Value: Int32) (op 645): s15
s_mul_hi_u32 s15, s20, s6 // (op 645) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 645) most significant: sum
s_add_u32 s1, s1, s15 // (op 645) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 645): s15

 // Freeing : 1 SGPR (Value: Int32) (op 645): s14

 // (op 645) reg expression
 // (op 645) reg expression
s_mov_b32 s48, s22 // (op 645) reg expression
s_mov_b32 s49, s23 // (op 645) reg expression
s_mov_b32 s50, s0 // (op 645) reg expression
s_mov_b32 s51, 131072 // (op 645) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 645): s0, s1

 // (op 507) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(645) END
 // (op 507) NOP(636) BEGIN
 // (op 636) generate({})
 // (op 636) end: generate({})
 // (op 507) NOP(636) END
 // (op 507) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(650) BEGIN
 // (op 650) Assign dim(533) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 650) Assign stride expression
 // (op 507) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(650) END
 // (op 507) NOP(637) BEGIN
 // (op 637) generate({})
 // (op 637) end: generate({})
 // (op 507) NOP(637) END
 // (op 507) Assign VGPR 1:U64(653) BEGIN
 // (op 653) Assign dim(535) = 1:U64
 // (op 653) Assign stride expression
 // (op 507) Assign VGPR 1:U64(653) END
 // (op 507) NOP(698) BEGIN
 // (op 698) generate({})
 // (op 698) end: generate({})
 // (op 507) NOP(698) END
 // (op 507) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(707) BEGIN
 // (op 707) Assign dim(547) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 547: v**UNALLOCATED**
 // (op 707) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag547: VGPR Value: UInt32 x 1: (unallocated)
 // (op 707) reg expression
 // (op 707) reg expression
 // (op 707) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 707) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 707): v16
v_and_b32 v16, -2, v1 // (op 707) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 707): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 707) 
 // (op 707) BEGIN: Flatten[1]Tile
 // (op 707) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
 // (op 707) convert
// Allocated : 1 VGPR (Value: UInt32) (op 707): v17
v_sub_u32 v17, v1, v16 // (op 707) 
 // (op 707) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 707): v16

 // (op 707) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 707) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 707): s14
s_lshr_b32 s14, s0, 1 // (op 707) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 707): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 707): v16
v_lshrrev_b32 v16, 1, v1 // (op 707) 
 // (op 707) reg expression
 // (op 707) BEGIN: Tile
 // (op 707) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 707): v21
v_lshl_add_u32 v21, s14, 2, v17 // (op 707) 
 // (op 707) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 707): v17

 // Freeing : 1 SGPR (Value: UInt32) (op 707): s14

 // (op 707) convert
 // (op 707) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 707): v17
v_lshl_add_u32 v17, s17, 5, v16 // (op 707) 
 // (op 707) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 707): v16

 // (op 707) Multiply({Tile: v17:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 707): v16
v_mul_lo_u32 v16, v17, s24 // (op 707) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 707): v17

 // (op 707) BEGIN: Split
 // (op 707) {Split: Add({Tile: v21:U32}, v16:U32)U32}
// Allocated DataFlowTag547: 1 VGPR (Value: UInt32) (op 707): v17
v_add_u32 v17, v21, v16 // (op 707) 
 // (op 707) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 707): v16

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 707): v21

 // (op 507) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(707) END
 // (op 507) Assign VGPR 2:U64(708) BEGIN
 // (op 708) Assign dim(548) = 2:U64
 // (op 708) Assign stride expression
 // (op 507) Assign VGPR 2:U64(708) END
 // (op 507) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(709) BEGIN
 // (op 709) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 709) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // (op 709) reg expression
 // (op 709) reg expression
 // (op 709) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 709) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 709)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 709)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 709): s0, s1
s_mul_i32 s1, s24, s13 // (op 709) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 709): s14
s_mul_i32 s14, s25, s12 // (op 709) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 709) least significant
// Allocated : 1 SGPR (Value: Int32) (op 709): s15
s_mul_hi_u32 s15, s24, s12 // (op 709) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 709) most significant: sum
s_add_u32 s1, s1, s15 // (op 709) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 709): s15

 // Freeing : 1 SGPR (Value: Int32) (op 709): s14

 // (op 709) reg expression
 // (op 709) reg expression
s_mov_b32 s52, s26 // (op 709) reg expression
s_mov_b32 s53, s27 // (op 709) reg expression
s_mov_b32 s54, s0 // (op 709) reg expression
s_mov_b32 s55, 131072 // (op 709) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 709): s0, s1

 // (op 507) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(709) END
 // (op 507) NOP(700) BEGIN
 // (op 700) generate({})
 // (op 700) end: generate({})
 // (op 507) NOP(700) END
 // (op 507) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(714) BEGIN
 // (op 714) Assign dim(552) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 714) Assign stride expression
 // (op 507) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(714) END
 // (op 507) NOP(701) BEGIN
 // (op 701) generate({})
 // (op 701) end: generate({})
 // (op 507) NOP(701) END
 // (op 507) Assign VGPR 1:U64(717) BEGIN
 // (op 717) Assign dim(554) = 1:U64
 // (op 717) Assign stride expression
 // (op 507) Assign VGPR 1:U64(717) END
 // (op 507) NOP(795) BEGIN
 // (op 795) generate({})
 // (op 795) end: generate({})
 // (op 507) NOP(795) END
 // (op 507) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(800) BEGIN
 // (op 800) Assign dim(576) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 576: s**UNALLOCATED**
 // (op 800) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag576: SGPR Value: UInt32 x 1: (unallocated)
 // (op 800) reg expression
 // (op 800) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 800): v16
v_and_b32 v16, -4, v1 // (op 800) 
 // (op 800) BEGIN: Tile[1]
 // (op 800) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 800): v21
v_sub_u32 v21, v1, v16 // (op 800) 
 // (op 800) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 800): v16

 // (op 800) BEGIN: Flatten
 // (op 800) {Flatten: ShiftL({Tile[1]: v21:U32}, 4:U32)U32}
 // (op 800) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 800)         value (Tile[1]: VGPR Value: UInt32 x 1: v21) 
 // (op 800)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 800): v16
v_lshlrev_b32 v16, 4, v21 // (op 800) 
 // (op 800) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 800): v21

 // (op 800) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 800): v21
v_lshrrev_b32 v21, 2, v1 // (op 800) 
 // (op 800) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 800): v22
v_lshl_add_u32 v22, v21, 6, v16 // (op 800) 
 // (op 800) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 800): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 800): v21

// Allocated DataFlowTag576: 1 SGPR (Value: UInt32) (op 800): s0
s_nop 0
v_readfirstlane_b32 s0, v22 // (op 800) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 800): v22

 // (op 507) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(800) END
 // (op 507) Assign VGPR 1024:U32(801) BEGIN
 // (op 801) Assign dim(577) = 1024:U32
 // (op 801) Assign stride expression
 // (op 507) Assign VGPR 1024:U32(801) END
 // (op 507) NOP(796) BEGIN
 // (op 796) generate({})
 // (op 796) end: generate({})
 // (op 507) NOP(796) END
 // (op 507) Assign VGPR 1:U32(805) BEGIN
 // (op 805) Assign dim(579) = 1:U32
 // (op 805) Assign stride expression
 // (op 507) Assign VGPR 1:U32(805) END
 // (op 507) NOP(847) BEGIN
 // (op 847) generate({})
 // (op 847) end: generate({})
 // (op 507) NOP(847) END
 // (op 507) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(852) BEGIN
 // (op 852) Assign dim(592) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 592: s**UNALLOCATED**
 // (op 852) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag592: SGPR Value: UInt32 x 1: (unallocated)
 // (op 852) reg expression
 // (op 852) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 852): v16
v_and_b32 v16, -4, v1 // (op 852) 
 // (op 852) BEGIN: Tile[1]
 // (op 852) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 852): v21
v_sub_u32 v21, v1, v16 // (op 852) 
 // (op 852) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 852): v16

 // (op 852) BEGIN: Flatten
 // (op 852) {Flatten: ShiftL({Tile[1]: v21:U32}, 4:U32)U32}
 // (op 852) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 852)         value (Tile[1]: VGPR Value: UInt32 x 1: v21) 
 // (op 852)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 852): v16
v_lshlrev_b32 v16, 4, v21 // (op 852) 
 // (op 852) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 852): v21

 // (op 852) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 852): v21
v_lshrrev_b32 v21, 2, v1 // (op 852) 
 // (op 852) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 852): v22
v_lshl_add_u32 v22, v21, 6, v16 // (op 852) 
 // (op 852) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 852): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 852): v21

// Allocated DataFlowTag592: 1 SGPR (Value: UInt32) (op 852): s1
s_nop 0
v_readfirstlane_b32 s1, v22 // (op 852) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 852): v22

 // (op 507) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(852) END
 // (op 507) Assign VGPR 1024:U32(853) BEGIN
 // (op 853) Assign dim(593) = 1024:U32
 // (op 853) Assign stride expression
 // (op 507) Assign VGPR 1024:U32(853) END
 // (op 507) NOP(848) BEGIN
 // (op 848) generate({})
 // (op 848) end: generate({})
 // (op 507) NOP(848) END
 // (op 507) Assign VGPR 1:U32(857) BEGIN
 // (op 857) Assign dim(595) = 1:U32
 // (op 857) Assign stride expression
 // (op 507) Assign VGPR 1:U32(857) END
 // (op 507) NOP(899) BEGIN
 // (op 899) generate({})
 // (op 899) end: generate({})
 // (op 507) NOP(899) END
 // (op 507) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(904) BEGIN
 // (op 904) Assign dim(608) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 608: v**UNALLOCATED**
 // (op 904) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag608: VGPR Value: UInt32 x 1: (unallocated)
 // (op 904) reg expression
 // (op 904) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 904): v16
v_and_b32 v16, -2, v1 // (op 904) 
 // (op 904) BEGIN: Tile[1]Flatten
 // (op 904) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 904): v21
v_sub_u32 v21, v1, v16 // (op 904) 
 // (op 904) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 904): v16

 // (op 904) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 904): v16
v_and_b32 v16, 4294967294, v1 // (op 904) 
 // (op 904) BEGIN: Flatten
 // (op 904) {Flatten: Add(v16:U32, {Tile[1]Flatten: v21:U32})U32}
// Allocated DataFlowTag608: 1 VGPR (Value: UInt32) (op 904): v22
v_add_u32 v22, v16, v21 // (op 904) 
 // (op 904) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 904): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 904): v16

 // (op 507) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(904) END
 // (op 507) Assign VGPR 2:U32(905) BEGIN
 // (op 905) Assign dim(609) = 2:U32
 // (op 905) Assign stride expression
 // (op 507) Assign VGPR 2:U32(905) END
 // (op 507) NOP(900) BEGIN
 // (op 900) generate({})
 // (op 900) end: generate({})
 // (op 507) NOP(900) END
 // (op 507) Assign VGPR 1:U32(909) BEGIN
 // (op 909) Assign dim(611) = 1:U32
 // (op 909) Assign stride expression
 // (op 507) Assign VGPR 1:U32(909) END
 // (op 507) NOP(952) BEGIN
 // (op 952) generate({})
 // (op 952) end: generate({})
 // (op 507) NOP(952) END
 // (op 507) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(957) BEGIN
 // (op 957) Assign dim(624) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 624: v**UNALLOCATED**
 // (op 957) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag624: VGPR Value: UInt32 x 1: (unallocated)
 // (op 957) reg expression
 // (op 957) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 957): v16
v_and_b32 v16, -2, v1 // (op 957) 
 // (op 957) BEGIN: Tile[1]Flatten
 // (op 957) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 957): v21
v_sub_u32 v21, v1, v16 // (op 957) 
 // (op 957) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 957): v16

 // (op 957) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 957): v16
v_and_b32 v16, 4294967294, v1 // (op 957) 
 // (op 957) BEGIN: Flatten
 // (op 957) {Flatten: Add(v16:U32, {Tile[1]Flatten: v21:U32})U32}
// Allocated DataFlowTag624: 1 VGPR (Value: UInt32) (op 957): v23
v_add_u32 v23, v16, v21 // (op 957) 
 // (op 957) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 957): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 957): v16

 // (op 507) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(957) END
 // (op 507) Assign VGPR 2:U32(958) BEGIN
 // (op 958) Assign dim(625) = 2:U32
 // (op 958) Assign stride expression
 // (op 507) Assign VGPR 2:U32(958) END
 // (op 507) NOP(953) BEGIN
 // (op 953) generate({})
 // (op 953) end: generate({})
 // (op 507) NOP(953) END
 // (op 507) Assign VGPR 1:U32(962) BEGIN
 // (op 962) Assign dim(627) = 1:U32
 // (op 962) Assign stride expression
 // (op 507) Assign VGPR 1:U32(962) END
 // (op 507) ForLoopOp KLoopTail: LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(171) BEGIN
 // (op 171) Initialize For Loop
 // (op 171) generate({172})
 // (op 171) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(172) BEGIN
 // (op 172) Assign dim(448) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32
 // tag 448: s**UNALLOCATED**
 // (op 172) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag448: SGPR Value: UInt32 x 1: (unallocated)
 // (op 172) reg expression
 // (op 172) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 172): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 172) 
 // (op 172) convert
 // (op 172) BitwiseAnd({ convertForwardSubset: s14:U32}, 4294967294:U32)U32
// Allocated DataFlowTag448: 1 SGPR (Value: UInt32) (op 172): s18
s_and_b32 s18, s14, 4294967294 // (op 172) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 172): s14, s15

 // (op 171) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(172) END
 // (op 171) end: generate({172})
 // (op 171) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 171) 
 // (op 171) Generate LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 171) LessThan({DataFlowTag448: s18:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 171) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 171): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 171) 
// Allocated : 1 SGPR (Value: UInt32) (op 171): s19
s_mov_b32 s19, s14 // (op 171) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 171): s14, s15

s_cmp_lt_u32 s18, s19 // (op 171) 
 // Freeing : 1 SGPR (Value: UInt32) (op 171): s19

 // Freeing : 1 VGPR (Value: E8M0) (op 381): v12

 // Freeing : 1 VGPR (Value: E8M0) (op 387): v13

 // Freeing : 8 VGPRs (Value: BF8x4) (op 409): v26, v27, v28, v29, v30, v31, v32, v33

 // Freeing : 1 VGPR (Value: E8M0) (op 415): v20

 // Freeing : 8 VGPRs (Value: BF8x4) (op 406): v34, v35, v36, v37, v38, v39, v40, v41

 // Freeing : 1 VGPR (Value: E8M0) (op 412): v24

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 171) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 171) 
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 171) 
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171:
 // (op 171) 
 // (op 171) generate({173, 1193})
 // (op 171) Assign SGPR Add(DataFlowTag(448)U32, 1:U32)U32(173) BEGIN
 // (op 173) Assign dim(448) = Add(DataFlowTag(448)U32, 1:U32)U32
 // (op 173) Generate Add(DataFlowTag(448)U32, 1:U32)U32 into DataFlowTag448: SGPR Value: UInt32 x 1: s18
 // (op 173) reg expression
 // (op 173) Add({DataFlowTag448: s18:U32}, 1:U32)U32
 // (op 171) Barrier(1193) BEGIN
s_barrier  // (op 1193) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 171) Barrier(1193) END
s_add_u32 s18, s18, 1 // (op 173) 
 // (op 171) Assign SGPR Add(DataFlowTag(448)U32, 1:U32)U32(173) END
 // (op 171) SetCoordinate(222) BEGIN
 // (op 222) generate({})
 // (op 222) end: generate({})
 // (op 222) generate({464})
 // (op 222) LoadTileDirect2LDS Value: BF8x4(464) BEGIN
 // (op 464) GEN: LoadTileDirect2LDS
 // (op 464) GEN: loadMacroTileDirect2LDS OP 464 LDS 17 MacroTile 450
 // (op 464) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 464) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride495: VGPR Value: UInt64 x 1: (unallocated)
 // (op 464) reg expression
 // (op 464) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 464) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 464)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 464)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 464): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 464) 
// Allocated Stride495: 2 VGPRs (Value: UInt64) (op 464): v12, v13
v_mov_b64 v[12:13], s[14:15] // (op 464) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 464): s14, s15

 // (op 464) Generate 1:U64 into nullptr
 // (op 464) Generate 1024:U32 into nullptr
 // (op 464) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride495: VGPR Value: UInt64 x 1: v[12:13]
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
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 494: v**UNALLOCATED**
// Allocated Offset494: 2 VGPRs (Value: UInt64) (op 464): v20, v21
v_mov_b64 v[20:21], v[18:19] // (op 464) 
// Allocated : 2 VGPRs (Value: UInt64) (op 464): v26, v27
v_mov_b64 v[26:27], v[20:21] // (op 464) 
 // (op 464) Generate Add(2048:U32, {DataFlowTag576Flatten: s0:U32})U32 into nullptr
 // (op 464) reg expression
 // (op 464) Add(2048:U32, {DataFlowTag576Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 464): s19
s_add_u32 s19, 2048, s0 // (op 464) 
 // (op 464)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride495: v[12:13]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 464) Generate s19:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s19 // (op 464) reg expression
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds
// VMEM: Expected complete at 4664 (current 4184)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 464) Generate Add(v[26:27]:U64, v12:R)U64 into VGPR Value: UInt64 x 1: v[26:27]
 // (op 464) reg expression
 // (op 464) reg expression
 // (op 464) Add(v[26:27]:U64, v12:R)U64
 // (op 464) Add: dest (VGPR Value: UInt64 x 1: v[26:27]) = 
 // (op 464)      lhs (VGPR Value: UInt64 x 1: v[26:27]) 
 // (op 464)      rhs (VGPR Value: Raw32 x 1: v12)
// Allocated : 2 SGPRs (Value: Bool64) (op 464): s14, s15
v_add_co_u32 v26, s[14:15], v26, v12 // (op 464) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 464) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 464): s14, s15

 // (op 464) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 464) reg expression
 // (op 464) Add(m0:U32, 1024:U32)U32
 // (op 464) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 464) 
 // (op 464) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 464) Load value direct to lds
// VMEM: Expected complete at 4668 (current 4188)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing Stride495: 2 VGPRs (Value: UInt64) (op 464): v12, v13

 // Freeing : 2 VGPRs (Value: UInt64) (op 464): v26, v27

 // Freeing : 1 SGPR (Value: UInt32) (op 464): s19

 // (op 222) LoadTileDirect2LDS Value: BF8x4(464) END
 // (op 222) Deallocate{}(1365) BEGIN
 // (op 1365) Deallocate 578
 // Deleting tag 578
 // (op 1365) Deallocate 494
 // Freeing Offset494: 2 VGPRs (Value: UInt64) (op 464): v20, v21

 // Deleting tag 494
 // (op 1365) Deallocate 496
 // Deleting tag 496
 // (op 1365) Deallocate 450
 // Deleting tag 450
 // (op 222) Deallocate{}(1365) END
 // (op 222) Assign VGPR Add(DataFlowTag(490)U64, DataFlowTag(491)U64)U64(502) BEGIN
 // (op 502) Assign dim(490) = Add(DataFlowTag(490)U64, DataFlowTag(491)U64)U64
 // (op 502) Generate Add(DataFlowTag(490)U64, DataFlowTag(491)U64)U64 into DataFlowTag490: VGPR Value: UInt64 x 1: v[18:19]
 // (op 502) reg expression
 // (op 502) Add({DataFlowTag490: v[18:19]:U64}, 64:U64)U64
 // (op 502) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 502)      lhs (DataFlowTag490: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 502)      rhs (Literal Value: UInt64 x 0: 64)
// Allocated : 2 SGPRs (Value: Bool64) (op 502): s14, s15
v_add_co_u32 v18, s[14:15], 64, v18 // (op 502) least significant half
v_addc_co_u32 v19, s[14:15], 0, v19, s[14:15] // (op 502) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 502): s14, s15

 // (op 222) Assign VGPR Add(DataFlowTag(490)U64, DataFlowTag(491)U64)U64(502) END
 // (op 222) end: generate({464})
 // (op 171) SetCoordinate(222) END
 // (op 171) NOP(466) BEGIN
 // (op 466) generate({})
 // (op 466) end: generate({})
 // (op 171) NOP(466) END
 // (op 171) Barrier(1209) BEGIN
s_barrier  // (op 1209) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 171) Barrier(1209) END
 // (op 171) SetCoordinate(227) BEGIN
 // (op 227) generate({})
 // (op 227) end: generate({})
 // (op 227) generate({470})
 // (op 227) LoadTileDirect2LDS Value: BF8x4(470) BEGIN
 // (op 470) GEN: LoadTileDirect2LDS
 // (op 470) GEN: loadMacroTileDirect2LDS OP 470 LDS 25 MacroTile 452
 // (op 470) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 470) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride514: VGPR Value: UInt64 x 1: (unallocated)
 // (op 470) reg expression
 // (op 470) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 470) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 470)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 470)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 470): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 470) 
// Allocated Stride514: 2 VGPRs (Value: UInt64) (op 470): v12, v13
v_mov_b64 v[12:13], s[14:15] // (op 470) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 470): s14, s15

 // (op 470) Generate 1:U64 into nullptr
 // (op 470) Generate 1024:U32 into nullptr
 // (op 470) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride514: VGPR Value: UInt64 x 1: v[12:13]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 513: v**UNALLOCATED**
// Allocated Offset513: 2 VGPRs (Value: UInt64) (op 470): v20, v21
v_mov_b64 v[20:21], v[14:15] // (op 470) 
// Allocated : 2 VGPRs (Value: UInt64) (op 470): v26, v27
v_mov_b64 v[26:27], v[20:21] // (op 470) 
 // (op 470) Generate Add(0:U32, {DataFlowTag592Flatten: s1:U32})U32 into nullptr
 // (op 470)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride514: v[12:13]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 470) Generate {DataFlowTag592Flatten: s1:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 470) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 470) Load value direct to lds
// VMEM: Expected complete at 4677 (current 4197)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 470) Generate Add(v[26:27]:U64, v12:R)U64 into VGPR Value: UInt64 x 1: v[26:27]
 // (op 470) reg expression
 // (op 470) reg expression
 // (op 470) Add(v[26:27]:U64, v12:R)U64
 // (op 470) Add: dest (VGPR Value: UInt64 x 1: v[26:27]) = 
 // (op 470)      lhs (VGPR Value: UInt64 x 1: v[26:27]) 
 // (op 470)      rhs (VGPR Value: Raw32 x 1: v12)
// Allocated : 2 SGPRs (Value: Bool64) (op 470): s14, s15
v_add_co_u32 v26, s[14:15], v26, v12 // (op 470) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 470) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 470): s14, s15

 // (op 470) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 470) reg expression
 // (op 470) Add(m0:U32, 1024:U32)U32
 // (op 470) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 470) 
 // (op 470) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 470) Load value direct to lds
// VMEM: Expected stall of 464. CBNW: 0, Inc: 3
// VMEM: Expected complete at 5145 (current 4665)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing Stride514: 2 VGPRs (Value: UInt64) (op 470): v12, v13

 // Freeing : 2 VGPRs (Value: UInt64) (op 470): v26, v27

 // (op 227) LoadTileDirect2LDS Value: BF8x4(470) END
 // (op 227) Deallocate{}(1367) BEGIN
 // (op 1367) Deallocate 594
 // Deleting tag 594
 // (op 1367) Deallocate 515
 // Deleting tag 515
 // (op 1367) Deallocate 513
 // Freeing Offset513: 2 VGPRs (Value: UInt64) (op 470): v20, v21

 // Deleting tag 513
 // (op 1367) Deallocate 452
 // Deleting tag 452
 // (op 227) Deallocate{}(1367) END
 // (op 227) Assign VGPR Add(DataFlowTag(509)U64, DataFlowTag(510)U64)U64(571) BEGIN
 // (op 571) Assign dim(509) = Add(DataFlowTag(509)U64, DataFlowTag(510)U64)U64
 // (op 571) Generate Add(DataFlowTag(509)U64, DataFlowTag(510)U64)U64 into DataFlowTag509: VGPR Value: UInt64 x 1: v[14:15]
 // (op 571) reg expression
 // (op 571) Add({DataFlowTag509: v[14:15]:U64}, 64:U64)U64
 // (op 571) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 571)      lhs (DataFlowTag509: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 571)      rhs (Literal Value: UInt64 x 0: 64)
// Allocated : 2 SGPRs (Value: Bool64) (op 571): s14, s15
v_add_co_u32 v14, s[14:15], 64, v14 // (op 571) least significant half
v_addc_co_u32 v15, s[14:15], 0, v15, s[14:15] // (op 571) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 571): s14, s15

 // (op 227) Assign VGPR Add(DataFlowTag(509)U64, DataFlowTag(510)U64)U64(571) END
 // (op 227) end: generate({470})
 // (op 171) SetCoordinate(227) END
 // (op 171) SetCoordinate(232) BEGIN
 // (op 232) generate({})
 // (op 232) end: generate({})
 // (op 232) generate({191})
 // (op 232) LoadTiled Value: E8M0(191) BEGIN
 // (op 191) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 191) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 191) Generate 1:U64 into nullptr
 // tag 454: v**UNALLOCATED**
 // (op 191) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 532: v**UNALLOCATED**
 // (op 171) NOP(472) BEGIN
 // (op 472) generate({})
 // (op 472) end: generate({})
 // (op 171) NOP(472) END
// Allocated Offset532: 1 VGPR (Value: UInt32) (op 191): v16
v_mov_b32 v16, v25 // (op 191) 
 // (op 191) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset532: VGPR Value: UInt32 x 1: v16
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
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 191) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 191): v24
buffer_load_ubyte v24, v16, s[48:51], 0 offen offset: 0 // (op 191) Load value
// VMEM: Expected complete at 5149 (current 4669)
 // (op 232) LoadTiled Value: E8M0(191) END
 // (op 232) Deallocate{}(1307) BEGIN
 // (op 1307) Deallocate 532
 // Freeing Offset532: 1 VGPR (Value: UInt32) (op 191): v16

 // Deleting tag 532
 // (op 1307) Deallocate 534
 // Deleting tag 534
 // (op 232) Deallocate{}(1307) END
 // (op 232) Assign VGPR Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32(635) BEGIN
 // (op 635) Assign dim(528) = Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32
 // (op 635) Generate Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32 into DataFlowTag528: VGPR Value: UInt32 x 1: v25
 // (op 635) reg expression
 // (op 635) Add({DataFlowTag528: v25:U32}, 2:U32)U32
v_add_u32 v25, 2, v25 // (op 635) 
 // (op 232) Assign VGPR Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32(635) END
 // (op 232) end: generate({191})
 // (op 171) SetCoordinate(232) END
 // (op 171) SetCoordinate(237) BEGIN
 // (op 237) generate({})
 // (op 237) end: generate({})
 // (op 237) generate({192})
 // (op 237) LoadTiled Value: E8M0(192) BEGIN
 // (op 192) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 192) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 192) Generate 1:U64 into nullptr
 // tag 456: v**UNALLOCATED**
 // (op 192) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 551: v**UNALLOCATED**
 // (op 171) Barrier(1226) BEGIN
s_barrier  // (op 1226) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 171) Barrier(1226) END
// Allocated Offset551: 1 VGPR (Value: UInt32) (op 192): v16
v_mov_b32 v16, v17 // (op 192) 
 // (op 192) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset551: VGPR Value: UInt32 x 1: v16
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
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
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

 // (op 171) StoreLDSTile Value: E8M0(188) BEGIN
 // (op 188) GEN: StoreLDSTile
 // (op 188) GEN: storeMacroTileLDS OP 188 LDS 33 MacroTile 454
 // (op 188) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 188) Generate 1:U32 into nullptr
 // (op 188) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

 // (op 188) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v24
 // 	info.rowOffsetReg = DataFlowTag608: VGPR Value: UInt32 x 1: v22
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 188) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 192): v12
buffer_load_ubyte v12, v16, s[52:55], 0 offen offset: 0 // (op 192) Load value
// VMEM: Expected stall of 5. CBNW: 2, Inc: 3
// VMEM: Expected complete at 5158 (current 4678)
 // (op 237) LoadTiled Value: E8M0(192) END
 // (op 237) Deallocate{}(1309) BEGIN
 // (op 1309) Deallocate 551
 // Freeing Offset551: 1 VGPR (Value: UInt32) (op 192): v16

 // Deleting tag 551
 // (op 1309) Deallocate 553
 // Deleting tag 553
 // (op 237) Deallocate{}(1309) END
 // (op 237) Assign VGPR Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32(699) BEGIN
 // (op 699) Assign dim(547) = Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32
 // (op 699) Generate Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32 into DataFlowTag547: VGPR Value: UInt32 x 1: v17
 // (op 699) reg expression
 // (op 699) Add({DataFlowTag547: v17:U32}, 2:U32)U32
v_add_u32 v17, 2, v17 // (op 699) 
 // (op 237) Assign VGPR Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32(699) END
 // (op 237) end: generate({192})
 // (op 171) SetCoordinate(237) END
s_waitcnt vmcnt(1)
ds_write_b8 v22, v24 offset:8256 // (op 188) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 4757 (current 4662)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 171) StoreLDSTile Value: E8M0(188) END
 // (op 171) Barrier(1199) BEGIN
 // (op 171) Deallocate{}(1303) BEGIN
 // (op 1303) Deallocate 454
 // Freeing : 1 VGPR (Value: E8M0) (op 191): v24

 // Deleting tag 454
 // (op 1303) Deallocate 610
 // Deleting tag 610
 // (op 171) Deallocate{}(1303) END
s_barrier  // (op 1199) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 171) Barrier(1199) END
 // (op 171) Barrier(1243) BEGIN
 // (op 171) SetCoordinate(178) BEGIN
 // (op 178) generate({})
 // (op 178) end: generate({})
 // (op 178) generate({179})
 // (op 178) LoadLDSTile{Value: BF8x4}(179) BEGIN
 // (op 179) GEN: loadMacroTileWAVELDS OP 179 LDS 17 WaveTile 103
 // (op 179) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 179) Generate 1:U32 into nullptr
 // tag 458: v**UNALLOCATED**
 // (op 179) 	Dir = Load
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

 // (op 179) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag584: VGPR Value: UInt32 x 1: v3
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
 // (op 179) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

s_barrier  // (op 1243) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 171) Barrier(1243) END
// Allocated : 8 VGPRs (Value: BF8x4) (op 179): v26, v27, v28, v29, v30, v31, v32, v33
ds_read_b128 v[26:29], v3 offset:2048 // (op 179) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4760 (current 4665)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 171) StoreLDSTile Value: E8M0(189) BEGIN
 // (op 189) GEN: StoreLDSTile
 // (op 189) GEN: storeMacroTileLDS OP 189 LDS 41 MacroTile 456
 // (op 189) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 189) Generate 1:U32 into nullptr
 // (op 189) 	Dir = Store
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
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = DataFlowTag624: VGPR Value: UInt32 x 1: v23
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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

ds_read_b128 v[30:33], v3 offset:2080 // (op 179) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4761 (current 4666)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 178) LoadLDSTile{Value: BF8x4}(179) END
 // (op 178) end: generate({179})
 // (op 171) SetCoordinate(178) END
s_waitcnt vmcnt(0)
ds_write_b8 v23, v12 offset:8192 // (op 189) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 91. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4854 (current 4759)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 171) StoreLDSTile Value: E8M0(189) END
 // (op 171) Barrier(1216) BEGIN
 // (op 171) Deallocate{}(1305) BEGIN
 // (op 1305) Deallocate 456
 // Freeing : 1 VGPR (Value: E8M0) (op 192): v12

 // Deleting tag 456
 // (op 1305) Deallocate 626
 // Deleting tag 626
 // (op 171) Deallocate{}(1305) END
s_barrier  // (op 1216) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 171) Barrier(1216) END
 // (op 171) SetCoordinate(180) BEGIN
 // (op 180) generate({})
 // (op 180) end: generate({})
 // (op 180) generate({181})
 // (op 180) LoadLDSTile{Value: BF8x4}(181) BEGIN
 // (op 181) GEN: loadMacroTileWAVELDS OP 181 LDS 25 WaveTile 169
 // (op 181) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 181) Generate 1:U32 into nullptr
 // tag 460: v**UNALLOCATED**
 // (op 181) 	Dir = Load
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

 // (op 181) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag600: VGPR Value: UInt32 x 1: v4
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
 // (op 181) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated : 8 VGPRs (Value: BF8x4) (op 181): v34, v35, v36, v37, v38, v39, v40, v41
ds_read_b128 v[34:37], v4 // (op 181) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4856 (current 4761)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
ds_read_b128 v[38:41], v4 offset:32 // (op 181) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4857 (current 4762)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 180) LoadLDSTile{Value: BF8x4}(181) END
 // (op 180) end: generate({181})
 // (op 171) SetCoordinate(180) END
 // (op 171) Barrier(1233) BEGIN
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1233) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 171) Barrier(1233) END
 // (op 171) SetCoordinate(182) BEGIN
 // (op 182) generate({})
 // (op 182) end: generate({})
 // (op 182) generate({183})
 // (op 182) LoadLDSTile{Value: E8M0}(183) BEGIN
 // (op 183) GEN: loadMacroTileWAVELDS OP 183 LDS 33 WaveTile 235
 // (op 183) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 183) Generate 1:U32 into nullptr
 // tag 462: v**UNALLOCATED**
 // (op 183) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag616: VGPR Value: UInt32 x 1: v0
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 183): v16
ds_read_u8 v16, v0 offset:8256 // (op 183) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected stall of 90. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 4950 (current 4855)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 182) LoadLDSTile{Value: E8M0}(183) END
 // (op 182) end: generate({183})
 // (op 171) SetCoordinate(182) END
 // (op 171) Barrier(1250) BEGIN
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1250) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 171) Barrier(1250) END
 // (op 171) SetCoordinate(184) BEGIN
 // (op 184) generate({})
 // (op 184) end: generate({})
 // (op 184) generate({185})
 // (op 184) LoadLDSTile{Value: E8M0}(185) BEGIN
 // (op 185) GEN: loadMacroTileWAVELDS OP 185 LDS 41 WaveTile 295
 // (op 185) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 185) Generate 1:U32 into nullptr
 // tag 464: v**UNALLOCATED**
 // (op 185) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag632: VGPR Value: UInt32 x 1: v6
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
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 185): v24
ds_read_u8 v24, v6 offset:8192 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4953 (current 4858)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 184) LoadLDSTile{Value: E8M0}(185) END
 // (op 184) end: generate({185})
 // (op 171) SetCoordinate(184) END
 // (op 171) Multiply(186) BEGIN
 // (op 186) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile4xBF8, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[26:33], v[34:41], a[0:15], v16, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 186) 
 // (op 171) Multiply(186) END
 // (op 171) Deallocate{}(1301) BEGIN
 // (op 1301) Deallocate 458
 // Freeing : 8 VGPRs (Value: BF8x4) (op 179): v26, v27, v28, v29, v30, v31, v32, v33

 // Deleting tag 458
 // (op 1301) Deallocate 460
 // Freeing : 8 VGPRs (Value: BF8x4) (op 181): v34, v35, v36, v37, v38, v39, v40, v41

 // Deleting tag 460
 // (op 1301) Deallocate 462
 // Freeing : 1 VGPR (Value: E8M0) (op 183): v16

 // Deleting tag 462
 // (op 1301) Deallocate 464
 // Freeing : 1 VGPR (Value: E8M0) (op 185): v24

 // Deleting tag 464
 // (op 171) Deallocate{}(1301) END
 // (op 171) end: generate({173, 1193})
 // (op 171) For Loop Increment
 // (op 171) generate({})
 // (op 171) end: generate({})
 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
 // (op 171) Generate LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 171) LessThan({DataFlowTag448: s18:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 171) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 171): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 171) 
// Allocated : 1 SGPR (Value: UInt32) (op 171): s19
s_mov_b32 s19, s14 // (op 171) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 171): s14, s15

s_cmp_lt_u32 s18, s19 // (op 171) 
 // Freeing : 1 SGPR (Value: UInt32) (op 171): s19

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 171) 
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171:
 // (op 171) 
 // (op 171) Unlock For Loop
 // Unlock None
 // (op 507) ForLoopOp KLoopTail: LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(171) END
 // (op 507) Deallocate{}(1299) BEGIN
 // (op 1299) Deallocate 529
 // Deleting tag 529
 // (op 1299) Deallocate 490
 // Freeing DataFlowTag490: 2 VGPRs (Value: UInt64) (op 512): v18, v19

 // Deleting tag 490
 // (op 1299) Deallocate 535
 // Deleting tag 535
 // (op 1299) Deallocate 593
 // Deleting tag 593
 // (op 1299) Deallocate 576
 // Freeing DataFlowTag576: 1 SGPR (Value: UInt32) (op 800): s0

 // Deleting tag 576
 // (op 1299) Deallocate 448
 // Freeing DataFlowTag448: 1 SGPR (Value: UInt32) (op 172): s18

 // Deleting tag 448
 // (op 1299) Deallocate 625
 // Deleting tag 625
 // (op 1299) Deallocate 577
 // Deleting tag 577
 // (op 1299) Deallocate 627
 // Deleting tag 627
 // (op 1299) Deallocate 509
 // Freeing DataFlowTag509: 2 VGPRs (Value: UInt64) (op 579): v14, v15

 // Deleting tag 509
 // (op 1299) Deallocate 611
 // Deleting tag 611
 // (op 1299) Deallocate 552
 // Deleting tag 552
 // (op 1299) Deallocate 609
 // Deleting tag 609
 // (op 1299) Deallocate 608
 // Freeing DataFlowTag608: 1 VGPR (Value: UInt32) (op 904): v22

 // Deleting tag 608
 // (op 1299) Deallocate 595
 // Deleting tag 595
 // (op 1299) Deallocate 592
 // Freeing DataFlowTag592: 1 SGPR (Value: UInt32) (op 852): s1

 // Deleting tag 592
 // (op 1299) Deallocate 547
 // Freeing DataFlowTag547: 1 VGPR (Value: UInt32) (op 707): v17

 // Deleting tag 547
 // (op 1299) Deallocate 491
 // Deleting tag 491
 // (op 1299) Deallocate 497
 // Deleting tag 497
 // (op 1299) Deallocate 624
 // Freeing DataFlowTag624: 1 VGPR (Value: UInt32) (op 957): v23

 // Deleting tag 624
 // (op 1299) Deallocate 510
 // Deleting tag 510
 // (op 1299) Deallocate 514
 // Deleting tag 514
 // (op 1299) Deallocate 516
 // Deleting tag 516
 // (op 1299) Deallocate 528
 // Freeing DataFlowTag528: 1 VGPR (Value: UInt32) (op 643): v25

 // Deleting tag 528
 // (op 1299) Deallocate 548
 // Deleting tag 548
 // (op 1299) Deallocate 495
 // Deleting tag 495
 // (op 1299) Deallocate 554
 // Deleting tag 554
 // (op 1299) Deallocate 579
 // Deleting tag 579
 // (op 1299) Deallocate 533
 // Deleting tag 533
 // (op 507) Deallocate{}(1299) END
 // (op 507) end: generate({501})
 // (op 507) Unlock Scope 507
 // Unlock None
 // (op 218) Scope(507) END
 // (op 218) end: generate({507})
 // (op 217) SetCoordinate(218) END
 // (op 217) end: generate({218})
 // (op 340) SetCoordinate(217) END
 // (op 340) Deallocate{}(1311) BEGIN
 // (op 1311) Deallocate 157
 // Deleting tag 157
 // (op 1311) Deallocate 142
 // Deleting tag 142
 // (op 1311) Deallocate 586
 // Deleting tag 586
 // (op 1311) Deallocate 618
 // Deleting tag 618
 // (op 1311) Deallocate 4
 // Deleting tag 4
 // (op 1311) Deallocate 512
 // Freeing DataFlowTag512: 4 SGPRs (Buffer: None) (op 604): s44, s45, s46, s47

 // Deleting tag 512
 // (op 1311) Deallocate 429
 // Deleting tag 429
 // (op 1311) Deallocate 208
 // Deleting tag 208
 // (op 1311) Deallocate 493
 // Freeing DataFlowTag493: 4 SGPRs (Buffer: None) (op 540): s40, s41, s42, s43

 // Deleting tag 493
 // (op 1311) Deallocate 427
 // Deleting tag 427
 // (op 1311) Deallocate 76
 // Deleting tag 76
 // (op 1311) Deallocate 41
 // Deleting tag 41
 // (op 1311) Deallocate 268
 // Deleting tag 268
 // (op 1311) Deallocate 634
 // Deleting tag 634
 // (op 1311) Deallocate 602
 // Deleting tag 602
 // (op 1311) Deallocate 223
 // Deleting tag 223
 // (op 1311) Deallocate 3
 // Deleting tag 3
 // (op 1311) Deallocate 25
 // Deleting tag 25
 // (op 1311) Deallocate 91
 // Deleting tag 91
 // (op 1311) Deallocate 433
 // Deleting tag 433
 // (op 1311) Deallocate 33
 // Deleting tag 33
 // (op 1311) Deallocate 550
 // Freeing DataFlowTag550: 4 SGPRs (Buffer: None) (op 732): s52, s53, s54, s55

 // Deleting tag 550
 // (op 1311) Deallocate 283
 // Deleting tag 283
 // (op 1311) Deallocate 431
 // Deleting tag 431
 // (op 1311) Deallocate 17
 // Deleting tag 17
 // (op 1311) Deallocate 531
 // Freeing DataFlowTag531: 4 SGPRs (Buffer: None) (op 668): s48, s49, s50, s51

 // Deleting tag 531
 // (op 340) Deallocate{}(1311) END
 // (op 340) Deallocate{Tensor_4_stride_0_6, Tensor_0_stride_0_0, Tensor_7_pointer, Tensor_0_pointer, Tensor_2_pointer, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_0_size_0_1, Tensor_2_stride_1_3, Tensor_4_pointer}(1441) BEGIN
 // (op 1441) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1441) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1441) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1441) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1441) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 1441) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1441) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1441) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1441) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 1441) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 340) Deallocate{Tensor_4_stride_0_6, Tensor_0_stride_0_0, Tensor_7_pointer, Tensor_0_pointer, Tensor_2_pointer, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_0_size_0_1, Tensor_2_stride_1_3, Tensor_4_pointer}(1441) END
 // (op 340) Barrier(1313) BEGIN
s_barrier  // (op 1313) 
 // (op 340) Barrier(1313) END
 // (op 340) end: generate({345})
 // (op 340) Unlock Scope 340
 // Unlock None
 // (op 1) Scope(340) END
 // (op 1) Deallocate{}(1345) BEGIN
 // (op 1345) Deallocate 683
 // Deleting tag 683
 // (op 1345) Deallocate 654
 // Freeing DataFlowTag654: 1 VGPR (Value: UInt32) (op 1058): v7

 // Deleting tag 654
 // (op 1345) Deallocate 584
 // Freeing DataFlowTag584: 1 VGPR (Value: UInt32) (op 827): v3

 // Deleting tag 584
 // (op 1345) Deallocate 666
 // Freeing DataFlowTag666: 1 VGPR (Value: UInt32) (op 1097): v8

 // Deleting tag 666
 // (op 1345) Deallocate 633
 // Deleting tag 633
 // (op 1345) Deallocate 632
 // Freeing DataFlowTag632: 1 VGPR (Value: UInt32) (op 985): v6

 // Deleting tag 632
 // (op 1345) Deallocate 682
 // Freeing DataFlowTag682: 1 VGPR (Value: UInt32) (op 1145): v10

 // Deleting tag 682
 // (op 1345) Deallocate 603
 // Deleting tag 603
 // (op 1345) Deallocate 675
 // Deleting tag 675
 // (op 1345) Deallocate 674
 // Freeing DataFlowTag674: 1 VGPR (Value: UInt32) (op 1121): v5

 // Deleting tag 674
 // (op 1345) Deallocate 587
 // Deleting tag 587
 // (op 1345) Deallocate 657
 // Deleting tag 657
 // (op 1345) Deallocate 635
 // Deleting tag 635
 // (op 1345) Deallocate 677
 // Deleting tag 677
 // (op 1345) Deallocate 619
 // Deleting tag 619
 // (op 1345) Deallocate 667
 // Deleting tag 667
 // (op 1345) Deallocate 685
 // Deleting tag 685
 // (op 1345) Deallocate 655
 // Deleting tag 655
 // (op 1345) Deallocate 616
 // Freeing DataFlowTag616: 1 VGPR (Value: UInt32) (op 932): v0

 // Deleting tag 616
 // (op 1345) Deallocate 617
 // Deleting tag 617
 // (op 1345) Deallocate 601
 // Deleting tag 601
 // (op 1345) Deallocate 600
 // Freeing DataFlowTag600: 1 VGPR (Value: UInt32) (op 879): v4

 // Deleting tag 600
 // (op 1345) Deallocate 669
 // Deleting tag 669
 // (op 1345) Deallocate 585
 // Deleting tag 585
 // (op 1) Deallocate{}(1345) END
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(133) BEGIN
 // (op 133) Lock for Conditional Branch
 // Lock Branch
 // (op 133) Generate Equal(0.00000:S, DataFlowTag(44)S)BL into VCC Value: Bool64 x 1: vcc
 // (op 133) reg expression
 // (op 133) Equal(0.00000:S, v9:S)BL64
v_cmp_eq_f32 vcc, 0.00000, v9 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
 // (op 133) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 133) 
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133
 // (op 133) generate({348})
 // (op 133) Scope(348) BEGIN
 // (op 348) Lock Scope 348
 // Lock Branch
 // (op 348) generate({351})
 // (op 348) NOP(351) BEGIN
 // (op 351) generate({})
 // (op 351) end: generate({})
 // (op 348) NOP(351) END
 // (op 348) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(50) BEGIN
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

 // (op 348) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(50) END
 // (op 348) NOP(995) BEGIN
 // (op 995) generate({})
 // (op 995) end: generate({})
 // (op 348) NOP(995) END
 // (op 348) NOP(992) BEGIN
 // (op 992) generate({})
 // (op 992) end: generate({})
 // (op 348) NOP(992) END
 // (op 348) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(999) BEGIN
 // (op 999) Assign dim(636) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 636: v**UNALLOCATED**
 // (op 999) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag636: VGPR Value: UInt32 x 1: (unallocated)
 // (op 999) reg expression
 // (op 999) reg expression
 // (op 999) reg expression
 // (op 999) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 999) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 999): v0
v_lshrrev_b32 v0, 6, v1 // (op 999) 
 // (op 999) convert
 // (op 999) BEGIN: Tile[1]
 // (op 999) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 999): v10
v_and_b32 v10, 63, v1 // (op 999) 
 // (op 999) END: Tile[1]
 // (op 999) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 999): v3
v_lshrrev_b32 v3, 5, v10 // (op 999) 
 // (op 999) BEGIN: Flatten
 // (op 999) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 999) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 999)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 999)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 999): v4
v_lshlrev_b32 v4, 2, v3 // (op 999) 
 // (op 999) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 999): v3

 // (op 999) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 999): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 999) 
 // (op 999) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 999): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 999): v0

 // (op 999) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 999): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 999) 
 // (op 999) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 999): v3

 // (op 999) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 999): v3
v_lshrrev_b32 v3, 2, v10 // (op 999) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 999): v10

 // (op 999) BEGIN: Tile[1]
 // (op 999) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 999): v10
v_and_b32 v10, 7, v3 // (op 999) 
 // (op 999) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 999): v3

 // (op 999) BEGIN: Tile[1]
 // (op 999) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 999): v3
v_and_b32 v3, 3, v1 // (op 999) 
 // (op 999) END: Tile[1]
 // (op 999) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 999): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 999) 
 // (op 999) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 999): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 999): v10

 // (op 999) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 999): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 999) 
 // (op 999) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 999): v4

 // (op 999) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 999): v10
v_mul_lo_u32 v10, v3, s30 // (op 999) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 999): v3

 // (op 999) BEGIN: Join
// Allocated DataFlowTag636: 1 VGPR (Value: UInt32) (op 999): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 999) 
 // (op 999) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 999): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 999): v0

 // (op 348) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(999) END
 // (op 348) Assign VGPR 32:U64(1000) BEGIN
 // (op 1000) Assign dim(637) = 32:U64
 // (op 1000) Assign stride expression
 // (op 348) Assign VGPR 32:U64(1000) END
 // (op 348) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(1001) BEGIN
 // (op 1001) Assign dim(639) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 639: s**UNALLOCATED**
 // (op 1001) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag639: SGPR Buffer: None x 1: (unallocated)
 // (op 1001) reg expression
 // (op 1001) reg expression
 // (op 1001) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 1001) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 1001)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 1001)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 1001): s14, s15
s_mul_i32 s15, s30, s13 // (op 1001) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 1001): s0
s_mul_i32 s0, s31, s12 // (op 1001) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1001) least significant
// Allocated : 1 SGPR (Value: Int32) (op 1001): s1
s_mul_hi_u32 s1, s30, s12 // (op 1001) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1001) most significant: sum
s_add_u32 s15, s15, s1 // (op 1001) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 1001): s1

 // Freeing : 1 SGPR (Value: Int32) (op 1001): s0

 // (op 1001) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 1001) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 1001)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 1001)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 1001): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 1001) 
 // Freeing : 2 SGPRs (Value: Int64) (op 1001): s14, s15

 // (op 1001) reg expression
 // (op 1001) reg expression
// Allocated DataFlowTag639: 4 SGPRs (Buffer: None) (op 1001): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 1001) reg expression
s_mov_b32 s9, s33 // (op 1001) reg expression
s_mov_b32 s10, s0 // (op 1001) reg expression
s_mov_b32 s11, 131072 // (op 1001) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 1001): s0, s1

 // (op 348) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(1001) END
 // (op 348) NOP(993) BEGIN
 // (op 993) generate({})
 // (op 993) end: generate({})
 // (op 348) NOP(993) END
 // (op 348) Assign VGPR 4:U64(1006) BEGIN
 // (op 1006) Assign dim(641) = 4:U64
 // (op 1006) Assign stride expression
 // (op 348) Assign VGPR 4:U64(1006) END
 // (op 348) StoreTiled Value: Float(56) BEGIN
 // (op 56) GEN: storeMacroTileWAVE OP 56 MacroTile 59 WaveTile 365
 // (op 56) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 56) Generate 32:U64 into nullptr
 // (op 56) Generate 4:U64 into nullptr
 // (op 56) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag59: VGPR Value: Float x 16: v[28:43]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
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
 // 	info.bufDesc = DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 56) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag59: VGPR Value: Float x 16: v[28:43]
 // 	info.rowOffsetReg = DataFlowTag636: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
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
 // 	info.bufDesc = DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 56) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx4 v[28:31], v3, s[8:11], 0 offen // (op 56) Store value
// VMEM: Expected complete at 5904 (current 5424)
buffer_store_dwordx4 v[32:35], v3, s[8:11], 0 offen offset:32 // (op 56) Store value
// VMEM: Expected complete at 5905 (current 5425)
buffer_store_dwordx4 v[36:39], v3, s[8:11], 0 offen offset:64 // (op 56) Store value
// VMEM: Expected complete at 5906 (current 5426)
buffer_store_dwordx4 v[40:43], v3, s[8:11], 0 offen offset:96 // (op 56) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 6385 (current 5905)
 // (op 348) StoreTiled Value: Float(56) END
 // (op 348) Deallocate{}(1283) BEGIN
 // (op 1283) Deallocate 59
 // Freeing DataFlowTag59: 16 VGPRs (Value: Float) (op 50): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 59
 // (op 1283) Deallocate 640
 // Deleting tag 640
 // (op 1283) Deallocate 636
 // Freeing DataFlowTag636: 1 VGPR (Value: UInt32) (op 999): v3

 // Deleting tag 636
 // (op 1283) Deallocate 641
 // Deleting tag 641
 // (op 1283) Deallocate 637
 // Deleting tag 637
 // (op 348) Deallocate{}(1283) END
 // (op 348) end: generate({351})
 // (op 348) Unlock Scope 348
 // Unlock None
 // (op 133) Scope(348) END
 // (op 133) end: generate({348})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 133) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 133) 
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133:
 // (op 133) 
 // (op 133) generate({354})
 // (op 133) Scope(354) BEGIN
 // (op 354) Lock Scope 354
 // Lock Branch
 // (op 354) generate({357})
 // (op 354) NOP(357) BEGIN
 // (op 357) generate({})
 // (op 357) end: generate({})
 // (op 354) NOP(357) END
 // (op 354) NOP(141) BEGIN
 // (op 141) generate({})
 // (op 141) end: generate({})
 // (op 354) NOP(141) END
 // (op 354) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(142) BEGIN
 // (op 142) Assign dim(423) = Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA
 // (op 142) Generate Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA into nullptr
 // (op 142) reg expression
 // (op 142) reg expression
 // (op 142) Multiply(v11:S, {DataFlowTag55: a[0:15]:S})S
 // (op 354) Scope(765) BEGIN
 // (op 765) Lock Scope 765
 // Lock Branch
 // (op 765) generate({762})
 // (op 765) NOP(762) BEGIN
 // (op 762) generate({})
 // (op 762) end: generate({})
 // (op 765) NOP(762) END
 // (op 765) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(770) BEGIN
 // (op 770) Assign dim(566) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}
 // tag 566: v**UNALLOCATED**
 // (op 770) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32} into DataFlowTag566: VGPR Value: UInt32 x 1: (unallocated)
 // (op 770) reg expression
 // (op 770) reg expression
 // (op 770) reg expression
 // (op 770) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 770) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 770): v0
v_lshrrev_b32 v0, 6, v1 // (op 770) 
 // (op 770) convert
 // (op 770) BEGIN: Flatten[1]
 // (op 770) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 770): v10
v_and_b32 v10, 63, v1 // (op 770) 
 // (op 770) END: Flatten[1]
 // (op 770) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 770): v3
v_lshrrev_b32 v3, 5, v10 // (op 770) 
 // (op 770) BEGIN: Tile
 // (op 770) {Tile: ShiftL(v3:U32, 2:U32)U32}
 // (op 770) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 770)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 770)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 770): v4
v_lshlrev_b32 v4, 2, v3 // (op 770) 
 // (op 770) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 770): v3

 // (op 770) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 770): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 770) 
 // (op 770) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 770): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 770): v0

 // (op 770) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 770): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 770) 
 // (op 770) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 770): v3

 // (op 770) LogicalShiftR({Flatten[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 770): v3
v_lshrrev_b32 v3, 2, v10 // (op 770) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 770): v10

 // (op 770) BEGIN: Flatten[1]
 // (op 770) {Flatten[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 770): v10
v_and_b32 v10, 7, v3 // (op 770) 
 // (op 770) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 770): v3

 // (op 770) BEGIN: Flatten[1]
 // (op 770) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 770): v3
v_and_b32 v3, 3, v1 // (op 770) 
 // (op 770) END: Flatten[1]
 // (op 770) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 770): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 770) 
 // (op 770) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 770): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 770): v10

 // (op 770) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 770): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 770) 
 // (op 770) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 770): v4

 // (op 770) Multiply({Tile: v3:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 770): v10
v_mul_lo_u32 v10, v3, s34 // (op 770) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 770): v3

 // (op 770) BEGIN: Split
// Allocated DataFlowTag566: 1 VGPR (Value: UInt32) (op 770): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 770) 
 // (op 770) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 770): v10

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 770): v0

 // (op 765) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(770) END
 // (op 765) Deallocate{}(1381) BEGIN
 // (op 1381) Deallocate 435
 // Deleting tag 435
 // (op 1381) Deallocate 437
 // Deleting tag 437
 // (op 1381) Deallocate 324
 // Deleting tag 324
 // (op 765) Deallocate{}(1381) END
 // (op 765) Assign VGPR 32:U64(771) BEGIN
 // (op 771) Assign dim(567) = 32:U64
 // (op 771) Assign stride expression
 // (op 765) Assign VGPR 32:U64(771) END
 // (op 765) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(772) BEGIN
 // (op 772) Assign dim(569) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 569: s**UNALLOCATED**
 // (op 772) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag569: SGPR Buffer: None x 1: (unallocated)
 // (op 772) reg expression
 // (op 772) reg expression
 // (op 772) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 772) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 772)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 772)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 772): s14, s15
s_mul_i32 s15, s34, s13 // (op 772) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 772): s0
s_mul_i32 s0, s35, s12 // (op 772) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 772) least significant
// Allocated : 1 SGPR (Value: Int32) (op 772): s1
s_mul_hi_u32 s1, s34, s12 // (op 772) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 772) most significant: sum
s_add_u32 s15, s15, s1 // (op 772) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 772): s1

 // Freeing : 1 SGPR (Value: Int32) (op 772): s0

 // (op 772) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 772) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 772)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 772)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 772): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 772) 
 // Freeing : 2 SGPRs (Value: Int64) (op 772): s14, s15

 // (op 772) reg expression
 // (op 772) reg expression
// Allocated DataFlowTag569: 4 SGPRs (Buffer: None) (op 772): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 772) reg expression
s_mov_b32 s5, s37 // (op 772) reg expression
s_mov_b32 s6, s0 // (op 772) reg expression
s_mov_b32 s7, 131072 // (op 772) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 772): s0, s1

 // (op 765) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(772) END
 // (op 765) Deallocate{Tensor_10_pointer}(1445) BEGIN
 // (op 1445) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 765) Deallocate{Tensor_10_pointer}(1445) END
 // (op 765) NOP(763) BEGIN
 // (op 763) generate({})
 // (op 763) end: generate({})
 // (op 765) NOP(763) END
 // (op 765) Assign VGPR 4:U64(777) BEGIN
 // (op 777) Assign dim(571) = 4:U64
 // (op 777) Assign stride expression
 // (op 765) Assign VGPR 4:U64(777) END
 // (op 765) LoadTiled Value: Float(145) BEGIN
 // (op 145) GEN: loadMacroTileWAVECIACCUM OP 145 WaveTile 325
 // (op 145) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 145) Generate 32:U64 into nullptr
 // (op 145) Generate 4:U64 into nullptr
 // tag 419: v**UNALLOCATED**
 // (op 145) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
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
 // 	info.bufDesc = DataFlowTag569: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 145) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag566: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
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
 // 	info.bufDesc = DataFlowTag569: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 145) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
buffer_load_dwordx4 v[12:15], v3, s[4:7], 0 offen // (op 145) Load value
// VMEM: Expected complete at 6893 (current 6413)
buffer_load_dwordx4 v[16:19], v3, s[4:7], 0 offen offset:32 // (op 145) Load value
// VMEM: Expected complete at 6894 (current 6414)
buffer_load_dwordx4 v[20:23], v3, s[4:7], 0 offen offset:64 // (op 145) Load value
// VMEM: Expected complete at 6895 (current 6415)
buffer_load_dwordx4 v[24:27], v3, s[4:7], 0 offen offset:96 // (op 145) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 7374 (current 6894)
 // (op 765) LoadTiled Value: Float(145) END
 // (op 765) Deallocate{Tensor_10_stride_1_10}(1295) BEGIN
 // (op 1295) Deallocate 5
 // Deleting tag 5
 // (op 1295) Deallocate 566
 // Freeing DataFlowTag566: 1 VGPR (Value: UInt32) (op 770): v3

 // Deleting tag 566
 // (op 1295) Deallocate 570
 // Deleting tag 570
 // (op 1295) Deallocate 571
 // Deleting tag 571
 // (op 1295) Deallocate 569
 // Freeing DataFlowTag569: 4 SGPRs (Buffer: None) (op 772): s4, s5, s6, s7

 // Deleting tag 569
 // (op 1295) Deallocate 567
 // Deleting tag 567
 // (op 1295) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 765) Deallocate{Tensor_10_stride_1_10}(1295) END
 // (op 765) end: generate({762})
 // (op 765) Unlock Scope 765
 // Unlock None
 // (op 354) Scope(765) END
// Allocated : 16 VGPRs (Value: Float) (op 142): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43
v_accvgpr_read v28, a0 // (op 142) 
 // (op 354) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(419)NA)NA(146) BEGIN
 // (op 146) Assign dim(421) = Multiply(DataFlowTag(44)NA, DataFlowTag(419)NA)NA
 // (op 146) Generate Multiply(DataFlowTag(44)NA, DataFlowTag(419)NA)NA into nullptr
 // (op 146) reg expression
 // (op 146) reg expression
 // (op 146) Multiply(v9:S, v[12:27]:S)S
v_accvgpr_read v29, a1 // (op 142) 
v_accvgpr_read v30, a2 // (op 142) 
v_accvgpr_read v31, a3 // (op 142) 
v_accvgpr_read v32, a4 // (op 142) 
v_accvgpr_read v33, a5 // (op 142) 
v_accvgpr_read v34, a6 // (op 142) 
v_accvgpr_read v35, a7 // (op 142) 
v_accvgpr_read v36, a8 // (op 142) 
v_accvgpr_read v37, a9 // (op 142) 
v_accvgpr_read v38, a10 // (op 142) 
v_accvgpr_read v39, a11 // (op 142) 
v_accvgpr_read v40, a12 // (op 142) 
v_accvgpr_read v41, a13 // (op 142) 
v_accvgpr_read v42, a14 // (op 142) 
v_accvgpr_read v43, a15 // (op 142) 
// Allocated : 16 VGPRs (Value: Float) (op 142): v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59
v_mul_f32 v44, v11, v28 // (op 142) 
v_mul_f32 v45, v11, v29 // (op 142) 
v_mul_f32 v46, v11, v30 // (op 142) 
v_mul_f32 v47, v11, v31 // (op 142) 
v_mul_f32 v48, v11, v32 // (op 142) 
v_mul_f32 v49, v11, v33 // (op 142) 
v_mul_f32 v50, v11, v34 // (op 142) 
v_mul_f32 v51, v11, v35 // (op 142) 
v_mul_f32 v52, v11, v36 // (op 142) 
v_mul_f32 v53, v11, v37 // (op 142) 
v_mul_f32 v54, v11, v38 // (op 142) 
v_mul_f32 v55, v11, v39 // (op 142) 
v_mul_f32 v56, v11, v40 // (op 142) 
v_mul_f32 v57, v11, v41 // (op 142) 
v_mul_f32 v58, v11, v42 // (op 142) 
v_mul_f32 v59, v11, v43 // (op 142) 
 // Freeing : 16 VGPRs (Value: Float) (op 142): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // (op 354) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(142) END
// Allocated : 16 VGPRs (Value: Float) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43
s_waitcnt vmcnt(3)
v_mul_f32 v28, v9, v12 // (op 146) 
v_mul_f32 v29, v9, v13 // (op 146) 
v_mul_f32 v30, v9, v14 // (op 146) 
v_mul_f32 v31, v9, v15 // (op 146) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v9, v16 // (op 146) 
v_mul_f32 v33, v9, v17 // (op 146) 
v_mul_f32 v34, v9, v18 // (op 146) 
v_mul_f32 v35, v9, v19 // (op 146) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v9, v20 // (op 146) 
v_mul_f32 v37, v9, v21 // (op 146) 
v_mul_f32 v38, v9, v22 // (op 146) 
v_mul_f32 v39, v9, v23 // (op 146) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v9, v24 // (op 146) 
v_mul_f32 v41, v9, v25 // (op 146) 
v_mul_f32 v42, v9, v26 // (op 146) 
v_mul_f32 v43, v9, v27 // (op 146) 
 // (op 354) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(419)NA)NA(146) END
 // (op 354) Deallocate{}(1297) BEGIN
 // (op 1297) Deallocate 419
 // Freeing : 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 419
 // (op 354) Deallocate{}(1297) END
 // (op 354) Assign VGPR Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA(143) BEGIN
 // (op 143) Assign dim(425) = Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA
 // (op 143) Generate Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA into nullptr
 // (op 143) reg expression
 // (op 143) reg expression
 // (op 143) Add({DataFlowTag421: v[28:43]:S}, {DataFlowTag423: v[44:59]:S})S
// Allocated : 16 VGPRs (Value: Float) (op 143): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
v_add_f32 v12, v28, v44 // (op 143) 
v_add_f32 v13, v29, v45 // (op 143) 
v_add_f32 v14, v30, v46 // (op 143) 
v_add_f32 v15, v31, v47 // (op 143) 
v_add_f32 v16, v32, v48 // (op 143) 
v_add_f32 v17, v33, v49 // (op 143) 
v_add_f32 v18, v34, v50 // (op 143) 
v_add_f32 v19, v35, v51 // (op 143) 
v_add_f32 v20, v36, v52 // (op 143) 
v_add_f32 v21, v37, v53 // (op 143) 
v_add_f32 v22, v38, v54 // (op 143) 
v_add_f32 v23, v39, v55 // (op 143) 
v_add_f32 v24, v40, v56 // (op 143) 
v_add_f32 v25, v41, v57 // (op 143) 
v_add_f32 v26, v42, v58 // (op 143) 
v_add_f32 v27, v43, v59 // (op 143) 
 // (op 354) Assign VGPR Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA(143) END
 // (op 354) Deallocate{}(1291) BEGIN
 // (op 1291) Deallocate 421
 // Freeing DataFlowTag421: 16 VGPRs (Value: Float) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 421
 // (op 1291) Deallocate 423
 // Freeing DataFlowTag423: 16 VGPRs (Value: Float) (op 142): v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59

 // Deleting tag 423
 // (op 354) Deallocate{}(1291) END
 // (op 354) NOP(1012) BEGIN
 // (op 1012) generate({})
 // (op 1012) end: generate({})
 // (op 354) NOP(1012) END
 // (op 354) NOP(1009) BEGIN
 // (op 1009) generate({})
 // (op 1009) end: generate({})
 // (op 354) NOP(1009) END
 // (op 354) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(1016) BEGIN
 // (op 1016) Assign dim(642) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 642: v**UNALLOCATED**
 // (op 1016) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag642: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1016) reg expression
 // (op 1016) reg expression
 // (op 1016) reg expression
 // (op 1016) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1016) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v0
v_lshrrev_b32 v0, 6, v1 // (op 1016) 
 // (op 1016) convert
 // (op 1016) BEGIN: Tile[1]
 // (op 1016) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v10
v_and_b32 v10, 63, v1 // (op 1016) 
 // (op 1016) END: Tile[1]
 // (op 1016) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v3
v_lshrrev_b32 v3, 5, v10 // (op 1016) 
 // (op 1016) BEGIN: Flatten
 // (op 1016) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 1016) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1016)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 1016)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v4
v_lshlrev_b32 v4, 2, v3 // (op 1016) 
 // (op 1016) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1016): v3

 // (op 1016) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 1016) 
 // (op 1016) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1016): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 1016): v0

 // (op 1016) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 1016) 
 // (op 1016) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1016): v3

 // (op 1016) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v3
v_lshrrev_b32 v3, 2, v10 // (op 1016) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1016): v10

 // (op 1016) BEGIN: Tile[1]
 // (op 1016) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v10
v_and_b32 v10, 7, v3 // (op 1016) 
 // (op 1016) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1016): v3

 // (op 1016) BEGIN: Tile[1]
 // (op 1016) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v3
v_and_b32 v3, 3, v1 // (op 1016) 
 // (op 1016) END: Tile[1]
 // (op 1016) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 1016) 
 // (op 1016) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1016): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1016): v10

 // (op 1016) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 1016) 
 // (op 1016) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 1016): v4

 // (op 1016) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 1016): v10
v_mul_lo_u32 v10, v3, s30 // (op 1016) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1016): v3

 // (op 1016) BEGIN: Join
// Allocated DataFlowTag642: 1 VGPR (Value: UInt32) (op 1016): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 1016) 
 // (op 1016) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 1016): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1016): v0

 // (op 354) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(1016) END
 // (op 354) Assign VGPR 32:U64(1017) BEGIN
 // (op 1017) Assign dim(643) = 32:U64
 // (op 1017) Assign stride expression
 // (op 354) Assign VGPR 32:U64(1017) END
 // (op 354) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(1018) BEGIN
 // (op 1018) Assign dim(639) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 1018) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // (op 1018) reg expression
 // (op 1018) reg expression
 // (op 1018) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 1018) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 1018)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 1018)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 1018): s14, s15
s_mul_i32 s15, s30, s13 // (op 1018) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 1018): s0
s_mul_i32 s0, s31, s12 // (op 1018) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1018) least significant
// Allocated : 1 SGPR (Value: Int32) (op 1018): s1
s_mul_hi_u32 s1, s30, s12 // (op 1018) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1018) most significant: sum
s_add_u32 s15, s15, s1 // (op 1018) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 1018): s1

 // Freeing : 1 SGPR (Value: Int32) (op 1018): s0

 // (op 1018) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 1018) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 1018)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 1018)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 1018): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 1018) 
 // Freeing : 2 SGPRs (Value: Int64) (op 1018): s14, s15

 // (op 1018) reg expression
 // (op 1018) reg expression
s_mov_b32 s8, s32 // (op 1018) reg expression
s_mov_b32 s9, s33 // (op 1018) reg expression
s_mov_b32 s10, s0 // (op 1018) reg expression
s_mov_b32 s11, 131072 // (op 1018) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 1018): s0, s1

 // (op 354) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(1018) END
 // (op 354) NOP(1010) BEGIN
 // (op 1010) generate({})
 // (op 1010) end: generate({})
 // (op 354) NOP(1010) END
 // (op 354) Assign VGPR 4:U64(1023) BEGIN
 // (op 1023) Assign dim(645) = 4:U64
 // (op 1023) Assign stride expression
 // (op 354) Assign VGPR 4:U64(1023) END
 // (op 354) StoreTiled Value: Float(144) BEGIN
 // (op 144) GEN: storeMacroTileWAVE OP 144 MacroTile 425 WaveTile 365
 // (op 144) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 144) Generate 32:U64 into nullptr
 // (op 144) Generate 4:U64 into nullptr
 // (op 144) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag425: VGPR Value: Float x 16: v[12:27]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
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
 // 	info.bufDesc = DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 144) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag425: VGPR Value: Float x 16: v[12:27]
 // 	info.rowOffsetReg = DataFlowTag642: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
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
 // 	info.bufDesc = DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 144) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx4 v[12:15], v3, s[8:11], 0 offen // (op 144) Store value
// VMEM: Expected complete at 7900 (current 7420)
buffer_store_dwordx4 v[16:19], v3, s[8:11], 0 offen offset:32 // (op 144) Store value
// VMEM: Expected complete at 7901 (current 7421)
buffer_store_dwordx4 v[20:23], v3, s[8:11], 0 offen offset:64 // (op 144) Store value
// VMEM: Expected complete at 7902 (current 7422)
buffer_store_dwordx4 v[24:27], v3, s[8:11], 0 offen offset:96 // (op 144) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 8381 (current 7901)
 // (op 354) StoreTiled Value: Float(144) END
 // (op 354) Deallocate{}(1293) BEGIN
 // (op 1293) Deallocate 644
 // Deleting tag 644
 // (op 1293) Deallocate 645
 // Deleting tag 645
 // (op 1293) Deallocate 642
 // Freeing DataFlowTag642: 1 VGPR (Value: UInt32) (op 1016): v3

 // Deleting tag 642
 // (op 1293) Deallocate 643
 // Deleting tag 643
 // (op 1293) Deallocate 425
 // Freeing DataFlowTag425: 16 VGPRs (Value: Float) (op 143): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 425
 // (op 354) Deallocate{}(1293) END
 // (op 354) end: generate({357})
 // (op 354) Unlock Scope 354
 // Unlock None
 // (op 133) Scope(354) END
 // (op 133) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1347) BEGIN
 // (op 1347) Deallocate 364
 // Deleting tag 364
 // (op 1347) Deallocate 639
 // Freeing DataFlowTag639: 4 SGPRs (Buffer: None) (op 1001): s8, s9, s10, s11

 // Deleting tag 639
 // (op 1347) Deallocate 441
 // Deleting tag 441
 // (op 1347) Deallocate 439
 // Deleting tag 439
 // (op 1347) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 1347) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 133) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1347) END
 // (op 133) end: generate({354})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 133) 
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133:
 // (op 133) 
 // (op 133) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(133) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1289) BEGIN
 // (op 1289) Deallocate 55
 // Freeing DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 63): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15

 // Deleting tag 55
 // (op 1289) Deallocate 53
 // Freeing : 1 VGPR (Value: Float) (op 43): v11

 // Deleting tag 53
 // (op 1289) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 2): v9

 // Deleting tag 44
 // (op 1289) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1289) END
 // (op 1) end: generate({823, 875, 928, 981, 1054, 1093, 1117, 1141})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 76
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
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
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [32, 32, 64, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  StaggerU: 0
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
    .vgpr_count: 60
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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

