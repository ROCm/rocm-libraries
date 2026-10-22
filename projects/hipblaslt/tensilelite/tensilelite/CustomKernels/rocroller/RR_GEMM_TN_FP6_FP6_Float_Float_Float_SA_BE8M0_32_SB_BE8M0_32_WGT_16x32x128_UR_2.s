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
.globl RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
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
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:
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
 // (op 1) generate({739, 792, 845, 903, 927, 951})
 // (op 1) NOP(739) BEGIN
 // (op 739) generate({})
 // (op 739) end: generate({})
 // (op 1) NOP(739) END
 // (op 1) NOP(792) BEGIN
 // (op 792) generate({})
 // (op 792) end: generate({})
 // (op 1) NOP(792) END
 // (op 1) NOP(845) BEGIN
 // (op 845) generate({})
 // (op 845) end: generate({})
 // (op 1) NOP(845) END
 // (op 1) NOP(903) BEGIN
 // (op 903) generate({})
 // (op 903) end: generate({})
 // (op 1) NOP(903) END
 // (op 1) NOP(927) BEGIN
 // (op 927) generate({})
 // (op 927) end: generate({})
 // (op 1) NOP(927) END
 // (op 1) NOP(951) BEGIN
 // (op 951) generate({})
 // (op 951) end: generate({})
 // (op 1) NOP(951) END
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(743) BEGIN
 // (op 743) Assign dim(544) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 544: v**UNALLOCATED**
 // (op 743) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag544: VGPR Value: UInt32 x 1: (unallocated)
 // (op 743) reg expression
 // (op 743) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(796) BEGIN
 // (op 796) Assign dim(560) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 560: v**UNALLOCATED**
 // (op 796) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 796) reg expression
 // (op 796) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}(849) BEGIN
 // (op 849) Assign dim(576) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 576: v**UNALLOCATED**
 // (op 849) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag576: VGPR Value: UInt32 x 1: (unallocated)
 // (op 849) reg expression
 // (op 849) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(907) BEGIN
 // (op 907) Assign dim(594) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 594: v**UNALLOCATED**
 // (op 907) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag594: VGPR Value: UInt32 x 1: (unallocated)
 // (op 907) reg expression
 // (op 907) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(931) BEGIN
 // (op 931) Assign dim(602) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 602: v**UNALLOCATED**
 // (op 931) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag602: VGPR Value: UInt32 x 1: (unallocated)
 // (op 931) reg expression
 // (op 931) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}(955) BEGIN
 // (op 955) Assign dim(610) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 610: v**UNALLOCATED**
 // (op 955) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag610: VGPR Value: UInt32 x 1: (unallocated)
 // (op 955) reg expression
 // (op 955) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 743): v0
v_lshrrev_b32 v0, 7, v1 // (op 743) 
 // (op 743) BEGIN: Flatten[1]Tile
 // (op 743) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 743): v3
v_and_b32 v3, 15, v1 // (op 743) 
 // (op 743) END: Flatten[1]Tile
 // (op 743) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 743): v4
v_lshl_add_u32 v4, v0, 4, v3 // (op 743) 
 // (op 743) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 743): v3

 // Freeing : 1 VGPR (Value: UInt32) (op 743): v0

 // (op 743) BEGIN: Flatten[1]
 // (op 743) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 743): v0
v_and_b32 v0, 63, v1 // (op 743) 
 // (op 743) END: Flatten[1]
 // (op 743) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 743): v3
v_lshrrev_b32 v3, 4, v0 // (op 743) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 743): v0

 // (op 743) BEGIN: TileTile
 // (op 743) {TileTile: ShiftL(v3:U32, 5:U32)U32}
 // (op 743) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 743)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 743)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 743): v0
v_lshlrev_b32 v0, 5, v3 // (op 743) 
 // (op 743) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 743): v3

 // (op 743) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 743): v3
v_lshl_add_u32 v3, v4, 7, v0 // (op 743) 
 // (op 743) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 743): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 743): v4

 // (op 743) Multiply({Tile: v3:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 743): v0
v_mul_lo_u32 v0, v3, 6 // (op 743) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 743): v3

 // (op 743) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag544: 1 VGPR (Value: UInt32) (op 743): v3
v_lshrrev_b32 v3, 3, v0 // (op 743) 
 // Freeing : 1 VGPR (Value: UInt32) (op 743): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(743) END
// Allocated : 1 VGPR (Value: UInt32) (op 796): v0
v_lshrrev_b32 v0, 6, v1 // (op 796) 
 // (op 796) BEGIN: Tile
 // (op 796) {Tile: BitwiseAnd(v0:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 796): v4
v_and_b32 v4, 1, v0 // (op 796) 
 // (op 796) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 796): v0

 // (op 796) BEGIN: Flatten[1]Tile
 // (op 796) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 796): v0
v_and_b32 v0, 15, v1 // (op 796) 
 // (op 796) END: Flatten[1]Tile
 // (op 796) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 796): v5
v_lshl_add_u32 v5, v4, 4, v0 // (op 796) 
 // (op 796) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 796): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 796): v4

 // (op 796) BEGIN: Flatten[1]
 // (op 796) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 796): v0
v_and_b32 v0, 63, v1 // (op 796) 
 // (op 796) END: Flatten[1]
 // (op 796) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 796): v4
v_lshrrev_b32 v4, 4, v0 // (op 796) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 796): v0

 // (op 796) BEGIN: TileTile
 // (op 796) {TileTile: ShiftL(v4:U32, 5:U32)U32}
 // (op 796) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 796)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 796)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 796): v0
v_lshlrev_b32 v0, 5, v4 // (op 796) 
 // (op 796) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 796): v4

 // (op 796) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 796): v4
v_lshl_add_u32 v4, v5, 7, v0 // (op 796) 
 // (op 796) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 796): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 796): v5

 // (op 796) Multiply({Tile: v4:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 796): v0
v_mul_lo_u32 v0, v4, 6 // (op 796) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 796): v4

 // (op 796) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 796): v4
v_lshrrev_b32 v4, 3, v0 // (op 796) 
 // Freeing : 1 VGPR (Value: UInt32) (op 796): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(796) END
// Allocated : 1 VGPR (Value: UInt32) (op 849): v0
v_lshrrev_b32 v0, 6, v1 // (op 849) 
 // (op 849) BEGIN: Tile
 // (op 849) {Tile: BitwiseAnd(v0:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 849): v5
v_and_b32 v5, 1, v0 // (op 849) 
 // (op 849) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 849): v0

 // (op 849) BEGIN: Flatten[1]
 // (op 849) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 849): v0
v_and_b32 v0, 15, v1 // (op 849) 
 // (op 849) END: Flatten[1]
 // (op 849) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 849): v6
v_lshl_add_u32 v6, v5, 4, v0 // (op 849) 
 // (op 849) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 849): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 849): v5

 // (op 849) BEGIN: Flatten[1]
 // (op 849) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 849): v0
v_and_b32 v0, 63, v1 // (op 849) 
 // (op 849) END: Flatten[1]
 // (op 849) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 849): v5
v_lshrrev_b32 v5, 4, v0 // (op 849) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 849): v0

 // (op 849) BEGIN: Tile
// Allocated DataFlowTag576: 1 VGPR (Value: UInt32) (op 849): v0
v_lshl_add_u32 v0, v6, 2, v5 // (op 849) 
 // (op 849) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 849): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 849): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}(849) END
// Allocated : 1 VGPR (Value: UInt32) (op 907): v5
v_lshrrev_b32 v5, 7, v1 // (op 907) 
 // (op 907) BEGIN: Flatten[1]Tile
 // (op 907) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 907): v6
v_and_b32 v6, 15, v1 // (op 907) 
 // (op 907) END: Flatten[1]Tile
 // (op 907) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 907): v7
v_lshl_add_u32 v7, v5, 4, v6 // (op 907) 
 // (op 907) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 907): v6

 // Freeing : 1 VGPR (Value: UInt32) (op 907): v5

 // (op 907) BEGIN: Flatten[1]
 // (op 907) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 907): v5
v_and_b32 v5, 63, v1 // (op 907) 
 // (op 907) END: Flatten[1]
 // (op 907) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 907): v6
v_lshrrev_b32 v6, 4, v5 // (op 907) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 907): v5

 // (op 907) BEGIN: TileTile
 // (op 907) {TileTile: ShiftL(v6:U32, 5:U32)U32}
 // (op 907) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 907)         value (VGPR Value: UInt32 x 1: v6) 
 // (op 907)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 907): v5
v_lshlrev_b32 v5, 5, v6 // (op 907) 
 // (op 907) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 907): v6

 // (op 907) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 907): v6
v_lshl_add_u32 v6, v7, 7, v5 // (op 907) 
 // (op 907) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 907): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 907): v7

 // (op 907) Multiply({Tile: v6:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 907): v5
v_mul_lo_u32 v5, v6, 6 // (op 907) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 907): v6

 // (op 907) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag594: 1 VGPR (Value: UInt32) (op 907): v6
v_lshrrev_b32 v6, 3, v5 // (op 907) 
 // Freeing : 1 VGPR (Value: UInt32) (op 907): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(907) END
// Allocated : 1 VGPR (Value: UInt32) (op 931): v5
v_lshrrev_b32 v5, 6, v1 // (op 931) 
 // (op 931) BEGIN: Tile
 // (op 931) {Tile: BitwiseAnd(v5:U32, 1:U32)U32}
 // (op 1) Deallocate{}(1145) BEGIN
 // (op 1145) Deallocate 98
 // Deleting tag 98
 // (op 1) Deallocate{}(1145) END
// Allocated : 1 VGPR (Value: UInt32) (op 931): v7
v_and_b32 v7, 1, v5 // (op 931) 
 // (op 931) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 931): v5

 // (op 931) BEGIN: Flatten[1]Tile
 // (op 931) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Assign VGPR 96:U32(744) BEGIN
 // (op 744) Assign dim(545) = 96:U32
 // (op 744) Assign stride expression
 // (op 1) Assign VGPR 96:U32(744) END
 // (op 1) Assign VGPR 96:U32(908) BEGIN
 // (op 908) Assign dim(595) = 96:U32
 // (op 908) Assign stride expression
 // (op 1) Assign VGPR 96:U32(908) END
 // (op 1) NOP(740) BEGIN
 // (op 740) generate({})
 // (op 740) end: generate({})
 // (op 1) NOP(740) END
 // (op 1) NOP(904) BEGIN
 // (op 904) generate({})
 // (op 904) end: generate({})
 // (op 1) NOP(904) END
 // (op 1) Assign VGPR 0:U32(748) BEGIN
 // (op 748) Assign dim(547) = 0:U32
 // (op 748) Assign stride expression
 // (op 1) Assign VGPR 0:U32(748) END
 // (op 1) Assign VGPR 0:U32(912) BEGIN
 // (op 912) Assign dim(597) = 0:U32
 // (op 912) Assign stride expression
 // (op 1) Assign VGPR 0:U32(912) END
// Allocated : 1 VGPR (Value: UInt32) (op 931): v5
v_and_b32 v5, 15, v1 // (op 931) 
 // (op 931) END: Flatten[1]Tile
 // (op 931) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 931): v8
v_lshl_add_u32 v8, v7, 4, v5 // (op 931) 
 // (op 931) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 931): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 931): v7

 // (op 931) BEGIN: Flatten[1]
 // (op 931) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 931): v5
v_and_b32 v5, 63, v1 // (op 931) 
 // (op 931) END: Flatten[1]
 // (op 931) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 931): v7
v_lshrrev_b32 v7, 4, v5 // (op 931) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 931): v5

 // (op 931) BEGIN: TileTile
 // (op 931) {TileTile: ShiftL(v7:U32, 5:U32)U32}
 // (op 931) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 931)         value (VGPR Value: UInt32 x 1: v7) 
 // (op 931)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 931): v5
v_lshlrev_b32 v5, 5, v7 // (op 931) 
 // (op 931) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 931): v7

 // (op 931) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 931): v7
v_lshl_add_u32 v7, v8, 7, v5 // (op 931) 
 // (op 931) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 931): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 931): v8

 // (op 931) Multiply({Tile: v7:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 931): v5
v_mul_lo_u32 v5, v7, 6 // (op 931) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 931): v7

 // (op 931) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag602: 1 VGPR (Value: UInt32) (op 931): v7
v_lshrrev_b32 v7, 3, v5 // (op 931) 
 // Freeing : 1 VGPR (Value: UInt32) (op 931): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(931) END
// Allocated : 1 VGPR (Value: UInt32) (op 955): v5
v_lshrrev_b32 v5, 6, v1 // (op 955) 
 // (op 955) BEGIN: Tile
 // (op 955) {Tile: BitwiseAnd(v5:U32, 1:U32)U32}
 // (op 1) Deallocate{}(1148) BEGIN
 // (op 1148) Deallocate 164
 // Deleting tag 164
 // (op 1) Deallocate{}(1148) END
// Allocated : 1 VGPR (Value: UInt32) (op 955): v8
v_and_b32 v8, 1, v5 // (op 955) 
 // (op 955) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 955): v5

 // (op 955) BEGIN: Flatten[1]
 // (op 955) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Assign VGPR 96:U32(797) BEGIN
 // (op 797) Assign dim(561) = 96:U32
 // (op 797) Assign stride expression
 // (op 1) Assign VGPR 96:U32(797) END
 // (op 1) Assign VGPR 96:U32(932) BEGIN
 // (op 932) Assign dim(603) = 96:U32
 // (op 932) Assign stride expression
 // (op 1) Assign VGPR 96:U32(932) END
 // (op 1) NOP(793) BEGIN
 // (op 793) generate({})
 // (op 793) end: generate({})
 // (op 1) NOP(793) END
 // (op 1) NOP(928) BEGIN
 // (op 928) generate({})
 // (op 928) end: generate({})
 // (op 1) NOP(928) END
 // (op 1) Assign VGPR 0:U32(801) BEGIN
 // (op 801) Assign dim(563) = 0:U32
 // (op 801) Assign stride expression
 // (op 1) Assign VGPR 0:U32(801) END
 // (op 1) Assign VGPR 0:U32(936) BEGIN
 // (op 936) Assign dim(605) = 0:U32
 // (op 936) Assign stride expression
 // (op 1) Assign VGPR 0:U32(936) END
// Allocated : 1 VGPR (Value: UInt32) (op 955): v5
v_and_b32 v5, 15, v1 // (op 955) 
 // (op 955) END: Flatten[1]
 // (op 955) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 955): v9
v_lshl_add_u32 v9, v8, 4, v5 // (op 955) 
 // (op 955) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 955): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 955): v8

 // (op 955) BEGIN: Flatten[1]
 // (op 955) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 955): v5
v_and_b32 v5, 63, v1 // (op 955) 
 // (op 955) END: Flatten[1]
 // (op 955) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 955): v8
v_lshrrev_b32 v8, 4, v5 // (op 955) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 955): v5

 // (op 955) BEGIN: Tile
// Allocated DataFlowTag610: 1 VGPR (Value: UInt32) (op 955): v5
v_lshl_add_u32 v5, v9, 2, v8 // (op 955) 
 // (op 955) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 955): v8

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 955): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}(955) END
 // (op 1) Deallocate{}(1151) BEGIN
 // (op 1151) Deallocate 260
 // Deleting tag 260
 // (op 1) Deallocate{}(1151) END
 // (op 1) Assign VGPR 4:U32(850) BEGIN
 // (op 850) Assign dim(577) = 4:U32
 // (op 850) Assign stride expression
 // (op 1) Assign VGPR 4:U32(850) END
 // (op 1) Assign VGPR 4:U32(956) BEGIN
 // (op 956) Assign dim(611) = 4:U32
 // (op 956) Assign stride expression
 // (op 1) Assign VGPR 4:U32(956) END
 // (op 1) NOP(846) BEGIN
 // (op 846) generate({})
 // (op 846) end: generate({})
 // (op 1) NOP(846) END
 // (op 1) NOP(952) BEGIN
 // (op 952) generate({})
 // (op 952) end: generate({})
 // (op 1) NOP(952) END
 // (op 1) Assign VGPR 1:U32(854) BEGIN
 // (op 854) Assign dim(579) = 1:U32
 // (op 854) Assign stride expression
 // (op 1) Assign VGPR 1:U32(854) END
 // (op 1) Assign VGPR 1:U32(960) BEGIN
 // (op 960) Assign dim(613) = 1:U32
 // (op 960) Assign stride expression
 // (op 1) Assign VGPR 1:U32(960) END
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
 // (op 1) Deallocate{user_Float_Value_14}(1196) BEGIN
 // (op 1196) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1196) END
 // (op 1) Deallocate{user_Float_Value_12}(1198) BEGIN
 // (op 1198) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1198) END
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
 // (op 310) Scope(445) BEGIN
 // (op 445) Lock Scope 445
 // Lock Branch
 // (op 445) generate({440})
 // (op 445) NOP(440) BEGIN
 // (op 440) generate({})
 // (op 440) end: generate({})
 // (op 445) NOP(440) END
 // (op 445) Assign VGPR 96:U64(450) BEGIN
 // (op 450) Assign dim(458) = 96:U64
 // (op 450) Assign stride expression
 // (op 445) Assign VGPR 96:U64(450) END
 // (op 445) NOP(441) BEGIN
 // (op 441) generate({})
 // (op 441) end: generate({})
 // (op 445) NOP(441) END
 // (op 445) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(453) BEGIN
 // (op 453) Assign dim(459) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 459: v**UNALLOCATED**
 // (op 453) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag459: VGPR Value: UInt32 x 1: (unallocated)
 // (op 453) reg expression
 // (op 453) reg expression
 // (op 453) reg expression
 // (op 453) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 453): v10
v_lshrrev_b32 v10, 3, v1 // (op 453) 
 // (op 453) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 453): v11
v_lshl_add_u32 v11, s16, 4, v10 // (op 453) 
 // (op 453) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 453): v10

 // (op 453) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 453): v10
v_and_b32 v10, -8, v1 // (op 453) 
 // (op 453) BEGIN: Flatten[1]
 // (op 453) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 453): v12
v_sub_u32 v12, v1, v10 // (op 453) 
 // (op 453) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 453): v10

 // (op 453) BEGIN: TileTile
 // (op 453) {TileTile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 453) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 453)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 453)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 453): v10
v_lshlrev_b32 v10, 4, v12 // (op 453) 
 // (op 453) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 453): v12

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 453): v12, v13
v_mov_b32 v13, 0 // (op 453) convert
v_mov_b32 v12, v10 // (op 453) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 453): v10

 // (op 453) BEGIN: Split
 // (op 453) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 453)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 453)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 453): v14, v15
v_mul_lo_u32 v15, v11, s3 // (op 453) most significant: low of low * high
 // (op 453) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 453): v10
v_mul_hi_u32 v10, v11, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 453) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 453) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 453): v10

 // (op 453) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 453)      lhs (VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 453)      rhs (TileTile: VGPR Value: UInt64 x 1: v[12:13])
// Allocated : 2 SGPRs (Value: Bool64) (op 453): s0, s1
v_add_co_u32 v14, s[0:1], v14, v12 // (op 453) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 453) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 453): s0, s1

 // (op 453) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 453): v12, v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 453): v11

 // (op 453) Multiply({Split: v[14:15]:U64}, 6:U32)U64
 // (op 453) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 453)           lhs (Split: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 453)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 453) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 453): v10, v11
v_mul_lo_u32 v11, v15, 6 // (op 453) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 453): v12
v_mul_hi_u32 v12, v14, 6 // (op 453) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 453) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 453) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 453): v12

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 453): v14, v15

 // (op 453) LogicalShiftR(v[10:11]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 453): v12, v13
v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 453) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 453): v10, v11

// Allocated DataFlowTag459: 1 VGPR (Value: UInt32) (op 453): v10
v_mov_b32 v10, v12 // (op 453) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 453): v12, v13

 // (op 445) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(453) END
 // (op 445) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(454) BEGIN
 // (op 454) Assign dim(460) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 454) Assign stride expression
 // (op 445) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(454) END
 // (op 445) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(455) BEGIN
 // (op 455) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 453: s**UNALLOCATED**
 // (op 455) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: (unallocated)
 // (op 455) reg expression
 // (op 455) reg expression
 // (op 455) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 455) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 455)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 455)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 455) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 455): s0, s1
s_mul_i32 s1, s7, 6 // (op 455) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 455) least significant
// Allocated : 1 SGPR (Value: Int32) (op 455): s14
s_mul_hi_u32 s14, s6, 6 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 455) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 455): s14

 // (op 455) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 455) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 455)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 455)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 455): s14, s15
s_mul_i32 s15, s2, s1 // (op 455) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 455): s18
s_mul_i32 s18, s3, s0 // (op 455) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 455) least significant
// Allocated : 1 SGPR (Value: Int32) (op 455): s19
s_mul_hi_u32 s19, s2, s0 // (op 455) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 455) most significant: sum
s_add_u32 s15, s15, s19 // (op 455) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 455): s19

 // Freeing : 1 SGPR (Value: Int32) (op 455): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 455): s0, s1

 // (op 455) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 455): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 455) 
 // Freeing : 2 SGPRs (Value: Int64) (op 455): s14, s15

 // (op 455) reg expression
 // (op 455) reg expression
// Allocated DataFlowTag453: 4 SGPRs (Buffer: None) (op 455): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 455) reg expression
s_mov_b32 s41, s5 // (op 455) reg expression
s_mov_b32 s42, s0 // (op 455) reg expression
s_mov_b32 s43, 131072 // (op 455) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 455): s0, s1

 // (op 445) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(455) END
 // (op 445) NOP(442) BEGIN
 // (op 442) generate({})
 // (op 442) end: generate({})
 // (op 445) NOP(442) END
 // (op 445) Assign VGPR 0:U64(460) BEGIN
 // (op 460) Assign dim(462) = 0:U64
 // (op 460) Assign stride expression
 // (op 445) Assign VGPR 0:U64(460) END
 // (op 445) NOP(463) BEGIN
 // (op 463) generate({})
 // (op 463) end: generate({})
 // (op 445) NOP(463) END
 // (op 445) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(472) BEGIN
 // (op 472) Assign dim(463) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 463: v**UNALLOCATED**
 // (op 472) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag463: VGPR Value: UInt32 x 1: (unallocated)
 // (op 472) reg expression
 // (op 472) reg expression
 // (op 472) reg expression
 // (op 472) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 472): v11
v_lshrrev_b32 v11, 3, v1 // (op 472) 
 // (op 472) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 472): v12
v_lshl_add_u32 v12, s16, 4, v11 // (op 472) 
 // (op 472) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 472): v11

 // (op 472) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 472): v11
v_and_b32 v11, -8, v1 // (op 472) 
 // (op 472) BEGIN: Flatten[1]
 // (op 472) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 472): v13
v_sub_u32 v13, v1, v11 // (op 472) 
 // (op 472) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 472): v11

 // (op 472) BEGIN: TileTile
 // (op 472) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 472) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 472)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 472)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 472): v11
v_lshlrev_b32 v11, 4, v13 // (op 472) 
 // (op 472) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 472): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 472): v14, v15
v_mov_b32 v15, 0 // (op 472) convert
v_mov_b32 v14, v11 // (op 472) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 472): v11

 // (op 472) BEGIN: Split
 // (op 472) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 472)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 472)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 472): v16, v17
v_mul_lo_u32 v17, v12, s3 // (op 472) most significant: low of low * high
 // (op 472) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 472): v11
v_mul_hi_u32 v11, v12, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 472) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 472) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 472): v11

 // (op 472) Add: dest (VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 472)      lhs (VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 472)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 472): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 472) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 472) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 472): s0, s1

 // (op 472) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 472): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 472): v12

 // (op 472) Multiply({Split: v[16:17]:U64}, 6:U32)U64
 // (op 472) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 472)           lhs (Split: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 472)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 472) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 472): v14, v15
v_mul_lo_u32 v15, v17, 6 // (op 472) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 472): v11
v_mul_hi_u32 v11, v16, 6 // (op 472) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 472) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 472) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 472): v11

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 472): v16, v17

 // (op 472) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 472): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 472) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 472): v14, v15

// Allocated DataFlowTag463: 1 VGPR (Value: UInt32) (op 472): v11
v_mov_b32 v11, v12 // (op 472) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 472): v12, v13

 // (op 445) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(472) END
 // (op 445) Assign VGPR 96:U64(473) BEGIN
 // (op 473) Assign dim(464) = 96:U64
 // (op 473) Assign stride expression
 // (op 445) Assign VGPR 96:U64(473) END
 // (op 445) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(474) BEGIN
 // (op 474) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 474) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // (op 474) reg expression
 // (op 474) reg expression
 // (op 474) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 474) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 474)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 474)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 474) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 474): s0, s1
s_mul_i32 s1, s7, 6 // (op 474) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 474) least significant
// Allocated : 1 SGPR (Value: Int32) (op 474): s14
s_mul_hi_u32 s14, s6, 6 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 474) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 474): s14

 // (op 474) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 474) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 474)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 474)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 474): s14, s15
s_mul_i32 s15, s2, s1 // (op 474) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 474): s18
s_mul_i32 s18, s3, s0 // (op 474) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 474) least significant
// Allocated : 1 SGPR (Value: Int32) (op 474): s19
s_mul_hi_u32 s19, s2, s0 // (op 474) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 474) most significant: sum
s_add_u32 s15, s15, s19 // (op 474) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 474): s19

 // Freeing : 1 SGPR (Value: Int32) (op 474): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 474): s0, s1

 // (op 474) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 474): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 474) 
 // Freeing : 2 SGPRs (Value: Int64) (op 474): s14, s15

 // (op 474) reg expression
 // (op 474) reg expression
s_mov_b32 s40, s4 // (op 474) reg expression
s_mov_b32 s41, s5 // (op 474) reg expression
s_mov_b32 s42, s0 // (op 474) reg expression
s_mov_b32 s43, 131072 // (op 474) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 474): s0, s1

 // (op 445) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(474) END
 // (op 445) NOP(465) BEGIN
 // (op 465) generate({})
 // (op 465) end: generate({})
 // (op 445) NOP(465) END
 // (op 445) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(479) BEGIN
 // (op 479) Assign dim(466) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 479) Assign stride expression
 // (op 445) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(479) END
 // (op 445) NOP(466) BEGIN
 // (op 466) generate({})
 // (op 466) end: generate({})
 // (op 445) NOP(466) END
 // (op 445) Assign VGPR 0:U64(482) BEGIN
 // (op 482) Assign dim(468) = 0:U64
 // (op 482) Assign stride expression
 // (op 445) Assign VGPR 0:U64(482) END
 // (op 445) NOP(507) BEGIN
 // (op 507) generate({})
 // (op 507) end: generate({})
 // (op 445) NOP(507) END
 // (op 445) Assign VGPR 96:U64(514) BEGIN
 // (op 514) Assign dim(477) = 96:U64
 // (op 514) Assign stride expression
 // (op 445) Assign VGPR 96:U64(514) END
 // (op 445) NOP(508) BEGIN
 // (op 508) generate({})
 // (op 508) end: generate({})
 // (op 445) NOP(508) END
 // (op 445) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(517) BEGIN
 // (op 517) Assign dim(478) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 478: v**UNALLOCATED**
 // (op 517) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag478: VGPR Value: UInt32 x 1: (unallocated)
 // (op 517) reg expression
 // (op 517) reg expression
 // (op 517) reg expression
 // (op 517) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 517): v12
v_and_b32 v12, -4, v1 // (op 517) 
 // (op 517) BEGIN: Flatten[1]
 // (op 517) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 517): v13
v_sub_u32 v13, v1, v12 // (op 517) 
 // (op 517) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 517): v12

 // (op 517) BEGIN: TileTile
 // (op 517) {TileTile: ShiftL({Flatten[1]: v13:U32}, 5:U32)U32}
 // (op 517) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 517)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 517)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 517): v12
v_lshlrev_b32 v12, 5, v13 // (op 517) 
 // (op 517) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 517): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 517): v14, v15
v_mov_b32 v15, 0 // (op 517) convert
v_mov_b32 v14, v12 // (op 517) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 517): v12

 // (op 517) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 517): v12
v_lshrrev_b32 v12, 2, v1 // (op 517) 
 // (op 517) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 517): v13
v_lshl_add_u32 v13, s17, 5, v12 // (op 517) 
 // (op 517) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 517): v12

 // (op 517) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 517) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 517)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 517)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 517): v16, v17
v_mul_lo_u32 v17, v13, s9 // (op 517) most significant: low of low * high
 // (op 517) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 517): v12
v_mul_hi_u32 v12, v13, s8 // (op 517) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 517) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 517) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 517): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 517): v13

 // (op 517) BEGIN: Split
 // (op 517) {Split: Add({TileTile: v[14:15]:U64}, v[16:17]:I64)U64}
 // (op 517) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 517)      lhs (TileTile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 517)      rhs (VGPR Value: Int64 x 1: v[16:17])
// Allocated : 2 VGPRs (Value: UInt64) (op 517): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 517): s0, s1
v_add_co_u32 v12, s[0:1], v14, v16 // (op 517) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 517) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 517): s0, s1

 // (op 517) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 517): v16, v17

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 517): v14, v15

 // (op 517) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 517) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 517)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 517)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 517) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 517): v14, v15
v_mul_lo_u32 v15, v13, 6 // (op 517) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 517): v16
v_mul_hi_u32 v16, v12, 6 // (op 517) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 517) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 517) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 517): v16

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 517): v12, v13

 // (op 517) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 517): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 517) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 517): v14, v15

// Allocated DataFlowTag478: 1 VGPR (Value: UInt32) (op 517): v14
v_mov_b32 v14, v12 // (op 517) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 517): v12, v13

 // (op 445) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(517) END
 // (op 445) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(518) BEGIN
 // (op 518) Assign dim(479) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 518) Assign stride expression
 // (op 445) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(518) END
 // (op 445) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(519) BEGIN
 // (op 519) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 472: s**UNALLOCATED**
 // (op 519) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: (unallocated)
 // (op 519) reg expression
 // (op 519) reg expression
 // (op 519) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 519) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 519)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 519)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 519) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 519): s0, s1
s_mul_i32 s1, s13, 6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 519) least significant
// Allocated : 1 SGPR (Value: Int32) (op 519): s14
s_mul_hi_u32 s14, s12, 6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 519) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 519): s14

 // (op 519) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 519) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 519)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 519)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 519): s14, s15
s_mul_i32 s15, s8, s1 // (op 519) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 519): s18
s_mul_i32 s18, s9, s0 // (op 519) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 519) least significant
// Allocated : 1 SGPR (Value: Int32) (op 519): s19
s_mul_hi_u32 s19, s8, s0 // (op 519) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 519) most significant: sum
s_add_u32 s15, s15, s19 // (op 519) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 519): s19

 // Freeing : 1 SGPR (Value: Int32) (op 519): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 519): s0, s1

 // (op 519) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 519): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 519) 
 // Freeing : 2 SGPRs (Value: Int64) (op 519): s14, s15

 // (op 519) reg expression
 // (op 519) reg expression
// Allocated DataFlowTag472: 4 SGPRs (Buffer: None) (op 519): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 519) reg expression
s_mov_b32 s45, s11 // (op 519) reg expression
s_mov_b32 s46, s0 // (op 519) reg expression
s_mov_b32 s47, 131072 // (op 519) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 519): s0, s1

 // (op 445) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(519) END
 // (op 445) NOP(509) BEGIN
 // (op 509) generate({})
 // (op 509) end: generate({})
 // (op 445) NOP(509) END
 // (op 445) Assign VGPR 0:U64(524) BEGIN
 // (op 524) Assign dim(481) = 0:U64
 // (op 524) Assign stride expression
 // (op 445) Assign VGPR 0:U64(524) END
 // (op 445) NOP(527) BEGIN
 // (op 527) generate({})
 // (op 527) end: generate({})
 // (op 445) NOP(527) END
 // (op 445) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(536) BEGIN
 // (op 536) Assign dim(482) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 482: v**UNALLOCATED**
 // (op 536) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag482: VGPR Value: UInt32 x 1: (unallocated)
 // (op 536) reg expression
 // (op 536) reg expression
 // (op 536) reg expression
 // (op 536) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 536): v12
v_and_b32 v12, -4, v1 // (op 536) 
 // (op 536) BEGIN: Flatten[1]
 // (op 536) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 536): v13
v_sub_u32 v13, v1, v12 // (op 536) 
 // (op 536) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 536): v12

 // (op 536) BEGIN: TileTile
 // (op 536) {TileTile: ShiftL({Flatten[1]: v13:U32}, 5:U32)U32}
 // (op 536) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 536)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 536)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 536): v12
v_lshlrev_b32 v12, 5, v13 // (op 536) 
 // (op 536) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 536): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 536): v16, v17
v_mov_b32 v17, 0 // (op 536) convert
v_mov_b32 v16, v12 // (op 536) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 536): v12

 // (op 536) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 536): v15
v_lshrrev_b32 v15, 2, v1 // (op 536) 
 // (op 536) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 536): v12
v_lshl_add_u32 v12, s17, 5, v15 // (op 536) 
 // (op 536) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 536): v15

 // (op 536) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 536) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 536)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 536)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 536): v18, v19
v_mul_lo_u32 v19, v12, s9 // (op 536) most significant: low of low * high
 // (op 536) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 536): v13
v_mul_hi_u32 v13, v12, s8 // (op 536) most significant: high of low * low
v_mul_lo_u32 v18, v12, s8 // (op 536) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 536) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 536): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 536): v12

 // (op 536) BEGIN: Split
 // (op 536) {Split: Add({TileTile: v[16:17]:U64}, v[18:19]:I64)U64}
 // (op 536) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 536)      lhs (TileTile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 536)      rhs (VGPR Value: Int64 x 1: v[18:19])
// Allocated : 2 VGPRs (Value: UInt64) (op 536): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 536): s0, s1
v_add_co_u32 v12, s[0:1], v16, v18 // (op 536) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 536) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 536): s0, s1

 // (op 536) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 536): v18, v19

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 536): v16, v17

 // (op 536) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 536) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 536)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 536)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 536) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 536): v16, v17
v_mul_lo_u32 v17, v13, 6 // (op 536) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 536): v15
v_mul_hi_u32 v15, v12, 6 // (op 536) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 536) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 536) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 536): v15

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 536): v12, v13

 // (op 536) LogicalShiftR(v[16:17]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 536): v12, v13
v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 536) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 536): v16, v17

// Allocated DataFlowTag482: 1 VGPR (Value: UInt32) (op 536): v15
v_mov_b32 v15, v12 // (op 536) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 536): v12, v13

 // (op 445) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(536) END
 // (op 445) Assign VGPR 96:U64(537) BEGIN
 // (op 537) Assign dim(483) = 96:U64
 // (op 537) Assign stride expression
 // (op 445) Assign VGPR 96:U64(537) END
 // (op 445) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(538) BEGIN
 // (op 538) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 538) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // (op 538) reg expression
 // (op 538) reg expression
 // (op 538) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 538) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 538)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 538)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 538) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 538): s0, s1
s_mul_i32 s1, s13, 6 // (op 538) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 538) least significant
// Allocated : 1 SGPR (Value: Int32) (op 538): s14
s_mul_hi_u32 s14, s12, 6 // (op 538) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 538) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 538): s14

 // (op 538) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 538) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 538)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 538)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 538): s14, s15
s_mul_i32 s15, s8, s1 // (op 538) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 538): s18
s_mul_i32 s18, s9, s0 // (op 538) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 538) least significant
// Allocated : 1 SGPR (Value: Int32) (op 538): s19
s_mul_hi_u32 s19, s8, s0 // (op 538) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 538) most significant: sum
s_add_u32 s15, s15, s19 // (op 538) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 538): s19

 // Freeing : 1 SGPR (Value: Int32) (op 538): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 538): s0, s1

 // (op 538) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 538): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 538) 
 // Freeing : 2 SGPRs (Value: Int64) (op 538): s14, s15

 // (op 538) reg expression
 // (op 538) reg expression
s_mov_b32 s44, s10 // (op 538) reg expression
s_mov_b32 s45, s11 // (op 538) reg expression
s_mov_b32 s46, s0 // (op 538) reg expression
s_mov_b32 s47, 131072 // (op 538) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 538): s0, s1

 // (op 445) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(538) END
 // (op 445) NOP(529) BEGIN
 // (op 529) generate({})
 // (op 529) end: generate({})
 // (op 445) NOP(529) END
 // (op 445) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(543) BEGIN
 // (op 543) Assign dim(485) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 543) Assign stride expression
 // (op 445) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(543) END
 // (op 445) NOP(530) BEGIN
 // (op 530) generate({})
 // (op 530) end: generate({})
 // (op 445) NOP(530) END
 // (op 445) Assign VGPR 0:U64(546) BEGIN
 // (op 546) Assign dim(487) = 0:U64
 // (op 546) Assign stride expression
 // (op 445) Assign VGPR 0:U64(546) END
 // (op 445) NOP(571) BEGIN
 // (op 571) generate({})
 // (op 571) end: generate({})
 // (op 445) NOP(571) END
 // (op 445) Assign VGPR 4:U64(578) BEGIN
 // (op 578) Assign dim(496) = 4:U64
 // (op 578) Assign stride expression
 // (op 445) Assign VGPR 4:U64(578) END
 // (op 445) NOP(572) BEGIN
 // (op 572) generate({})
 // (op 572) end: generate({})
 // (op 445) NOP(572) END
 // (op 445) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}(581) BEGIN
 // (op 581) Assign dim(497) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 497: v**UNALLOCATED**
 // (op 581) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag497: VGPR Value: UInt32 x 1: (unallocated)
 // (op 581) reg expression
 // (op 581) reg expression
 // (op 581) reg expression
 // (op 581) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 581) convert
// Allocated : 1 VGPR (Value: UInt32) (op 581): v12
v_lshrrev_b32 v12, 7, v1 // (op 581) 
 // (op 581) BEGIN: Flatten[1]
 // (op 581) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 581): v13
v_and_b32 v13, 15, v1 // (op 581) 
 // (op 581) END: Flatten[1]
 // (op 581) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 581): v16
v_lshl_add_u32 v16, v12, 4, v13 // (op 581) 
 // (op 581) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 581): v13

 // Freeing : 1 VGPR (Value: UInt32) (op 581): v12

 // (op 581) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 581): v12
v_lshl_add_u32 v12, s16, 4, v16 // (op 581) 
 // (op 581) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 581): v16

 // (op 581) BEGIN: Flatten[1]
 // (op 581) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 581): v13
v_and_b32 v13, 63, v1 // (op 581) 
 // (op 581) END: Flatten[1]
 // (op 581) LogicalShiftR({Flatten[1]: v13:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 581): v16
v_lshrrev_b32 v16, 4, v13 // (op 581) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 581): v13

 // (op 581) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag497: 1 VGPR (Value: UInt32) (op 581): v13
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v13, v12, s20 // (op 581) 
v_add_u32 v13, v13, v16 // (op 581) 
 // (op 581) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 581): v16

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 581): v12

 // (op 445) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}(581) END
 // (op 445) Assign VGPR 4:U64(582) BEGIN
 // (op 582) Assign dim(498) = 4:U64
 // (op 582) Assign stride expression
 // (op 445) Assign VGPR 4:U64(582) END
 // (op 445) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(583) BEGIN
 // (op 583) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 491: s**UNALLOCATED**
 // (op 583) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: (unallocated)
 // (op 583) reg expression
 // (op 583) reg expression
 // (op 583) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 583) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 583)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 583)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 583): s0, s1
s_mul_i32 s1, s20, s7 // (op 583) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 583): s14
s_mul_i32 s14, s21, s6 // (op 583) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 583) least significant
// Allocated : 1 SGPR (Value: Int32) (op 583): s15
s_mul_hi_u32 s15, s20, s6 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 583) most significant: sum
s_add_u32 s1, s1, s15 // (op 583) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 583): s15

 // Freeing : 1 SGPR (Value: Int32) (op 583): s14

 // (op 583) reg expression
 // (op 583) reg expression
// Allocated DataFlowTag491: 4 SGPRs (Buffer: None) (op 583): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 583) reg expression
s_mov_b32 s49, s23 // (op 583) reg expression
s_mov_b32 s50, s0 // (op 583) reg expression
s_mov_b32 s51, 131072 // (op 583) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 583): s0, s1

 // (op 445) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(583) END
 // (op 445) NOP(573) BEGIN
 // (op 573) generate({})
 // (op 573) end: generate({})
 // (op 445) NOP(573) END
 // (op 445) Assign VGPR 1:U64(588) BEGIN
 // (op 588) Assign dim(500) = 1:U64
 // (op 588) Assign stride expression
 // (op 445) Assign VGPR 1:U64(588) END
 // (op 445) NOP(591) BEGIN
 // (op 591) generate({})
 // (op 591) end: generate({})
 // (op 445) NOP(591) END
 // (op 445) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}(600) BEGIN
 // (op 600) Assign dim(501) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 501: v**UNALLOCATED**
 // (op 600) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag501: VGPR Value: UInt32 x 1: (unallocated)
 // (op 600) reg expression
 // (op 600) reg expression
 // (op 600) reg expression
 // (op 600) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 600) convert
// Allocated : 1 VGPR (Value: UInt32) (op 600): v12
v_lshrrev_b32 v12, 7, v1 // (op 600) 
 // (op 600) BEGIN: Flatten[1]
 // (op 600) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 600): v16
v_and_b32 v16, 15, v1 // (op 600) 
 // (op 600) END: Flatten[1]
 // (op 600) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 600): v17
v_lshl_add_u32 v17, v12, 4, v16 // (op 600) 
 // (op 600) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 600): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 600): v12

 // (op 600) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 600): v12
v_lshl_add_u32 v12, s16, 4, v17 // (op 600) 
 // (op 600) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 600): v17

 // (op 600) BEGIN: Flatten[1]
 // (op 600) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 600): v16
v_and_b32 v16, 63, v1 // (op 600) 
 // (op 600) END: Flatten[1]
 // (op 600) LogicalShiftR({Flatten[1]: v16:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 600): v17
v_lshrrev_b32 v17, 4, v16 // (op 600) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 600): v16

 // (op 600) BEGIN: Split
// Allocated DataFlowTag501: 1 VGPR (Value: UInt32) (op 600): v16
v_mul_lo_u32 v16, v12, s20 // (op 600) 
v_add_u32 v16, v16, v17 // (op 600) 
 // (op 600) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 600): v17

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 600): v12

 // (op 445) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}(600) END
 // (op 445) Assign VGPR 4:U64(601) BEGIN
 // (op 601) Assign dim(502) = 4:U64
 // (op 601) Assign stride expression
 // (op 445) Assign VGPR 4:U64(601) END
 // (op 445) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(602) BEGIN
 // (op 602) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 602) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // (op 602) reg expression
 // (op 602) reg expression
 // (op 602) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 602) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 602)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 602)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 602): s0, s1
s_mul_i32 s1, s20, s7 // (op 602) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 602): s14
s_mul_i32 s14, s21, s6 // (op 602) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 602) least significant
// Allocated : 1 SGPR (Value: Int32) (op 602): s15
s_mul_hi_u32 s15, s20, s6 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 602) most significant: sum
s_add_u32 s1, s1, s15 // (op 602) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 602): s15

 // Freeing : 1 SGPR (Value: Int32) (op 602): s14

 // (op 602) reg expression
 // (op 602) reg expression
s_mov_b32 s48, s22 // (op 602) reg expression
s_mov_b32 s49, s23 // (op 602) reg expression
s_mov_b32 s50, s0 // (op 602) reg expression
s_mov_b32 s51, 131072 // (op 602) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 602): s0, s1

 // (op 445) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(602) END
 // (op 445) NOP(593) BEGIN
 // (op 593) generate({})
 // (op 593) end: generate({})
 // (op 445) NOP(593) END
 // (op 445) Assign VGPR 4:U64(607) BEGIN
 // (op 607) Assign dim(504) = 4:U64
 // (op 607) Assign stride expression
 // (op 445) Assign VGPR 4:U64(607) END
 // (op 445) NOP(594) BEGIN
 // (op 594) generate({})
 // (op 594) end: generate({})
 // (op 445) NOP(594) END
 // (op 445) Assign VGPR 1:U64(610) BEGIN
 // (op 610) Assign dim(506) = 1:U64
 // (op 610) Assign stride expression
 // (op 445) Assign VGPR 1:U64(610) END
 // (op 445) NOP(635) BEGIN
 // (op 635) generate({})
 // (op 635) end: generate({})
 // (op 445) NOP(635) END
 // (op 445) Assign VGPR 4:U64(642) BEGIN
 // (op 642) Assign dim(515) = 4:U64
 // (op 642) Assign stride expression
 // (op 445) Assign VGPR 4:U64(642) END
 // (op 445) NOP(636) BEGIN
 // (op 636) generate({})
 // (op 636) end: generate({})
 // (op 445) NOP(636) END
 // (op 445) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(645) BEGIN
 // (op 645) Assign dim(516) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 516: v**UNALLOCATED**
 // (op 645) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag516: VGPR Value: UInt32 x 1: (unallocated)
 // (op 645) reg expression
 // (op 645) reg expression
 // (op 645) reg expression
 // (op 645) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
 // (op 645) convert
// Allocated : 1 VGPR (Value: UInt32) (op 645): v12
v_and_b32 v12, -4, v1 // (op 645) 
 // (op 645) BEGIN: Flatten[1]TileTile
 // (op 645) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 645): v17
v_sub_u32 v17, v1, v12 // (op 645) 
 // (op 645) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 645): v12

 // (op 645) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 645): v12
v_lshrrev_b32 v12, 2, v1 // (op 645) 
 // (op 645) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 645): v18
v_lshl_add_u32 v18, s17, 5, v12 // (op 645) 
 // (op 645) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 645): v12

 // (op 645) Multiply({Tile: v18:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 645): v12
v_mul_lo_u32 v12, v18, s24 // (op 645) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 645): v18

 // (op 645) BEGIN: Split
 // (op 645) {Split: Add({Flatten[1]TileTile: v17:U32}, v12:U32)U32}
// Allocated DataFlowTag516: 1 VGPR (Value: UInt32) (op 645): v18
v_add_u32 v18, v17, v12 // (op 645) 
 // (op 645) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 645): v12

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 645): v17

 // (op 445) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(645) END
 // (op 445) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(646) BEGIN
 // (op 646) Assign dim(517) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 646) Assign stride expression
 // (op 445) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(646) END
 // (op 445) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(647) BEGIN
 // (op 647) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 510: s**UNALLOCATED**
 // (op 647) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: (unallocated)
 // (op 647) reg expression
 // (op 647) reg expression
 // (op 647) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 647) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 647)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 647)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 647): s0, s1
s_mul_i32 s1, s24, s13 // (op 647) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 647): s14
s_mul_i32 s14, s25, s12 // (op 647) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 647) least significant
// Allocated : 1 SGPR (Value: Int32) (op 647): s15
s_mul_hi_u32 s15, s24, s12 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 647) most significant: sum
s_add_u32 s1, s1, s15 // (op 647) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 647): s15

 // Freeing : 1 SGPR (Value: Int32) (op 647): s14

 // (op 647) reg expression
 // (op 647) reg expression
// Allocated DataFlowTag510: 4 SGPRs (Buffer: None) (op 647): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 647) reg expression
s_mov_b32 s53, s27 // (op 647) reg expression
s_mov_b32 s54, s0 // (op 647) reg expression
s_mov_b32 s55, 131072 // (op 647) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 647): s0, s1

 // (op 445) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(647) END
 // (op 445) NOP(637) BEGIN
 // (op 637) generate({})
 // (op 637) end: generate({})
 // (op 445) NOP(637) END
 // (op 445) Assign VGPR 1:U64(652) BEGIN
 // (op 652) Assign dim(519) = 1:U64
 // (op 652) Assign stride expression
 // (op 445) Assign VGPR 1:U64(652) END
 // (op 445) NOP(655) BEGIN
 // (op 655) generate({})
 // (op 655) end: generate({})
 // (op 445) NOP(655) END
 // (op 445) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(664) BEGIN
 // (op 664) Assign dim(520) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 520: v**UNALLOCATED**
 // (op 664) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag520: VGPR Value: UInt32 x 1: (unallocated)
 // (op 664) reg expression
 // (op 664) reg expression
 // (op 664) reg expression
 // (op 664) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
 // (op 664) convert
// Allocated : 1 VGPR (Value: UInt32) (op 664): v12
v_and_b32 v12, -4, v1 // (op 664) 
 // (op 664) BEGIN: Flatten[1]TileTile
 // (op 664) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 664): v17
v_sub_u32 v17, v1, v12 // (op 664) 
 // (op 664) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 664): v12

 // (op 664) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 664): v12
v_lshrrev_b32 v12, 2, v1 // (op 664) 
 // (op 664) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 664): v19
v_lshl_add_u32 v19, s17, 5, v12 // (op 664) 
 // (op 664) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 664): v12

 // (op 664) Multiply({Tile: v19:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 664): v12
v_mul_lo_u32 v12, v19, s24 // (op 664) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 664): v19

 // (op 664) BEGIN: Split
 // (op 664) {Split: Add({Flatten[1]TileTile: v17:U32}, v12:U32)U32}
// Allocated DataFlowTag520: 1 VGPR (Value: UInt32) (op 664): v19
v_add_u32 v19, v17, v12 // (op 664) 
 // (op 664) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 664): v12

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 664): v17

 // (op 445) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(664) END
 // (op 445) Assign VGPR 4:U64(665) BEGIN
 // (op 665) Assign dim(521) = 4:U64
 // (op 665) Assign stride expression
 // (op 445) Assign VGPR 4:U64(665) END
 // (op 445) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(666) BEGIN
 // (op 666) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 666) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // (op 666) reg expression
 // (op 666) reg expression
 // (op 666) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 666) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 666)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 666)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 666): s0, s1
s_mul_i32 s1, s24, s13 // (op 666) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 666): s14
s_mul_i32 s14, s25, s12 // (op 666) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 666) least significant
// Allocated : 1 SGPR (Value: Int32) (op 666): s15
s_mul_hi_u32 s15, s24, s12 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 666): s15

 // Freeing : 1 SGPR (Value: Int32) (op 666): s14

 // (op 666) reg expression
 // (op 666) reg expression
s_mov_b32 s52, s26 // (op 666) reg expression
s_mov_b32 s53, s27 // (op 666) reg expression
s_mov_b32 s54, s0 // (op 666) reg expression
s_mov_b32 s55, 131072 // (op 666) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 666): s0, s1

 // (op 445) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(666) END
 // (op 445) NOP(657) BEGIN
 // (op 657) generate({})
 // (op 657) end: generate({})
 // (op 445) NOP(657) END
 // (op 445) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(671) BEGIN
 // (op 671) Assign dim(523) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 671) Assign stride expression
 // (op 445) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(671) END
 // (op 445) NOP(658) BEGIN
 // (op 658) generate({})
 // (op 658) end: generate({})
 // (op 445) NOP(658) END
 // (op 445) Assign VGPR 1:U64(674) BEGIN
 // (op 674) Assign dim(525) = 1:U64
 // (op 674) Assign stride expression
 // (op 445) Assign VGPR 1:U64(674) END
 // (op 445) NOP(330) BEGIN
 // (op 330) generate({})
 // (op 330) end: generate({})
 // (op 445) NOP(330) END
 // (op 445) SetCoordinate(335) BEGIN
 // (op 335) generate({})
 // (op 335) end: generate({})
 // (op 335) generate({336})
 // (op 335) SetCoordinate(336) BEGIN
 // (op 336) generate({})
 // (op 336) end: generate({})
 // (op 336) generate({337})
 // (op 336) LoadTiled Value: E8M0(337) BEGIN
 // (op 337) GEN: loadMacroTileWAVE OP337 WaveTile 201
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

 // (op 337) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag497: VGPR Value: UInt32 x 1: v13
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
 // (op 337) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 337): v12
buffer_load_ubyte v12, v13, s[48:51], 0 offen offset: 0 // (op 337) Load value
// VMEM: Expected complete at 819 (current 339)
 // (op 336) LoadTiled Value: E8M0(337) END
 // (op 336) end: generate({337})
 // (op 335) SetCoordinate(336) END
 // (op 335) end: generate({336})
 // (op 445) SetCoordinate(335) END
 // (op 445) SetCoordinate(340) BEGIN
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 341): v20, v21, v22, v23, v24, v25
buffer_load_dwordx4 v[20:23], v14, s[44:47], 0 offen // (op 341) Load value
// VMEM: Expected complete at 820 (current 340)
buffer_load_dwordx2 v[24:25], v14, s[44:47], 0 offen offset:16 // (op 341) Load value
// VMEM: Expected complete at 821 (current 341)
 // (op 340) LoadTiled Value: FP6x16(341) END
 // (op 340) end: generate({341})
 // (op 445) SetCoordinate(340) END
 // (op 445) SetCoordinate(343) BEGIN
 // (op 343) generate({})
 // (op 343) end: generate({})
 // (op 343) generate({344})
 // (op 343) LoadTiled Value: E8M0(344) BEGIN
 // (op 344) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 344) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 344) Generate 1:U64 into nullptr
 // tag 10: v**UNALLOCATED**
 // (op 344) 	Dir = Load
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

 // (op 344) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag516: VGPR Value: UInt32 x 1: v18
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
 // (op 344) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 344): v17
buffer_load_ubyte v17, v18, s[52:55], 0 offen offset: 0 // (op 344) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1300 (current 820)
 // (op 343) LoadTiled Value: E8M0(344) END
 // (op 343) end: generate({344})
 // (op 445) SetCoordinate(343) END
 // (op 445) SetCoordinate(346) BEGIN
 // (op 346) generate({})
 // (op 346) end: generate({})
 // (op 346) generate({347})
 // (op 346) LoadTiled Value: FP6x16(347) BEGIN
 // (op 347) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 347) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 347) Generate 0:U64 into nullptr
 // tag 8: v**UNALLOCATED**
 // (op 347) 	Dir = Load
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

 // (op 347) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
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
 // (op 347) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: FP6x16) (op 347): v26, v27, v28
buffer_load_dwordx3 v[26:28], v10, s[40:43], 0 offen // (op 347) Load value
// VMEM: Expected complete at 1301 (current 821)
 // (op 346) LoadTiled Value: FP6x16(347) END
 // (op 346) end: generate({347})
 // (op 445) SetCoordinate(346) END
 // (op 445) SetCoordinate(349) BEGIN
 // (op 349) generate({})
 // (op 349) end: generate({})
 // (op 349) generate({350})
 // (op 349) SetCoordinate(350) BEGIN
 // (op 350) generate({})
 // (op 350) end: generate({})
 // (op 350) generate({351})
 // (op 350) LoadTiled Value: E8M0(351) BEGIN
 // (op 351) GEN: loadMacroTileWAVE OP351 WaveTile 201
 // (op 351) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 351) Generate 1:U64 into nullptr
 // tag 446: v**UNALLOCATED**
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

 // (op 351) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag497: VGPR Value: UInt32 x 1: v13
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
 // (op 351) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 351): v29
buffer_load_ubyte v29, v13, s[48:51], 0 offen offset:4 // (op 351) Load value
// VMEM: Expected complete at 1302 (current 822)
 // (op 350) LoadTiled Value: E8M0(351) END
 // (op 350) end: generate({351})
 // (op 349) SetCoordinate(350) END
 // (op 349) end: generate({350})
 // (op 445) SetCoordinate(349) END
 // (op 445) Deallocate{}(1129) BEGIN
 // (op 1129) Deallocate 497
 // Freeing DataFlowTag497: 1 VGPR (Value: UInt32) (op 581): v13

 // Deleting tag 497
 // (op 1129) Deallocate 498
 // Deleting tag 498
 // (op 1129) Deallocate 500
 // Deleting tag 500
 // (op 1129) Deallocate 496
 // Deleting tag 496
 // (op 1129) Deallocate 499
 // Deleting tag 499
 // (op 445) Deallocate{}(1129) END
 // (op 445) SetCoordinate(354) BEGIN
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 355): v30, v31, v32, v33, v34, v35
buffer_load_dwordx4 v[30:33], v14, s[44:47], 0 offen offset:96 // (op 355) Load value
// VMEM: Expected stall of 478. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1781 (current 1301)
buffer_load_dwordx2 v[34:35], v14, s[44:47], 0 offen offset:112 // (op 355) Load value
// VMEM: Expected complete at 1782 (current 1302)
 // (op 354) LoadTiled Value: FP6x16(355) END
 // (op 354) end: generate({355})
 // (op 445) SetCoordinate(354) END
 // (op 445) Deallocate{}(1131) BEGIN
 // (op 1131) Deallocate 478
 // Freeing DataFlowTag478: 1 VGPR (Value: UInt32) (op 517): v14

 // Deleting tag 478
 // (op 1131) Deallocate 481
 // Deleting tag 481
 // (op 1131) Deallocate 477
 // Deleting tag 477
 // (op 1131) Deallocate 479
 // Deleting tag 479
 // (op 1131) Deallocate 480
 // Deleting tag 480
 // (op 445) Deallocate{}(1131) END
 // (op 445) SetCoordinate(357) BEGIN
 // (op 357) generate({})
 // (op 357) end: generate({})
 // (op 357) generate({358})
 // (op 357) LoadTiled Value: E8M0(358) BEGIN
 // (op 358) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 358) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 358) Generate 1:U64 into nullptr
 // tag 438: v**UNALLOCATED**
 // (op 358) 	Dir = Load
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

 // (op 358) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag516: VGPR Value: UInt32 x 1: v18
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
 // (op 358) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 358): v13
buffer_load_ubyte v13, v18, s[52:55], 0 offen offset:4 // (op 358) Load value
// VMEM: Expected complete at 1783 (current 1303)
 // (op 357) LoadTiled Value: E8M0(358) END
 // (op 357) end: generate({358})
 // (op 445) SetCoordinate(357) END
 // (op 445) Deallocate{}(1133) BEGIN
 // (op 1133) Deallocate 519
 // Deleting tag 519
 // (op 1133) Deallocate 516
 // Freeing DataFlowTag516: 1 VGPR (Value: UInt32) (op 645): v18

 // Deleting tag 516
 // (op 1133) Deallocate 517
 // Deleting tag 517
 // (op 1133) Deallocate 518
 // Deleting tag 518
 // (op 1133) Deallocate 515
 // Deleting tag 515
 // (op 445) Deallocate{}(1133) END
 // (op 445) SetCoordinate(360) BEGIN
 // (op 360) generate({})
 // (op 360) end: generate({})
 // (op 360) generate({361})
 // (op 360) LoadTiled Value: FP6x16(361) BEGIN
 // (op 361) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 361) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 361) Generate 0:U64 into nullptr
 // tag 430: v**UNALLOCATED**
 // (op 361) 	Dir = Load
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

 // (op 361) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
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
 // (op 361) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: FP6x16) (op 361): v36, v37, v38
buffer_load_dwordx3 v[36:38], v10, s[40:43], 0 offen offset:96 // (op 361) Load value
// VMEM: Expected stall of 478. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2262 (current 1782)
 // (op 360) LoadTiled Value: FP6x16(361) END
 // (op 360) end: generate({361})
 // (op 445) SetCoordinate(360) END
 // (op 445) Deallocate{}(1135) BEGIN
 // (op 1135) Deallocate 462
 // Deleting tag 462
 // (op 1135) Deallocate 458
 // Deleting tag 458
 // (op 1135) Deallocate 460
 // Deleting tag 460
 // (op 1135) Deallocate 459
 // Freeing DataFlowTag459: 1 VGPR (Value: UInt32) (op 453): v10

 // Deleting tag 459
 // (op 1135) Deallocate 461
 // Deleting tag 461
 // (op 445) Deallocate{}(1135) END
 // (op 445) Scope(779) BEGIN
 // (op 779) Lock Scope 779
 // Lock Branch
 // (op 779) generate({776})
 // (op 779) NOP(776) BEGIN
 // (op 776) generate({})
 // (op 776) end: generate({})
 // (op 779) NOP(776) END
 // (op 779) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(784) BEGIN
 // (op 784) Assign dim(556) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 556: v**UNALLOCATED**
 // (op 784) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag556: VGPR Value: UInt32 x 1: (unallocated)
 // (op 784) reg expression
 // (op 784) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 784): v10
v_and_b32 v10, -4, v1 // (op 784) 
 // (op 784) BEGIN: Tile[1]
 // (op 784) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 784): v14
v_sub_u32 v14, v1, v10 // (op 784) 
 // (op 784) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 784): v10

 // (op 784) BEGIN: Flatten
 // (op 784) {Flatten: ShiftL({Tile[1]: v14:U32}, 5:U32)U32}
 // (op 784) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 784)         value (Tile[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 784)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 784): v10
v_lshlrev_b32 v10, 5, v14 // (op 784) 
 // (op 784) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 784): v14

 // (op 784) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 784): v14
v_lshrrev_b32 v14, 2, v1 // (op 784) 
 // (op 784) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 784): v18
v_lshl_add_u32 v18, v14, 7, v10 // (op 784) 
 // (op 784) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 784): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 784): v14

 // (op 784) Multiply({Flatten: v18:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 784): v10
v_mul_lo_u32 v10, v18, 6 // (op 784) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 784): v18

 // (op 784) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag556: 1 VGPR (Value: UInt32) (op 784): v14
v_lshrrev_b32 v14, 3, v10 // (op 784) 
 // Freeing : 1 VGPR (Value: UInt32) (op 784): v10

 // (op 779) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(784) END
 // (op 779) Assign VGPR 96:U32(785) BEGIN
 // (op 785) Assign dim(557) = 96:U32
 // (op 785) Assign stride expression
 // (op 779) Assign VGPR 96:U32(785) END
 // (op 779) NOP(777) BEGIN
 // (op 777) generate({})
 // (op 777) end: generate({})
 // (op 779) NOP(777) END
 // (op 779) Assign VGPR 0:U32(789) BEGIN
 // (op 789) Assign dim(559) = 0:U32
 // (op 789) Assign stride expression
 // (op 779) Assign VGPR 0:U32(789) END
 // (op 779) StoreLDSTile Value: FP6x16(363) BEGIN
 // (op 363) GEN: StoreLDSTile
 // (op 363) GEN: storeMacroTileLDS OP 363 LDS 24 MacroTile 9
 // (op 363) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 363) Generate 0:U32 into nullptr
 // (op 363) 	Dir = Store
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[20:25]
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(8)
ds_write_b128 v14, v[20:23] // (op 363) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 1886 (current 1791)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
s_waitcnt vmcnt(7)
ds_write_b64 v14, v[24:25] offset:16 // (op 363) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 1888 (current 1793)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 779) StoreLDSTile Value: FP6x16(363) END
 // (op 779) Deallocate{}(1137) BEGIN
 // (op 1137) Deallocate 559
 // Deleting tag 559
 // (op 1137) Deallocate 557
 // Deleting tag 557
 // (op 1137) Deallocate 556
 // Freeing DataFlowTag556: 1 VGPR (Value: UInt32) (op 784): v14

 // Deleting tag 556
 // (op 1137) Deallocate 558
 // Deleting tag 558
 // (op 779) Deallocate{}(1137) END
 // (op 779) end: generate({776})
 // (op 779) Unlock Scope 779
 // Unlock None
 // (op 445) Scope(779) END
 // (op 445) Scope(832) BEGIN
 // (op 832) Lock Scope 832
 // Lock Branch
 // (op 832) generate({829})
 // (op 832) NOP(829) BEGIN
 // (op 829) generate({})
 // (op 829) end: generate({})
 // (op 832) NOP(829) END
 // (op 832) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(837) BEGIN
 // (op 837) Assign dim(572) = {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}
 // tag 572: v**UNALLOCATED**
 // (op 837) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32} into DataFlowTag572: VGPR Value: UInt32 x 1: (unallocated)
 // (op 837) reg expression
 // (op 837) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 837): v10
v_and_b32 v10, -4, v1 // (op 837) 
 // (op 837) BEGIN: Tile[1]Flatten
 // (op 837) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 837): v14
v_sub_u32 v14, v1, v10 // (op 837) 
 // (op 837) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 837): v10

 // (op 837) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 837): v10
v_and_b32 v10, 4294967292, v1 // (op 837) 
 // (op 837) BEGIN: Flatten
 // (op 837) {Flatten: Add(v10:U32, {Tile[1]Flatten: v14:U32})U32}
// Allocated DataFlowTag572: 1 VGPR (Value: UInt32) (op 837): v18
v_add_u32 v18, v10, v14 // (op 837) 
 // (op 837) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 837): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 837): v10

 // (op 832) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(837) END
 // (op 832) Assign VGPR 4:U32(838) BEGIN
 // (op 838) Assign dim(573) = 4:U32
 // (op 838) Assign stride expression
 // (op 832) Assign VGPR 4:U32(838) END
 // (op 832) NOP(830) BEGIN
 // (op 830) generate({})
 // (op 830) end: generate({})
 // (op 832) NOP(830) END
 // (op 832) Assign VGPR 1:U32(842) BEGIN
 // (op 842) Assign dim(575) = 1:U32
 // (op 842) Assign stride expression
 // (op 832) Assign VGPR 1:U32(842) END
 // (op 832) StoreLDSTile Value: E8M0(364) BEGIN
 // (op 364) GEN: StoreLDSTile
 // (op 364) GEN: storeMacroTileLDS OP 364 LDS 32 MacroTile 10
 // (op 364) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 364) Generate 1:U32 into nullptr
 // (op 364) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
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
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
 // 	info.rowOffsetReg = DataFlowTag572: VGPR Value: UInt32 x 1: v18
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v18, v17 offset:3072 // (op 364) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 1894 (current 1799)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 832) StoreLDSTile Value: E8M0(364) END
 // (op 832) Deallocate{}(1139) BEGIN
 // (op 1139) Deallocate 573
 // Deleting tag 573
 // (op 1139) Deallocate 572
 // Freeing DataFlowTag572: 1 VGPR (Value: UInt32) (op 837): v18

 // Deleting tag 572
 // (op 1139) Deallocate 575
 // Deleting tag 575
 // (op 1139) Deallocate 574
 // Deleting tag 574
 // (op 832) Deallocate{}(1139) END
 // (op 832) end: generate({829})
 // (op 832) Unlock Scope 832
 // Unlock None
 // (op 445) Scope(832) END
 // (op 445) Scope(726) BEGIN
 // (op 726) Lock Scope 726
 // Lock Branch
 // (op 726) generate({723})
 // (op 726) NOP(723) BEGIN
 // (op 723) generate({})
 // (op 723) end: generate({})
 // (op 726) NOP(723) END
 // (op 726) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(731) BEGIN
 // (op 731) Assign dim(540) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 540: v**UNALLOCATED**
 // (op 731) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag540: VGPR Value: UInt32 x 1: (unallocated)
 // (op 731) reg expression
 // (op 731) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 731): v10
v_and_b32 v10, -8, v1 // (op 731) 
 // (op 731) BEGIN: Tile[1]
 // (op 731) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 731): v14
v_sub_u32 v14, v1, v10 // (op 731) 
 // (op 731) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 731): v10

 // (op 731) BEGIN: Flatten
 // (op 731) {Flatten: ShiftL({Tile[1]: v14:U32}, 4:U32)U32}
 // (op 731) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 731)         value (Tile[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 731)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 731): v10
v_lshlrev_b32 v10, 4, v14 // (op 731) 
 // (op 731) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 731): v14

 // (op 731) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 731): v14
v_lshrrev_b32 v14, 3, v1 // (op 731) 
 // (op 731) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 731): v18
v_lshl_add_u32 v18, v14, 7, v10 // (op 731) 
 // (op 731) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 731): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 731): v14

 // (op 731) Multiply({Flatten: v18:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 731): v10
v_mul_lo_u32 v10, v18, 6 // (op 731) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 731): v18

 // (op 731) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag540: 1 VGPR (Value: UInt32) (op 731): v14
v_lshrrev_b32 v14, 3, v10 // (op 731) 
 // Freeing : 1 VGPR (Value: UInt32) (op 731): v10

 // (op 726) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(731) END
 // (op 726) Assign VGPR 96:U32(732) BEGIN
 // (op 732) Assign dim(541) = 96:U32
 // (op 732) Assign stride expression
 // (op 726) Assign VGPR 96:U32(732) END
 // (op 726) NOP(724) BEGIN
 // (op 724) generate({})
 // (op 724) end: generate({})
 // (op 726) NOP(724) END
 // (op 726) Assign VGPR 0:U32(736) BEGIN
 // (op 736) Assign dim(543) = 0:U32
 // (op 736) Assign stride expression
 // (op 726) Assign VGPR 0:U32(736) END
 // (op 726) StoreLDSTile Value: FP6x16(365) BEGIN
 // (op 365) GEN: StoreLDSTile
 // (op 365) GEN: storeMacroTileLDS OP 365 LDS 16 MacroTile 8
 // (op 365) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 365) Generate 0:U32 into nullptr
 // (op 365) 	Dir = Store
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[26:28]
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b96 v14, v[26:28] offset:3200 // (op 365) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected stall of 80. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1983 (current 1888)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 726) StoreLDSTile Value: FP6x16(365) END
 // (op 726) Deallocate{}(1141) BEGIN
 // (op 1141) Deallocate 543
 // Deleting tag 543
 // (op 1141) Deallocate 540
 // Freeing DataFlowTag540: 1 VGPR (Value: UInt32) (op 731): v14

 // Deleting tag 540
 // (op 1141) Deallocate 541
 // Deleting tag 541
 // (op 1141) Deallocate 542
 // Deleting tag 542
 // (op 726) Deallocate{}(1141) END
 // (op 726) end: generate({723})
 // (op 726) Unlock Scope 726
 // Unlock None
 // (op 445) Scope(726) END
 // (op 445) NOP(333) BEGIN
 // (op 333) generate({})
 // (op 333) end: generate({})
 // (op 445) NOP(333) END
 // (op 445) NOP(698) BEGIN
 // (op 698) generate({})
 // (op 698) end: generate({})
 // (op 445) NOP(698) END
 // (op 445) NOP(695) BEGIN
 // (op 695) generate({})
 // (op 695) end: generate({})
 // (op 445) NOP(695) END
 // (op 445) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(702) BEGIN
 // (op 702) Assign dim(532) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 532: v**UNALLOCATED**
 // (op 702) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag532: VGPR Value: UInt32 x 1: (unallocated)
 // (op 702) reg expression
 // (op 702) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 702): v10
v_and_b32 v10, -8, v1 // (op 702) 
 // (op 702) BEGIN: Tile[1]
 // (op 702) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 702): v14
v_sub_u32 v14, v1, v10 // (op 702) 
 // (op 702) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 702): v10

 // (op 702) BEGIN: Flatten
 // (op 702) {Flatten: ShiftL({Tile[1]: v14:U32}, 4:U32)U32}
 // (op 702) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 702)         value (Tile[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 702)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 702): v10
v_lshlrev_b32 v10, 4, v14 // (op 702) 
 // (op 702) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 702): v14

 // (op 702) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 702): v14
v_lshrrev_b32 v14, 3, v1 // (op 702) 
 // (op 702) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 702): v18
v_lshl_add_u32 v18, v14, 7, v10 // (op 702) 
 // (op 702) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 702): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 702): v14

 // (op 702) Multiply({Flatten: v18:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 702): v10
v_mul_lo_u32 v10, v18, 6 // (op 702) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 702): v18

 // (op 702) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag532: 1 VGPR (Value: UInt32) (op 702): v14
v_lshrrev_b32 v14, 3, v10 // (op 702) 
 // Freeing : 1 VGPR (Value: UInt32) (op 702): v10

 // (op 445) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(702) END
 // (op 445) Assign VGPR 96:U32(703) BEGIN
 // (op 703) Assign dim(533) = 96:U32
 // (op 703) Assign stride expression
 // (op 445) Assign VGPR 96:U32(703) END
 // (op 445) NOP(696) BEGIN
 // (op 696) generate({})
 // (op 696) end: generate({})
 // (op 445) NOP(696) END
 // (op 445) Assign VGPR 0:U32(707) BEGIN
 // (op 707) Assign dim(535) = 0:U32
 // (op 707) Assign stride expression
 // (op 445) Assign VGPR 0:U32(707) END
 // (op 445) NOP(750) BEGIN
 // (op 750) generate({})
 // (op 750) end: generate({})
 // (op 445) NOP(750) END
 // (op 445) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(755) BEGIN
 // (op 755) Assign dim(548) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 548: v**UNALLOCATED**
 // (op 755) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag548: VGPR Value: UInt32 x 1: (unallocated)
 // (op 755) reg expression
 // (op 755) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 755): v10
v_and_b32 v10, -4, v1 // (op 755) 
 // (op 755) BEGIN: Tile[1]
 // (op 755) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 755): v18
v_sub_u32 v18, v1, v10 // (op 755) 
 // (op 755) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 755): v10

 // (op 755) BEGIN: Flatten
 // (op 755) {Flatten: ShiftL({Tile[1]: v18:U32}, 5:U32)U32}
 // (op 755) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 755)         value (Tile[1]: VGPR Value: UInt32 x 1: v18) 
 // (op 755)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 755): v10
v_lshlrev_b32 v10, 5, v18 // (op 755) 
 // (op 755) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 755): v18

 // (op 755) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 755): v18
v_lshrrev_b32 v18, 2, v1 // (op 755) 
 // (op 755) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 755): v39
v_lshl_add_u32 v39, v18, 7, v10 // (op 755) 
 // (op 755) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 755): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 755): v18

 // (op 755) Multiply({Flatten: v39:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 755): v10
v_mul_lo_u32 v10, v39, 6 // (op 755) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 755): v39

 // (op 755) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag548: 1 VGPR (Value: UInt32) (op 755): v18
v_lshrrev_b32 v18, 3, v10 // (op 755) 
 // Freeing : 1 VGPR (Value: UInt32) (op 755): v10

 // (op 445) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(755) END
 // (op 445) Assign VGPR 96:U32(756) BEGIN
 // (op 756) Assign dim(549) = 96:U32
 // (op 756) Assign stride expression
 // (op 445) Assign VGPR 96:U32(756) END
 // (op 445) NOP(751) BEGIN
 // (op 751) generate({})
 // (op 751) end: generate({})
 // (op 445) NOP(751) END
 // (op 445) Assign VGPR 0:U32(760) BEGIN
 // (op 760) Assign dim(551) = 0:U32
 // (op 760) Assign stride expression
 // (op 445) Assign VGPR 0:U32(760) END
 // (op 445) NOP(803) BEGIN
 // (op 803) generate({})
 // (op 803) end: generate({})
 // (op 445) NOP(803) END
 // (op 445) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(808) BEGIN
 // (op 808) Assign dim(564) = {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}
 // tag 564: v**UNALLOCATED**
 // (op 808) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32} into DataFlowTag564: VGPR Value: UInt32 x 1: (unallocated)
 // (op 808) reg expression
 // (op 808) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 808): v10
v_and_b32 v10, -4, v1 // (op 808) 
 // (op 808) BEGIN: Tile[1]Flatten
 // (op 808) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 808): v39
v_sub_u32 v39, v1, v10 // (op 808) 
 // (op 808) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 808): v10

 // (op 808) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 808): v10
v_and_b32 v10, 4294967292, v1 // (op 808) 
 // (op 808) BEGIN: Flatten
 // (op 808) {Flatten: Add(v10:U32, {Tile[1]Flatten: v39:U32})U32}
// Allocated DataFlowTag564: 1 VGPR (Value: UInt32) (op 808): v40
v_add_u32 v40, v10, v39 // (op 808) 
 // (op 808) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 808): v39

 // Freeing : 1 VGPR (Value: UInt32) (op 808): v10

 // (op 445) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(808) END
 // (op 445) Assign VGPR 4:U32(809) BEGIN
 // (op 809) Assign dim(565) = 4:U32
 // (op 809) Assign stride expression
 // (op 445) Assign VGPR 4:U32(809) END
 // (op 445) NOP(804) BEGIN
 // (op 804) generate({})
 // (op 804) end: generate({})
 // (op 445) NOP(804) END
 // (op 445) Assign VGPR 1:U32(813) BEGIN
 // (op 813) Assign dim(567) = 1:U32
 // (op 813) Assign stride expression
 // (op 445) Assign VGPR 1:U32(813) END
 // (op 445) NOP(890) BEGIN
 // (op 890) generate({})
 // (op 890) end: generate({})
 // (op 445) NOP(890) END
 // (op 445) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(895) BEGIN
 // (op 895) Assign dim(590) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 590: v**UNALLOCATED**
 // (op 895) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag590: VGPR Value: UInt32 x 1: (unallocated)
 // (op 895) reg expression
 // (op 895) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 895): v10
v_and_b32 v10, -8, v1 // (op 895) 
 // (op 895) BEGIN: Tile[1]
 // (op 895) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 895): v39
v_sub_u32 v39, v1, v10 // (op 895) 
 // (op 895) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 895): v10

 // (op 895) BEGIN: Flatten
 // (op 895) {Flatten: ShiftL({Tile[1]: v39:U32}, 4:U32)U32}
 // (op 895) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 895)         value (Tile[1]: VGPR Value: UInt32 x 1: v39) 
 // (op 895)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 895): v10
v_lshlrev_b32 v10, 4, v39 // (op 895) 
 // (op 895) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 895): v39

 // (op 895) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 895): v39
v_lshrrev_b32 v39, 3, v1 // (op 895) 
 // (op 895) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 895): v41
v_lshl_add_u32 v41, v39, 7, v10 // (op 895) 
 // (op 895) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 895): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 895): v39

 // (op 895) Multiply({Flatten: v41:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 895): v10
v_mul_lo_u32 v10, v41, 6 // (op 895) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 895): v41

 // (op 895) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag590: 1 VGPR (Value: UInt32) (op 895): v39
v_lshrrev_b32 v39, 3, v10 // (op 895) 
 // Freeing : 1 VGPR (Value: UInt32) (op 895): v10

 // (op 445) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(895) END
 // (op 445) Assign VGPR 96:U32(896) BEGIN
 // (op 896) Assign dim(591) = 96:U32
 // (op 896) Assign stride expression
 // (op 445) Assign VGPR 96:U32(896) END
 // (op 445) NOP(891) BEGIN
 // (op 891) generate({})
 // (op 891) end: generate({})
 // (op 445) NOP(891) END
 // (op 445) Assign VGPR 0:U32(900) BEGIN
 // (op 900) Assign dim(593) = 0:U32
 // (op 900) Assign stride expression
 // (op 445) Assign VGPR 0:U32(900) END
 // (op 445) NOP(914) BEGIN
 // (op 914) generate({})
 // (op 914) end: generate({})
 // (op 445) NOP(914) END
 // (op 445) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(919) BEGIN
 // (op 919) Assign dim(598) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 598: v**UNALLOCATED**
 // (op 919) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag598: VGPR Value: UInt32 x 1: (unallocated)
 // (op 919) reg expression
 // (op 919) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 919): v10
v_and_b32 v10, -4, v1 // (op 919) 
 // (op 919) BEGIN: Tile[1]
 // (op 919) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 919): v41
v_sub_u32 v41, v1, v10 // (op 919) 
 // (op 919) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 919): v10

 // (op 919) BEGIN: Flatten
 // (op 919) {Flatten: ShiftL({Tile[1]: v41:U32}, 5:U32)U32}
 // (op 919) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 919)         value (Tile[1]: VGPR Value: UInt32 x 1: v41) 
 // (op 919)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 919): v10
v_lshlrev_b32 v10, 5, v41 // (op 919) 
 // (op 919) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 919): v41

 // (op 919) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 919): v41
v_lshrrev_b32 v41, 2, v1 // (op 919) 
 // (op 919) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 919): v42
v_lshl_add_u32 v42, v41, 7, v10 // (op 919) 
 // (op 919) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 919): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 919): v41

 // (op 919) Multiply({Flatten: v42:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 919): v10
v_mul_lo_u32 v10, v42, 6 // (op 919) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 919): v42

 // (op 919) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag598: 1 VGPR (Value: UInt32) (op 919): v41
v_lshrrev_b32 v41, 3, v10 // (op 919) 
 // Freeing : 1 VGPR (Value: UInt32) (op 919): v10

 // (op 445) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(919) END
 // (op 445) Assign VGPR 96:U32(920) BEGIN
 // (op 920) Assign dim(599) = 96:U32
 // (op 920) Assign stride expression
 // (op 445) Assign VGPR 96:U32(920) END
 // (op 445) NOP(915) BEGIN
 // (op 915) generate({})
 // (op 915) end: generate({})
 // (op 445) NOP(915) END
 // (op 445) Assign VGPR 0:U32(924) BEGIN
 // (op 924) Assign dim(601) = 0:U32
 // (op 924) Assign stride expression
 // (op 445) Assign VGPR 0:U32(924) END
 // (op 445) NOP(938) BEGIN
 // (op 938) generate({})
 // (op 938) end: generate({})
 // (op 445) NOP(938) END
 // (op 445) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(943) BEGIN
 // (op 943) Assign dim(606) = {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}
 // tag 606: v**UNALLOCATED**
 // (op 943) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32} into DataFlowTag606: VGPR Value: UInt32 x 1: (unallocated)
 // (op 943) reg expression
 // (op 943) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 943): v10
v_and_b32 v10, -4, v1 // (op 943) 
 // (op 943) BEGIN: Tile[1]Flatten
 // (op 943) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 943): v42
v_sub_u32 v42, v1, v10 // (op 943) 
 // (op 943) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 943): v10

 // (op 943) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 943): v10
v_and_b32 v10, 4294967292, v1 // (op 943) 
 // (op 943) BEGIN: Flatten
 // (op 943) {Flatten: Add(v10:U32, {Tile[1]Flatten: v42:U32})U32}
// Allocated DataFlowTag606: 1 VGPR (Value: UInt32) (op 943): v43
v_add_u32 v43, v10, v42 // (op 943) 
 // (op 943) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 943): v42

 // Freeing : 1 VGPR (Value: UInt32) (op 943): v10

 // (op 445) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(943) END
 // (op 445) Assign VGPR 4:U32(944) BEGIN
 // (op 944) Assign dim(607) = 4:U32
 // (op 944) Assign stride expression
 // (op 445) Assign VGPR 4:U32(944) END
 // (op 445) NOP(939) BEGIN
 // (op 939) generate({})
 // (op 939) end: generate({})
 // (op 445) NOP(939) END
 // (op 445) Assign VGPR 1:U32(948) BEGIN
 // (op 948) Assign dim(609) = 1:U32
 // (op 948) Assign stride expression
 // (op 445) Assign VGPR 1:U32(948) END
 // (op 445) ForLoopOp KLoop: LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL(54) BEGIN
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
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54:
 // (op 54) 
 // (op 54) generate({56, 988})
 // (op 54) Assign SGPR Add(DataFlowTag(361)U32, 2:U32)U32(56) BEGIN
 // (op 56) Assign dim(361) = Add(DataFlowTag(361)U32, 2:U32)U32
 // (op 56) Generate Add(DataFlowTag(361)U32, 2:U32)U32 into DataFlowTag361: SGPR Value: UInt32 x 1: s0
 // (op 56) reg expression
 // (op 56) Add({DataFlowTag361: s0:U32}, 2:U32)U32
 // (op 54) Barrier(988) BEGIN
s_add_u32 s0, s0, 2 // (op 56) 
 // (op 54) Assign SGPR Add(DataFlowTag(361)U32, 2:U32)U32(56) END
s_waitcnt lgkmcnt(0)
s_barrier  // (op 988) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 54) Barrier(988) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
ds_read_b128 v[44:47], v3 offset:3200 // (op 65) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 2081 (current 1986)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
ds_read_b64 v[48:49], v3 offset:3216 // (op 65) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 2082 (current 1987)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 63) LoadLDSTile{Value: FP6x16}(65) END
 // (op 63) end: generate({65})
 // (op 54) SetCoordinate(63) END
 // (op 54) Barrier(1001) BEGIN
s_barrier  // (op 1001) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 54) Barrier(1001) END
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
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 2084 (current 1989)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
ds_read_b64 v[54:55], v4 offset:16 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected stall of 92. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2177 (current 2082)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 68) LoadLDSTile{Value: FP6x16}(70) END
 // (op 68) end: generate({70})
 // (op 54) SetCoordinate(68) END
 // (op 54) Barrier(1015) BEGIN
s_barrier  // (op 1015) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 54) Barrier(1015) END
 // (op 54) SetCoordinate(77) BEGIN
 // (op 77) generate({})
 // (op 77) end: generate({})
 // (op 77) generate({79})
 // (op 77) LoadLDSTile{Value: E8M0}(79) BEGIN
 // (op 79) GEN: loadMacroTileWAVELDS OP 79 LDS 32 WaveTile 261
 // (op 79) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 79) Generate 1:U32 into nullptr
 // tag 30: v**UNALLOCATED**
 // (op 79) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
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

 // (op 79) LSTInfo {
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 79) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 79): v10
ds_read_u8 v10, v0 offset:3072 // (op 79) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 2179 (current 2084)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 77) LoadLDSTile{Value: E8M0}(79) END
 // (op 77) end: generate({79})
 // (op 54) SetCoordinate(77) END
 // (op 54) Multiply(82) BEGIN
 // (op 82) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile16xFP6, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:49], v[50:55], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 82) 
 // (op 54) Multiply(82) END
 // (op 54) Deallocate{}(1057) BEGIN
 // (op 1057) Deallocate 14
 // Freeing : 6 VGPRs (Value: FP6x16) (op 65): v44, v45, v46, v47, v48, v49

 // Deleting tag 14
 // (op 1057) Deallocate 22
 // Freeing : 6 VGPRs (Value: FP6x16) (op 70): v50, v51, v52, v53, v54, v55

 // Deleting tag 22
 // (op 1057) Deallocate 30
 // Freeing : 1 VGPR (Value: E8M0) (op 79): v10

 // Deleting tag 30
 // (op 54) Deallocate{}(1057) END
 // (op 54) NOP(384) BEGIN
 // (op 384) generate({})
 // (op 384) end: generate({})
 // (op 54) NOP(384) END
 // (op 54) SetCoordinate(73) BEGIN
 // (op 73) generate({})
 // (op 73) end: generate({})
 // (op 73) generate({264})
 // (op 73) SetCoordinate(264) BEGIN
 // (op 264) generate({})
 // (op 264) end: generate({})
 // (op 264) generate({75})
 // (op 264) LoadTiled Value: E8M0(75) BEGIN
 // (op 75) GEN: loadMacroTileWAVE OP75 WaveTile 201
 // (op 75) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 75) Generate 1:U64 into nullptr
 // (op 75) 	Dir = Load
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
// Allocated Offset503: 1 VGPR (Value: UInt32) (op 75): v10
v_mov_b32 v10, v16 // (op 75) 
 // (op 75) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = Offset503: VGPR Value: UInt32 x 1: v10
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
 // (op 75) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v12, v10, s[48:51], 0 offen offset:8 // (op 75) Load value
// VMEM: Expected complete at 2517 (current 2037)
 // (op 264) LoadTiled Value: E8M0(75) END
 // (op 264) end: generate({75})
 // (op 73) SetCoordinate(264) END
 // (op 73) end: generate({264})
 // (op 54) SetCoordinate(73) END
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[20:25]
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_dwordx4 v[20:23], v42, s[44:47], 0 offen offset:192 // (op 10) Load value
// VMEM: Expected complete at 2519 (current 2039)
buffer_load_dwordx2 v[24:25], v42, s[44:47], 0 offen offset:208 // (op 10) Load value
// VMEM: Expected stall of 223. CBNW: 4, Inc: 3
// VMEM: Expected complete at 2743 (current 2263)
 // (op 267) LoadTiled Value: FP6x16(10) END
 // (op 267) end: generate({10})
 // (op 54) SetCoordinate(267) END
 // (op 54) SetCoordinate(272) BEGIN
 // (op 272) generate({})
 // (op 272) end: generate({})
 // (op 272) generate({21})
 // (op 272) LoadTiled Value: E8M0(21) BEGIN
 // (op 21) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 21) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 21) Generate 1:U64 into nullptr
 // (op 21) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
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
// Allocated Offset522: 1 VGPR (Value: UInt32) (op 21): v44
v_mov_b32 v44, v19 // (op 21) 
 // (op 21) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
 // 	info.rowOffsetReg = Offset522: VGPR Value: UInt32 x 1: v44
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
 // (op 21) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v17, v44, s[52:55], 0 offen offset:8 // (op 21) Load value
// VMEM: Expected stall of 253. CBNW: 5, Inc: 3
// VMEM: Expected complete at 2998 (current 2518)
 // (op 272) LoadTiled Value: E8M0(21) END
 // (op 272) end: generate({21})
 // (op 54) SetCoordinate(272) END
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
// Allocated Offset465: 1 VGPR (Value: UInt32) (op 4): v45
v_mov_b32 v45, v11 // (op 4) 
 // (op 4) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[26:28]
 // 	info.rowOffsetReg = Offset465: VGPR Value: UInt32 x 1: v45
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_dwordx3 v[26:28], v45, s[40:43], 0 offen offset:192 // (op 4) Load value
// VMEM: Expected complete at 3000 (current 2520)
 // (op 259) LoadTiled Value: FP6x16(4) END
 // (op 259) end: generate({4})
 // (op 54) SetCoordinate(259) END
 // (op 54) Barrier(1038) BEGIN
s_barrier  // (op 1038) 
 // (op 54) Barrier(1038) END
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[30:35]
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(8)
ds_write_b128 v41, v[30:33] offset:4736 // (op 235) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 2764 (current 2669)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 3072)
s_waitcnt vmcnt(7)
ds_write_b64 v41, v[34:35] offset:4752 // (op 235) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 2766 (current 2671)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 3072)
 // (op 54) StoreLDSTile Value: FP6x16(235) END
 // (op 54) Deallocate{}(1115) BEGIN
 // (op 1115) Deallocate 600
 // Deleting tag 600
 // (op 54) Deallocate{}(1115) END
 // (op 54) Barrier(1044) BEGIN
s_barrier  // (op 1044) 
 // (op 54) Barrier(1044) END
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7808
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
 // 	info.data = VGPR Value: E8M0 x 1: v13
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
 // 	info.offset = Literal Value: UInt32 x 0: 7808
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

s_waitcnt vmcnt(6)
ds_write_b8 v43, v13 offset:7808 // (op 236) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 2769 (current 2674)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 7808, Size: 128)
 // (op 54) StoreLDSTile Value: E8M0(236) END
 // (op 54) Deallocate{}(1117) BEGIN
 // (op 1117) Deallocate 608
 // Deleting tag 608
 // (op 54) Deallocate{}(1117) END
 // (op 54) Barrier(1032) BEGIN
s_barrier  // (op 1032) 
 // (op 54) Barrier(1032) END
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
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[36:38]
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
 // 	info.offset = Literal Value: UInt32 x 0: 7936
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
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[36:38]
 // 	info.rowOffsetReg = DataFlowTag590: VGPR Value: UInt32 x 1: v39
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
 // 	info.offset = Literal Value: UInt32 x 0: 7936
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b96 v39, v[36:38] offset:7936 // (op 225) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2861 (current 2766)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 7936, Size: 1536)
 // (op 54) StoreLDSTile Value: FP6x16(225) END
 // (op 54) Deallocate{}(1091) BEGIN
 // (op 1091) Deallocate 592
 // Deleting tag 592
 // (op 54) Deallocate{}(1091) END
 // (op 54) NOP(378) BEGIN
 // (op 378) generate({})
 // (op 378) end: generate({})
 // (op 54) NOP(378) END
 // (op 54) Barrier(1029) BEGIN
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1029) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 7936, Size: 1536)
 // (op 54) Barrier(1029) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 7936
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
 // 	info.offset = Literal Value: UInt32 x 0: 7936
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
ds_read_b128 v[46:49], v6 offset:7936 // (op 227) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 2959 (current 2864)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 7936, Size: 1536)
ds_read_b64 v[50:51], v6 offset:7952 // (op 227) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 2960 (current 2865)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 7936, Size: 1536)
 // (op 226) LoadLDSTile{Value: FP6x16}(227) END
 // (op 226) Deallocate{}(1097) BEGIN
 // (op 1097) Deallocate 596
 // Deleting tag 596
 // (op 226) Deallocate{}(1097) END
 // (op 226) end: generate({227})
 // (op 54) SetCoordinate(226) END
 // (op 54) Deallocate{}(1093) BEGIN
 // (op 1093) Deallocate 432
 // Deleting tag 432
 // (op 54) Deallocate{}(1093) END
 // (op 54) Barrier(1035) BEGIN
 // (op 54) Barrier(1095) BEGIN
s_barrier  // (op 1035) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 3072)
 // (op 54) Barrier(1035) END
s_barrier  // (op 1095) 
 // (op 54) Barrier(1095) END
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
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 2963 (current 2868)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 3072)
ds_read_b64 v[56:57], v7 offset:4752 // (op 229) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected stall of 91. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 3055 (current 2960)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 4736, Size: 3072)
 // (op 228) LoadLDSTile{Value: FP6x16}(229) END
 // (op 228) Deallocate{}(1103) BEGIN
 // (op 1103) Deallocate 604
 // Deleting tag 604
 // (op 228) Deallocate{}(1103) END
 // (op 228) end: generate({229})
 // (op 54) SetCoordinate(228) END
 // (op 54) Deallocate{}(1099) BEGIN
 // (op 1099) Deallocate 436
 // Deleting tag 436
 // (op 54) Deallocate{}(1099) END
 // (op 54) Barrier(1041) BEGIN
 // (op 54) Barrier(1101) BEGIN
s_barrier  // (op 1041) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7808, Size: 128)
 // (op 54) Barrier(1041) END
s_barrier  // (op 1101) 
 // (op 54) Barrier(1101) END
 // (op 54) SetCoordinate(232) BEGIN
 // (op 232) generate({})
 // (op 232) end: generate({})
 // (op 232) generate({233})
 // (op 232) LoadLDSTile{Value: E8M0}(233) BEGIN
 // (op 233) GEN: loadMacroTileWAVELDS OP 233 LDS 440 WaveTile 261
 // (op 233) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 233) Generate 1:U32 into nullptr
 // tag 448: v**UNALLOCATED**
 // (op 233) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 7808
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

 // (op 233) LSTInfo {
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
 // 	info.offset = Literal Value: UInt32 x 0: 7808
 // 	info.bufDesc = (nullptr)
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

// Allocated : 1 VGPR (Value: E8M0) (op 233): v58
ds_read_u8 v58, v5 offset:7808 // (op 233) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 3058 (current 2963)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7808, Size: 128)
 // (op 232) LoadLDSTile{Value: E8M0}(233) END
 // (op 232) Deallocate{}(1111) BEGIN
 // (op 1111) Deallocate 612
 // Deleting tag 612
 // (op 232) Deallocate{}(1111) END
 // (op 232) end: generate({233})
 // (op 54) SetCoordinate(232) END
 // (op 54) Deallocate{}(1107) BEGIN
 // (op 1107) Deallocate 440
 // Deleting tag 440
 // (op 54) Deallocate{}(1107) END
 // (op 54) Multiply(234) BEGIN
 // (op 234) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile16xFP6, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
 // (op 54) Barrier(1109) BEGIN
s_barrier  // (op 1109) 
 // (op 54) Barrier(1109) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:57], a[0:3], v29, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 234) 
 // (op 54) Multiply(234) END
 // (op 54) Deallocate{}(1113) BEGIN
 // (op 1113) Deallocate 448
 // Freeing : 1 VGPR (Value: E8M0) (op 233): v58

 // Deleting tag 448
 // (op 1113) Deallocate 444
 // Freeing : 6 VGPRs (Value: FP6x16) (op 229): v52, v53, v54, v55, v56, v57

 // Deleting tag 444
 // (op 1113) Deallocate 442
 // Freeing : 6 VGPRs (Value: FP6x16) (op 227): v46, v47, v48, v49, v50, v51

 // Deleting tag 442
 // (op 54) Deallocate{}(1113) END
 // (op 54) NOP(402) BEGIN
 // (op 402) generate({})
 // (op 402) end: generate({})
 // (op 54) NOP(402) END
 // (op 54) SetCoordinate(230) BEGIN
 // (op 230) generate({})
 // (op 230) end: generate({})
 // (op 230) generate({283})
 // (op 230) SetCoordinate(283) BEGIN
 // (op 283) generate({})
 // (op 283) end: generate({})
 // (op 283) generate({231})
 // (op 283) LoadTiled Value: E8M0(231) BEGIN
 // (op 231) GEN: loadMacroTileWAVE OP231 WaveTile 201
 // (op 231) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 231) Generate 1:U64 into nullptr
 // (op 231) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v29
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

 // (op 231) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v29
 // 	info.rowOffsetReg = Offset503: VGPR Value: UInt32 x 1: v10
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
 // (op 231) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v29, v10, s[48:51], 0 offen offset:12 // (op 231) Load value
// VMEM: Expected stall of 18. CBNW: 2, Inc: 3
// VMEM: Expected complete at 3224 (current 2744)
 // (op 283) LoadTiled Value: E8M0(231) END
 // (op 283) end: generate({231})
 // (op 230) SetCoordinate(283) END
 // (op 230) end: generate({283})
 // (op 54) SetCoordinate(230) END
 // (op 54) Deallocate{}(1105) BEGIN
 // (op 1105) Deallocate 503
 // Freeing Offset503: 1 VGPR (Value: UInt32) (op 75): v10

 // Deleting tag 503
 // (op 1105) Deallocate 505
 // Deleting tag 505
 // (op 54) Deallocate{}(1105) END
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[30:35]
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 54) Assign VGPR Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32(592) BEGIN
 // (op 592) Assign dim(501) = Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32
 // (op 592) Generate Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32 into DataFlowTag501: VGPR Value: UInt32 x 1: v16
 // (op 592) reg expression
 // (op 592) Add({DataFlowTag501: v16:U32}, 8:U32)U32
v_add_u32 v16, 8, v16 // (op 592) 
 // (op 54) Assign VGPR Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32(592) END
buffer_load_dwordx4 v[30:33], v42, s[44:47], 0 offen offset:288 // (op 237) Load value
// VMEM: Expected stall of 253. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3479 (current 2999)
buffer_load_dwordx2 v[34:35], v42, s[44:47], 0 offen offset:304 // (op 237) Load value
// VMEM: Expected stall of 1. CBNW: 4, Inc: 3
// VMEM: Expected complete at 3481 (current 3001)
 // (op 286) LoadTiled Value: FP6x16(237) END
 // (op 286) end: generate({237})
 // (op 54) SetCoordinate(286) END
 // (op 54) Deallocate{}(1121) BEGIN
 // (op 1121) Deallocate 484
 // Freeing Offset484: 1 VGPR (Value: UInt32) (op 10): v42

 // Deleting tag 484
 // (op 1121) Deallocate 486
 // Deleting tag 486
 // (op 54) Deallocate{}(1121) END
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

 // (op 238) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = Offset522: VGPR Value: UInt32 x 1: v44
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
 // (op 238) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 54) Assign VGPR Add(DataFlowTag(482)U32, ShiftL(Convert(DataFlowTag(483)U64)U32, 1:U32)U32)U32(528) BEGIN
 // (op 528) Assign dim(482) = Add(DataFlowTag(482)U32, ShiftL(Convert(DataFlowTag(483)U64)U32, 1:U32)U32)U32
 // (op 528) Generate Add(DataFlowTag(482)U32, ShiftL(Convert(DataFlowTag(483)U64)U32, 1:U32)U32)U32 into DataFlowTag482: VGPR Value: UInt32 x 1: v15
 // (op 528) reg expression
 // (op 528) Add({DataFlowTag482: v15:U32}, 192:U32)U32
v_add_u32 v15, 192, v15 // (op 528) 
 // (op 54) Assign VGPR Add(DataFlowTag(482)U32, ShiftL(Convert(DataFlowTag(483)U64)U32, 1:U32)U32)U32(528) END
buffer_load_ubyte v13, v44, s[52:55], 0 offen offset:12 // (op 238) Load value
// VMEM: Expected stall of 222. CBNW: 5, Inc: 3
// VMEM: Expected complete at 3705 (current 3225)
 // (op 291) LoadTiled Value: E8M0(238) END
 // (op 291) end: generate({238})
 // (op 54) SetCoordinate(291) END
 // (op 54) Deallocate{}(1123) BEGIN
 // (op 1123) Deallocate 522
 // Freeing Offset522: 1 VGPR (Value: UInt32) (op 21): v44

 // Deleting tag 522
 // (op 1123) Deallocate 524
 // Deleting tag 524
 // (op 54) Deallocate{}(1123) END
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
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[36:38]
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
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[36:38]
 // 	info.rowOffsetReg = Offset465: VGPR Value: UInt32 x 1: v45
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 54) Assign VGPR Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32(656) BEGIN
 // (op 656) Assign dim(520) = Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32
 // (op 656) Generate Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32 into DataFlowTag520: VGPR Value: UInt32 x 1: v19
 // (op 656) reg expression
 // (op 656) Add({DataFlowTag520: v19:U32}, 8:U32)U32
v_add_u32 v19, 8, v19 // (op 656) 
 // (op 54) Assign VGPR Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32(656) END
buffer_load_dwordx3 v[36:38], v45, s[40:43], 0 offen offset:288 // (op 224) Load value
// VMEM: Expected stall of 253. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3960 (current 3480)
 // (op 278) LoadTiled Value: FP6x16(224) END
 // (op 278) end: generate({224})
 // (op 54) SetCoordinate(278) END
 // (op 54) Deallocate{}(1119) BEGIN
 // (op 1119) Deallocate 467
 // Deleting tag 467
 // (op 1119) Deallocate 465
 // Freeing Offset465: 1 VGPR (Value: UInt32) (op 4): v45

 // Deleting tag 465
 // (op 54) Deallocate{}(1119) END
 // (op 54) Assign VGPR Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32(464) BEGIN
 // (op 464) Assign dim(463) = Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32
 // (op 464) Generate Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32 into DataFlowTag463: VGPR Value: UInt32 x 1: v11
 // (op 464) reg expression
 // (op 464) Add({DataFlowTag463: v11:U32}, 192:U32)U32
 // (op 54) Barrier(1004) BEGIN
s_barrier  // (op 1004) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 54) Barrier(1004) END
v_add_u32 v11, 192, v11 // (op 464) 
 // (op 54) Assign VGPR Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32(464) END
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[20:25]
 // 	info.rowOffsetReg = DataFlowTag548: VGPR Value: UInt32 x 1: v18
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(8)
ds_write_b128 v18, v[20:23] // (op 13) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 3914 (current 3819)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
s_waitcnt vmcnt(7)
ds_write_b64 v18, v[24:25] offset:16 // (op 13) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 3916 (current 3821)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 54) StoreLDSTile Value: FP6x16(13) END
 // (op 54) Deallocate{}(1049) BEGIN
 // (op 1049) Deallocate 550
 // Deleting tag 550
 // (op 54) Deallocate{}(1049) END
 // (op 54) Barrier(1018) BEGIN
s_barrier  // (op 1018) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 54) Barrier(1018) END
 // (op 54) StoreLDSTile Value: E8M0(26) BEGIN
 // (op 26) GEN: StoreLDSTile
 // (op 26) GEN: storeMacroTileLDS OP 26 LDS 32 MacroTile 10
 // (op 26) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 26) Generate 1:U32 into nullptr
 // (op 26) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
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

 // (op 26) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 26) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v40, v17 offset:3072 // (op 26) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 3919 (current 3824)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 54) StoreLDSTile Value: E8M0(26) END
 // (op 54) Deallocate{}(1051) BEGIN
 // (op 1051) Deallocate 566
 // Deleting tag 566
 // (op 54) Deallocate{}(1051) END
 // (op 54) Barrier(991) BEGIN
s_barrier  // (op 991) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 54) Barrier(991) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[26:28]
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b96 v14, v[26:28] offset:3200 // (op 6) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4011 (current 3916)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 54) StoreLDSTile Value: FP6x16(6) END
 // (op 54) Deallocate{}(1047) BEGIN
 // (op 1047) Deallocate 534
 // Deleting tag 534
 // (op 54) Deallocate{}(1047) END
 // (op 54) NOP(379) BEGIN
 // (op 379) generate({})
 // (op 379) end: generate({})
 // (op 54) NOP(379) END
 // (op 54) end: generate({56, 988})
 // (op 54) For Loop Increment
 // (op 54) generate({})
 // (op 54) end: generate({})
 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 if true)
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
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54:
 // (op 54) 
 // (op 54) Unlock For Loop
 // Unlock None
 // (op 445) ForLoopOp KLoop: LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL(54) END
 // (op 445) Deallocate{}(1055) BEGIN
 // (op 1055) Deallocate 464
 // Deleting tag 464
 // (op 1055) Deallocate 485
 // Deleting tag 485
 // (op 1055) Deallocate 434
 // Deleting tag 434
 // (op 1055) Deallocate 502
 // Deleting tag 502
 // (op 1055) Deallocate 590
 // Freeing DataFlowTag590: 1 VGPR (Value: UInt32) (op 895): v39

 // Deleting tag 590
 // (op 1055) Deallocate 463
 // Freeing DataFlowTag463: 1 VGPR (Value: UInt32) (op 472): v11

 // Deleting tag 463
 // (op 1055) Deallocate 593
 // Deleting tag 593
 // (op 1055) Deallocate 466
 // Deleting tag 466
 // (op 1055) Deallocate 468
 // Deleting tag 468
 // (op 1055) Deallocate 599
 // Deleting tag 599
 // (op 1055) Deallocate 609
 // Deleting tag 609
 // (op 1055) Deallocate 482
 // Freeing DataFlowTag482: 1 VGPR (Value: UInt32) (op 536): v15

 // Deleting tag 482
 // (op 1055) Deallocate 483
 // Deleting tag 483
 // (op 1055) Deallocate 361
 // Freeing DataFlowTag361: 1 SGPR (Value: UInt32) (op 55): s0

 // Deleting tag 361
 // (op 1055) Deallocate 438
 // Deleting tag 438
 // (op 1055) Deallocate 565
 // Deleting tag 565
 // (op 1055) Deallocate 532
 // Freeing DataFlowTag532: 1 VGPR (Value: UInt32) (op 702): v14

 // Deleting tag 532
 // (op 1055) Deallocate 506
 // Deleting tag 506
 // (op 1055) Deallocate 533
 // Deleting tag 533
 // (op 1055) Deallocate 535
 // Deleting tag 535
 // (op 1055) Deallocate 548
 // Freeing DataFlowTag548: 1 VGPR (Value: UInt32) (op 755): v18

 // Deleting tag 548
 // (op 1055) Deallocate 549
 // Deleting tag 549
 // (op 1055) Deallocate 551
 // Deleting tag 551
 // (op 1055) Deallocate 564
 // Freeing DataFlowTag564: 1 VGPR (Value: UInt32) (op 808): v40

 // Deleting tag 564
 // (op 1055) Deallocate 567
 // Deleting tag 567
 // (op 1055) Deallocate 598
 // Freeing DataFlowTag598: 1 VGPR (Value: UInt32) (op 919): v41

 // Deleting tag 598
 // (op 1055) Deallocate 601
 // Deleting tag 601
 // (op 1055) Deallocate 446
 // Deleting tag 446
 // (op 1055) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 344): v17

 // Deleting tag 10
 // (op 1055) Deallocate 607
 // Deleting tag 607
 // (op 1055) Deallocate 430
 // Deleting tag 430
 // (op 1055) Deallocate 49
 // Freeing : 1 VGPR (Value: E8M0) (op 337): v12

 // Deleting tag 49
 // (op 1055) Deallocate 525
 // Deleting tag 525
 // (op 1055) Deallocate 523
 // Deleting tag 523
 // (op 1055) Deallocate 9
 // Freeing : 6 VGPRs (Value: FP6x16) (op 341): v20, v21, v22, v23, v24, v25

 // Deleting tag 9
 // (op 1055) Deallocate 521
 // Deleting tag 521
 // (op 1055) Deallocate 8
 // Freeing : 3 VGPRs (Value: FP6x16) (op 347): v26, v27, v28

 // Deleting tag 8
 // (op 1055) Deallocate 520
 // Freeing DataFlowTag520: 1 VGPR (Value: UInt32) (op 664): v19

 // Deleting tag 520
 // (op 1055) Deallocate 606
 // Freeing DataFlowTag606: 1 VGPR (Value: UInt32) (op 943): v43

 // Deleting tag 606
 // (op 1055) Deallocate 504
 // Deleting tag 504
 // (op 1055) Deallocate 501
 // Freeing DataFlowTag501: 1 VGPR (Value: UInt32) (op 600): v16

 // Deleting tag 501
 // (op 1055) Deallocate 487
 // Deleting tag 487
 // (op 1055) Deallocate 591
 // Deleting tag 591
 // (op 445) Deallocate{}(1055) END
 // (op 445) end: generate({440})
 // (op 445) Unlock Scope 445
 // Unlock None
 // (op 310) Scope(445) END
 // (op 310) SetCoordinate(201) BEGIN
 // (op 201) generate({})
 // (op 201) end: generate({})
 // (op 201) generate({202})
 // (op 201) SetCoordinate(202) BEGIN
 // (op 202) generate({})
 // (op 202) end: generate({})
 // (op 202) generate({422})
 // (op 202) Scope(422) BEGIN
 // (op 422) Lock Scope 422
 // Lock Branch
 // (op 422) generate({416})
 // (op 422) NOP(416) BEGIN
 // (op 416) generate({})
 // (op 416) end: generate({})
 // (op 422) NOP(416) END
 // (op 422) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(427) BEGIN
 // (op 427) Assign dim(450) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 450: v**UNALLOCATED**
 // (op 427) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag450: VGPR Value: UInt32 x 1: (unallocated)
 // (op 427) reg expression
 // (op 427) reg expression
 // (op 427) reg expression
 // (op 427) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
 // (op 427) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 427): v10
v_lshrrev_b32 v10, 3, v1 // (op 427) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 427): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 427) 
 // (op 427) BEGIN: Tile
 // (op 427) convert
// Allocated : 1 VGPR (Value: UInt32) (op 427): v11
v_lshl_add_u32 v11, s16, 4, v10 // (op 427) 
 // (op 427) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 427): v10

 // (op 427) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 427) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 427): s14
s_lshr_b32 s14, s0, 1 // (op 427) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 427): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 427): v10
v_and_b32 v10, -8, v1 // (op 427) 
 // (op 427) reg expression
 // (op 427) BEGIN: Flatten[1]
 // (op 427) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 427): v12
v_sub_u32 v12, v1, v10 // (op 427) 
 // (op 427) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 427): v10

 // (op 427) BEGIN: Tile
 // (op 427) {Tile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 427) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 427)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 427)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 427): v10
v_lshlrev_b32 v10, 4, v12 // (op 427) 
 // (op 427) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 427): v12

 // (op 427) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 427): v12
v_lshl_add_u32 v12, s14, 8, v10 // (op 427) 
 // (op 427) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 427): v10

 // Freeing : 1 SGPR (Value: UInt32) (op 427): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 427): v14, v15
v_mov_b32 v15, 0 // (op 427) convert
v_mov_b32 v14, v12 // (op 427) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 427): v12

 // (op 427) BEGIN: Split
 // (op 427) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 427)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 427)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 427): v16, v17
v_mul_lo_u32 v17, v11, s3 // (op 427) most significant: low of low * high
 // (op 427) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 427): v10
v_mul_hi_u32 v10, v11, s2 // (op 427) most significant: high of low * low
v_mul_lo_u32 v16, v11, s2 // (op 427) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 427) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 427): v10

 // (op 427) Add: dest (VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 427)      lhs (VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 427)      rhs (Tile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 427): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 427) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 427) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 427): s0, s1

 // (op 427) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 427): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 427): v11

 // (op 427) Multiply({Split: v[16:17]:U64}, 6:U32)U64
 // (op 427) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 427)           lhs (Split: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 427)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 427) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 427): v14, v15
v_mul_lo_u32 v15, v17, 6 // (op 427) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 427): v10
v_mul_hi_u32 v10, v16, 6 // (op 427) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 427) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 427) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 427): v10

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 427): v16, v17

 // (op 427) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 427): v10, v11
v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 427) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 427): v14, v15

// Allocated DataFlowTag450: 1 VGPR (Value: UInt32) (op 427): v12
v_mov_b32 v12, v10 // (op 427) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 427): v10, v11

 // (op 422) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(427) END
 // (op 422) Assign VGPR 96:U64(428) BEGIN
 // (op 428) Assign dim(451) = 96:U64
 // (op 428) Assign stride expression
 // (op 422) Assign VGPR 96:U64(428) END
 // (op 422) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(429) BEGIN
 // (op 429) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 429) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // (op 429) reg expression
 // (op 429) reg expression
 // (op 429) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 429) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 429)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 429)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 429) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 429): s0, s1
s_mul_i32 s1, s7, 6 // (op 429) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 429) least significant
// Allocated : 1 SGPR (Value: Int32) (op 429): s14
s_mul_hi_u32 s14, s6, 6 // (op 429) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 429) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 429): s14

 // (op 429) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 429) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 429)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 429)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 429): s14, s15
s_mul_i32 s15, s2, s1 // (op 429) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 429): s18
s_mul_i32 s18, s3, s0 // (op 429) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 429) least significant
// Allocated : 1 SGPR (Value: Int32) (op 429): s19
s_mul_hi_u32 s19, s2, s0 // (op 429) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 429) most significant: sum
s_add_u32 s15, s15, s19 // (op 429) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 429): s19

 // Freeing : 1 SGPR (Value: Int32) (op 429): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 429): s0, s1

 // (op 429) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 429): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 429) 
 // Freeing : 2 SGPRs (Value: Int64) (op 429): s14, s15

 // (op 429) reg expression
 // (op 429) reg expression
s_mov_b32 s40, s4 // (op 429) reg expression
s_mov_b32 s41, s5 // (op 429) reg expression
s_mov_b32 s42, s0 // (op 429) reg expression
s_mov_b32 s43, 131072 // (op 429) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 429): s0, s1

 // (op 422) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(429) END
 // (op 422) NOP(418) BEGIN
 // (op 418) generate({})
 // (op 418) end: generate({})
 // (op 422) NOP(418) END
 // (op 422) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(434) BEGIN
 // (op 434) Assign dim(455) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 434) Assign stride expression
 // (op 422) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(434) END
 // (op 422) NOP(419) BEGIN
 // (op 419) generate({})
 // (op 419) end: generate({})
 // (op 422) NOP(419) END
 // (op 422) Assign VGPR 0:U64(437) BEGIN
 // (op 437) Assign dim(457) = 0:U64
 // (op 437) Assign stride expression
 // (op 422) Assign VGPR 0:U64(437) END
 // (op 422) NOP(485) BEGIN
 // (op 485) generate({})
 // (op 485) end: generate({})
 // (op 422) NOP(485) END
 // (op 422) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(494) BEGIN
 // (op 494) Assign dim(469) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 469: v**UNALLOCATED**
 // (op 494) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag469: VGPR Value: UInt32 x 1: (unallocated)
 // (op 494) reg expression
 // (op 494) reg expression
 // (op 494) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
 // (op 494) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 494): v10
v_and_b32 v10, -4, v1 // (op 494) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 494): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 494) 
 // (op 494) BEGIN: Flatten[1]
 // (op 494) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
 // (op 494) convert
// Allocated : 1 VGPR (Value: UInt32) (op 494): v11
v_sub_u32 v11, v1, v10 // (op 494) 
 // (op 494) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 494): v10

 // (op 494) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 494) BEGIN: Tile
 // (op 494) {Tile: ShiftL({Flatten[1]: v11:U32}, 5:U32)U32}
 // (op 494) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 494)         value (Flatten[1]: VGPR Value: UInt32 x 1: v11) 
 // (op 494)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 SGPR (Value: UInt32) (op 494): s14
s_lshr_b32 s14, s0, 1 // (op 494) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 494): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 494): v10
v_lshlrev_b32 v10, 5, v11 // (op 494) 
 // (op 494) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 494): v11

 // (op 494) reg expression
 // (op 494) BEGIN: Tile
 // (op 494) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 494): v11
v_lshl_add_u32 v11, s14, 8, v10 // (op 494) 
 // (op 494) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 494): v10

 // Freeing : 1 SGPR (Value: UInt32) (op 494): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 494): v14, v15
v_mov_b32 v15, 0 // (op 494) convert
v_mov_b32 v14, v11 // (op 494) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 494): v11

 // (op 494) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 494): v10
v_lshrrev_b32 v10, 2, v1 // (op 494) 
 // (op 494) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 494): v11
v_lshl_add_u32 v11, s17, 5, v10 // (op 494) 
 // (op 494) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 494): v10

 // (op 494) Multiply({Tile: v11:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 494) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 494)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 494)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 494): v16, v17
v_mul_lo_u32 v17, v11, s9 // (op 494) most significant: low of low * high
 // (op 494) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 494): v10
v_mul_hi_u32 v10, v11, s8 // (op 494) most significant: high of low * low
v_mul_lo_u32 v16, v11, s8 // (op 494) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 494) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 494): v10

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 494): v11

 // (op 494) BEGIN: Split
 // (op 494) {Split: Add({Tile: v[14:15]:U64}, v[16:17]:I64)U64}
 // (op 494) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 494)      lhs (Tile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 494)      rhs (VGPR Value: Int64 x 1: v[16:17])
// Allocated : 2 VGPRs (Value: UInt64) (op 494): v10, v11
// Allocated : 2 SGPRs (Value: Bool64) (op 494): s0, s1
v_add_co_u32 v10, s[0:1], v14, v16 // (op 494) least significant half
v_addc_co_u32 v11, s[0:1], v15, v17, s[0:1] // (op 494) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 494): s0, s1

 // (op 494) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 494): v16, v17

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 494): v14, v15

 // (op 494) Multiply({Split: v[10:11]:U64}, 6:U32)U64
 // (op 494) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 494)           lhs (Split: VGPR Value: UInt64 x 1: v[10:11]) 
 // (op 494)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 494) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 494): v14, v15
v_mul_lo_u32 v15, v11, 6 // (op 494) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 494): v16
v_mul_hi_u32 v16, v10, 6 // (op 494) most significant: high of low * low
v_mul_lo_u32 v14, v10, 6 // (op 494) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 494) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 494): v16

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 494): v10, v11

 // (op 494) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 494): v10, v11
v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 494) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 494): v14, v15

// Allocated DataFlowTag469: 1 VGPR (Value: UInt32) (op 494): v14
v_mov_b32 v14, v10 // (op 494) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 494): v10, v11

 // (op 422) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -4:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(494) END
 // (op 422) Assign VGPR 96:U64(495) BEGIN
 // (op 495) Assign dim(470) = 96:U64
 // (op 495) Assign stride expression
 // (op 422) Assign VGPR 96:U64(495) END
 // (op 422) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(496) BEGIN
 // (op 496) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 496) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // (op 496) reg expression
 // (op 496) reg expression
 // (op 496) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 496) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 496)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 496)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 496) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 496): s0, s1
s_mul_i32 s1, s13, 6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 496) least significant
// Allocated : 1 SGPR (Value: Int32) (op 496): s14
s_mul_hi_u32 s14, s12, 6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 496) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 496): s14

 // (op 496) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 496) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 496)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 496)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 496): s14, s15
s_mul_i32 s15, s8, s1 // (op 496) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 496): s18
s_mul_i32 s18, s9, s0 // (op 496) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 496) least significant
// Allocated : 1 SGPR (Value: Int32) (op 496): s19
s_mul_hi_u32 s19, s8, s0 // (op 496) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 496) most significant: sum
s_add_u32 s15, s15, s19 // (op 496) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 496): s19

 // Freeing : 1 SGPR (Value: Int32) (op 496): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 496): s0, s1

 // (op 496) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 496): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 496) 
 // Freeing : 2 SGPRs (Value: Int64) (op 496): s14, s15

 // (op 496) reg expression
 // (op 496) reg expression
s_mov_b32 s44, s10 // (op 496) reg expression
s_mov_b32 s45, s11 // (op 496) reg expression
s_mov_b32 s46, s0 // (op 496) reg expression
s_mov_b32 s47, 131072 // (op 496) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 496): s0, s1

 // (op 422) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(496) END
 // (op 422) NOP(487) BEGIN
 // (op 487) generate({})
 // (op 487) end: generate({})
 // (op 422) NOP(487) END
 // (op 422) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(501) BEGIN
 // (op 501) Assign dim(474) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 501) Assign stride expression
 // (op 422) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(501) END
 // (op 422) NOP(488) BEGIN
 // (op 488) generate({})
 // (op 488) end: generate({})
 // (op 422) NOP(488) END
 // (op 422) Assign VGPR 0:U64(504) BEGIN
 // (op 504) Assign dim(476) = 0:U64
 // (op 504) Assign stride expression
 // (op 422) Assign VGPR 0:U64(504) END
 // (op 422) NOP(549) BEGIN
 // (op 549) generate({})
 // (op 549) end: generate({})
 // (op 422) NOP(549) END
 // (op 422) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32})U32}(558) BEGIN
 // (op 558) Assign dim(488) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32})U32}
 // tag 488: v**UNALLOCATED**
 // (op 558) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32})U32} into DataFlowTag488: VGPR Value: UInt32 x 1: (unallocated)
 // (op 558) reg expression
 // (op 558) reg expression
 // (op 558) reg expression
 // (op 558) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 558) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 558): v10
v_lshrrev_b32 v10, 7, v1 // (op 558) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 558): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 558) 
 // (op 558) BEGIN: Flatten[1]
 // (op 558) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 558) convert
// Allocated : 1 VGPR (Value: UInt32) (op 558): v11
v_and_b32 v11, 15, v1 // (op 558) 
 // (op 558) END: Flatten[1]
 // (op 558) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 558) BEGIN: Tile
// Allocated : 1 SGPR (Value: UInt32) (op 558): s14
s_lshr_b32 s14, s0, 1 // (op 558) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 558): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 558): v15
v_lshl_add_u32 v15, v10, 4, v11 // (op 558) 
 // (op 558) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 558): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 558): v10

 // (op 558) reg expression
 // (op 558) BEGIN: Tile
 // (op 558) convert
// Allocated : 1 VGPR (Value: UInt32) (op 558): v10
v_lshl_add_u32 v10, s16, 4, v15 // (op 558) 
 // (op 558) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 558): v15

 // (op 558) BEGIN: Flatten[1]
 // (op 558) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 558): v11
v_and_b32 v11, 63, v1 // (op 558) 
 // (op 558) END: Flatten[1]
 // (op 558) LogicalShiftR({Flatten[1]: v11:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 558): v15
v_lshrrev_b32 v15, 4, v11 // (op 558) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 558): v11

 // (op 558) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 558): v11
v_lshl_add_u32 v11, s14, 3, v15 // (op 558) 
 // (op 558) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 558): v15

 // Freeing : 1 SGPR (Value: UInt32) (op 558): s14

 // (op 558) BEGIN: Split
// Allocated DataFlowTag488: 1 VGPR (Value: UInt32) (op 558): v15
v_mul_lo_u32 v15, v10, s20 // (op 558) 
v_add_u32 v15, v15, v11 // (op 558) 
 // (op 558) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 558): v11

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 558): v10

 // (op 422) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32})U32}(558) END
 // (op 422) Assign VGPR 4:U64(559) BEGIN
 // (op 559) Assign dim(489) = 4:U64
 // (op 559) Assign stride expression
 // (op 422) Assign VGPR 4:U64(559) END
 // (op 422) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(560) BEGIN
 // (op 560) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 560) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // (op 560) reg expression
 // (op 560) reg expression
 // (op 560) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 560) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 560)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 560)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 560): s0, s1
s_mul_i32 s1, s20, s7 // (op 560) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 560): s14
s_mul_i32 s14, s21, s6 // (op 560) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 560) least significant
// Allocated : 1 SGPR (Value: Int32) (op 560): s15
s_mul_hi_u32 s15, s20, s6 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 560) most significant: sum
s_add_u32 s1, s1, s15 // (op 560) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 560): s15

 // Freeing : 1 SGPR (Value: Int32) (op 560): s14

 // (op 560) reg expression
 // (op 560) reg expression
s_mov_b32 s48, s22 // (op 560) reg expression
s_mov_b32 s49, s23 // (op 560) reg expression
s_mov_b32 s50, s0 // (op 560) reg expression
s_mov_b32 s51, 131072 // (op 560) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 560): s0, s1

 // (op 422) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(560) END
 // (op 422) NOP(551) BEGIN
 // (op 551) generate({})
 // (op 551) end: generate({})
 // (op 422) NOP(551) END
 // (op 422) Assign VGPR 4:U64(565) BEGIN
 // (op 565) Assign dim(493) = 4:U64
 // (op 565) Assign stride expression
 // (op 422) Assign VGPR 4:U64(565) END
 // (op 422) NOP(552) BEGIN
 // (op 552) generate({})
 // (op 552) end: generate({})
 // (op 422) NOP(552) END
 // (op 422) Assign VGPR 1:U64(568) BEGIN
 // (op 568) Assign dim(495) = 1:U64
 // (op 568) Assign stride expression
 // (op 422) Assign VGPR 1:U64(568) END
 // (op 422) NOP(613) BEGIN
 // (op 613) generate({})
 // (op 613) end: generate({})
 // (op 422) NOP(613) END
 // (op 422) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(622) BEGIN
 // (op 622) Assign dim(507) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 507: v**UNALLOCATED**
 // (op 622) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag507: VGPR Value: UInt32 x 1: (unallocated)
 // (op 622) reg expression
 // (op 622) reg expression
 // (op 622) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
 // (op 622) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 622): v10
v_and_b32 v10, -4, v1 // (op 622) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 622): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 622) 
 // (op 622) BEGIN: Flatten[1]Tile
 // (op 622) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
 // (op 622) convert
// Allocated : 1 VGPR (Value: UInt32) (op 622): v11
v_sub_u32 v11, v1, v10 // (op 622) 
 // (op 622) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 622): v10

 // (op 622) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 622) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 622): s14
s_lshr_b32 s14, s0, 1 // (op 622) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 622): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 622): v10
v_lshrrev_b32 v10, 2, v1 // (op 622) 
 // (op 622) reg expression
 // (op 622) BEGIN: Tile
 // (op 622) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 622): v16
v_lshl_add_u32 v16, s14, 3, v11 // (op 622) 
 // (op 622) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 622): v11

 // Freeing : 1 SGPR (Value: UInt32) (op 622): s14

 // (op 622) convert
 // (op 622) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 622): v11
v_lshl_add_u32 v11, s17, 5, v10 // (op 622) 
 // (op 622) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 622): v10

 // (op 622) Multiply({Tile: v11:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 622): v10
v_mul_lo_u32 v10, v11, s24 // (op 622) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 622): v11

 // (op 622) BEGIN: Split
 // (op 622) {Split: Add({Tile: v16:U32}, v10:U32)U32}
// Allocated DataFlowTag507: 1 VGPR (Value: UInt32) (op 622): v11
v_add_u32 v11, v16, v10 // (op 622) 
 // (op 622) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 622): v10

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 622): v16

 // (op 422) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(622) END
 // (op 422) Assign VGPR 4:U64(623) BEGIN
 // (op 623) Assign dim(508) = 4:U64
 // (op 623) Assign stride expression
 // (op 422) Assign VGPR 4:U64(623) END
 // (op 422) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(624) BEGIN
 // (op 624) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 624) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // (op 624) reg expression
 // (op 624) reg expression
 // (op 624) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 624) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 624)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 624)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 624): s0, s1
s_mul_i32 s1, s24, s13 // (op 624) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 624): s14
s_mul_i32 s14, s25, s12 // (op 624) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 624) least significant
// Allocated : 1 SGPR (Value: Int32) (op 624): s15
s_mul_hi_u32 s15, s24, s12 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 624) most significant: sum
s_add_u32 s1, s1, s15 // (op 624) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 624): s15

 // Freeing : 1 SGPR (Value: Int32) (op 624): s14

 // (op 624) reg expression
 // (op 624) reg expression
s_mov_b32 s52, s26 // (op 624) reg expression
s_mov_b32 s53, s27 // (op 624) reg expression
s_mov_b32 s54, s0 // (op 624) reg expression
s_mov_b32 s55, 131072 // (op 624) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 624): s0, s1

 // (op 422) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(624) END
 // (op 422) NOP(615) BEGIN
 // (op 615) generate({})
 // (op 615) end: generate({})
 // (op 422) NOP(615) END
 // (op 422) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(629) BEGIN
 // (op 629) Assign dim(512) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 629) Assign stride expression
 // (op 422) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(629) END
 // (op 422) NOP(616) BEGIN
 // (op 616) generate({})
 // (op 616) end: generate({})
 // (op 422) NOP(616) END
 // (op 422) Assign VGPR 1:U64(632) BEGIN
 // (op 632) Assign dim(514) = 1:U64
 // (op 632) Assign stride expression
 // (op 422) Assign VGPR 1:U64(632) END
 // (op 422) NOP(710) BEGIN
 // (op 710) generate({})
 // (op 710) end: generate({})
 // (op 422) NOP(710) END
 // (op 422) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(715) BEGIN
 // (op 715) Assign dim(536) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 536: v**UNALLOCATED**
 // (op 715) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag536: VGPR Value: UInt32 x 1: (unallocated)
 // (op 715) reg expression
 // (op 715) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 715): v10
v_and_b32 v10, -8, v1 // (op 715) 
 // (op 715) BEGIN: Tile[1]
 // (op 715) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 715): v16
v_sub_u32 v16, v1, v10 // (op 715) 
 // (op 715) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 715): v10

 // (op 715) BEGIN: Flatten
 // (op 715) {Flatten: ShiftL({Tile[1]: v16:U32}, 4:U32)U32}
 // (op 715) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 715)         value (Tile[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 715)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 715): v10
v_lshlrev_b32 v10, 4, v16 // (op 715) 
 // (op 715) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 715): v16

 // (op 715) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 715): v16
v_lshrrev_b32 v16, 3, v1 // (op 715) 
 // (op 715) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 715): v17
v_lshl_add_u32 v17, v16, 7, v10 // (op 715) 
 // (op 715) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 715): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 715): v16

 // (op 715) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 715): v10
v_mul_lo_u32 v10, v17, 6 // (op 715) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 715): v17

 // (op 715) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag536: 1 VGPR (Value: UInt32) (op 715): v16
v_lshrrev_b32 v16, 3, v10 // (op 715) 
 // Freeing : 1 VGPR (Value: UInt32) (op 715): v10

 // (op 422) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(715) END
 // (op 422) Assign VGPR 96:U32(716) BEGIN
 // (op 716) Assign dim(537) = 96:U32
 // (op 716) Assign stride expression
 // (op 422) Assign VGPR 96:U32(716) END
 // (op 422) NOP(711) BEGIN
 // (op 711) generate({})
 // (op 711) end: generate({})
 // (op 422) NOP(711) END
 // (op 422) Assign VGPR 0:U32(720) BEGIN
 // (op 720) Assign dim(539) = 0:U32
 // (op 720) Assign stride expression
 // (op 422) Assign VGPR 0:U32(720) END
 // (op 422) NOP(763) BEGIN
 // (op 763) generate({})
 // (op 763) end: generate({})
 // (op 422) NOP(763) END
 // (op 422) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(768) BEGIN
 // (op 768) Assign dim(552) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 552: v**UNALLOCATED**
 // (op 768) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag552: VGPR Value: UInt32 x 1: (unallocated)
 // (op 768) reg expression
 // (op 768) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 768): v10
v_and_b32 v10, -4, v1 // (op 768) 
 // (op 768) BEGIN: Tile[1]
 // (op 768) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 768): v17
v_sub_u32 v17, v1, v10 // (op 768) 
 // (op 768) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 768): v10

 // (op 768) BEGIN: Flatten
 // (op 768) {Flatten: ShiftL({Tile[1]: v17:U32}, 5:U32)U32}
 // (op 768) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 768)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 768)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 768): v10
v_lshlrev_b32 v10, 5, v17 // (op 768) 
 // (op 768) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 768): v17

 // (op 768) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 768): v17
v_lshrrev_b32 v17, 2, v1 // (op 768) 
 // (op 768) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 768): v18
v_lshl_add_u32 v18, v17, 7, v10 // (op 768) 
 // (op 768) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 768): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 768): v17

 // (op 768) Multiply({Flatten: v18:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 768): v10
v_mul_lo_u32 v10, v18, 6 // (op 768) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 768): v18

 // (op 768) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag552: 1 VGPR (Value: UInt32) (op 768): v17
v_lshrrev_b32 v17, 3, v10 // (op 768) 
 // Freeing : 1 VGPR (Value: UInt32) (op 768): v10

 // (op 422) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -4:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(768) END
 // (op 422) Assign VGPR 96:U32(769) BEGIN
 // (op 769) Assign dim(553) = 96:U32
 // (op 769) Assign stride expression
 // (op 422) Assign VGPR 96:U32(769) END
 // (op 422) NOP(764) BEGIN
 // (op 764) generate({})
 // (op 764) end: generate({})
 // (op 422) NOP(764) END
 // (op 422) Assign VGPR 0:U32(773) BEGIN
 // (op 773) Assign dim(555) = 0:U32
 // (op 773) Assign stride expression
 // (op 422) Assign VGPR 0:U32(773) END
 // (op 422) NOP(816) BEGIN
 // (op 816) generate({})
 // (op 816) end: generate({})
 // (op 422) NOP(816) END
 // (op 422) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(821) BEGIN
 // (op 821) Assign dim(568) = {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}
 // tag 568: v**UNALLOCATED**
 // (op 821) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32} into DataFlowTag568: VGPR Value: UInt32 x 1: (unallocated)
 // (op 821) reg expression
 // (op 821) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 821): v10
v_and_b32 v10, -4, v1 // (op 821) 
 // (op 821) BEGIN: Tile[1]Flatten
 // (op 821) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 821): v18
v_sub_u32 v18, v1, v10 // (op 821) 
 // (op 821) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 821): v10

 // (op 821) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 821): v10
v_and_b32 v10, 4294967292, v1 // (op 821) 
 // (op 821) BEGIN: Flatten
 // (op 821) {Flatten: Add(v10:U32, {Tile[1]Flatten: v18:U32})U32}
// Allocated DataFlowTag568: 1 VGPR (Value: UInt32) (op 821): v19
v_add_u32 v19, v10, v18 // (op 821) 
 // (op 821) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 821): v18

 // Freeing : 1 VGPR (Value: UInt32) (op 821): v10

 // (op 422) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(821) END
 // (op 422) Assign VGPR 4:U32(822) BEGIN
 // (op 822) Assign dim(569) = 4:U32
 // (op 822) Assign stride expression
 // (op 422) Assign VGPR 4:U32(822) END
 // (op 422) NOP(817) BEGIN
 // (op 817) generate({})
 // (op 817) end: generate({})
 // (op 422) NOP(817) END
 // (op 422) Assign VGPR 1:U32(826) BEGIN
 // (op 826) Assign dim(571) = 1:U32
 // (op 826) Assign stride expression
 // (op 422) Assign VGPR 1:U32(826) END
 // (op 422) ForLoopOp KLoopTail: LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL(161) BEGIN
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

 // Freeing : 1 VGPR (Value: E8M0) (op 351): v29

 // Freeing : 6 VGPRs (Value: FP6x16) (op 355): v30, v31, v32, v33, v34, v35

 // Freeing : 1 VGPR (Value: E8M0) (op 358): v13

 // Freeing : 3 VGPRs (Value: FP6x16) (op 361): v36, v37, v38

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 161) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 161) 
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 161) 
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161:
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
v_mov_b32 v10, v12 // (op 166) 
 // (op 166) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: FP6x16) (op 166): v20, v21, v22
buffer_load_dwordx3 v[20:22], v10, s[40:43], 0 offen // (op 166) Load value
// VMEM: Expected complete at 4447 (current 3967)
 // (op 206) LoadTiled Value: FP6x16(166) END
 // (op 206) Deallocate{}(1071) BEGIN
 // (op 1071) Deallocate 454
 // Freeing Offset454: 1 VGPR (Value: UInt32) (op 166): v10

 // Deleting tag 454
 // (op 1071) Deallocate 456
 // Deleting tag 456
 // (op 206) Deallocate{}(1071) END
 // (op 206) Assign VGPR Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32(417) BEGIN
 // (op 417) Assign dim(450) = Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32
 // (op 417) Generate Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32 into DataFlowTag450: VGPR Value: UInt32 x 1: v12
 // (op 417) reg expression
 // (op 417) Add({DataFlowTag450: v12:U32}, 96:U32)U32
v_add_u32 v12, 96, v12 // (op 417) 
 // (op 206) Assign VGPR Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32(417) END
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
 // (op 161) Barrier(994) BEGIN
s_barrier  // (op 994) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 161) Barrier(994) END
// Allocated Offset473: 1 VGPR (Value: UInt32) (op 179): v10
v_mov_b32 v10, v14 // (op 179) 
 // (op 179) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[20:22]
 // 	info.rowOffsetReg = DataFlowTag536: VGPR Value: UInt32 x 1: v16
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 179): v24, v25, v26, v27, v28, v29
buffer_load_dwordx4 v[24:27], v10, s[44:47], 0 offen // (op 179) Load value
// VMEM: Expected complete at 4451 (current 3971)
buffer_load_dwordx2 v[28:29], v10, s[44:47], 0 offen offset:16 // (op 179) Load value
// VMEM: Expected complete at 4452 (current 3972)
 // (op 214) LoadTiled Value: FP6x16(179) END
 // (op 214) Deallocate{}(1083) BEGIN
 // (op 1083) Deallocate 475
 // Deleting tag 475
 // (op 1083) Deallocate 473
 // Freeing Offset473: 1 VGPR (Value: UInt32) (op 179): v10

 // Deleting tag 473
 // (op 214) Deallocate{}(1083) END
 // (op 214) Assign VGPR Add(DataFlowTag(469)U32, Convert(DataFlowTag(470)U64)U32)U32(486) BEGIN
 // (op 486) Assign dim(469) = Add(DataFlowTag(469)U32, Convert(DataFlowTag(470)U64)U32)U32
 // (op 486) Generate Add(DataFlowTag(469)U32, Convert(DataFlowTag(470)U64)U32)U32 into DataFlowTag469: VGPR Value: UInt32 x 1: v14
 // (op 486) reg expression
 // (op 486) Add({DataFlowTag469: v14:U32}, 96:U32)U32
v_add_u32 v14, 96, v14 // (op 486) 
 // (op 214) Assign VGPR Add(DataFlowTag(469)U32, Convert(DataFlowTag(470)U64)U32)U32(486) END
 // (op 214) end: generate({179})
 // (op 161) SetCoordinate(214) END
s_waitcnt vmcnt(2)
ds_write_b96 v16, v[20:22] offset:3200 // (op 167) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 4169 (current 4074)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 161) StoreLDSTile Value: FP6x16(167) END
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
 // (op 161) Deallocate{}(1073) BEGIN
 // (op 1073) Deallocate 416
 // Freeing : 3 VGPRs (Value: FP6x16) (op 166): v20, v21, v22

 // Deleting tag 416
 // (op 1073) Deallocate 538
 // Deleting tag 538
 // (op 161) Deallocate{}(1073) END
// Allocated Offset511: 1 VGPR (Value: UInt32) (op 180): v10
v_mov_b32 v10, v11 // (op 180) 
 // (op 180) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset511: VGPR Value: UInt32 x 1: v10
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
 // (op 180) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 161) Barrier(1007) BEGIN
s_barrier  // (op 1007) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 161) Barrier(1007) END
// Allocated : 1 VGPR (Value: E8M0) (op 180): v13
buffer_load_ubyte v13, v10, s[52:55], 0 offen offset: 0 // (op 180) Load value
// VMEM: Expected complete at 4932 (current 4452)
 // (op 219) LoadTiled Value: E8M0(180) END
 // (op 219) Deallocate{}(1085) BEGIN
 // (op 1085) Deallocate 511
 // Freeing Offset511: 1 VGPR (Value: UInt32) (op 180): v10

 // Deleting tag 511
 // (op 1085) Deallocate 513
 // Deleting tag 513
 // (op 219) Deallocate{}(1085) END
 // (op 219) Assign VGPR Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32(614) BEGIN
 // (op 614) Assign dim(507) = Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32
 // (op 614) Generate Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32 into DataFlowTag507: VGPR Value: UInt32 x 1: v11
 // (op 614) reg expression
 // (op 614) Add({DataFlowTag507: v11:U32}, 4:U32)U32
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag552: VGPR Value: UInt32 x 1: v17
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

v_add_u32 v11, 4, v11 // (op 614) 
 // (op 219) Assign VGPR Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32(614) END
 // (op 219) end: generate({180})
 // (op 161) SetCoordinate(219) END
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] // (op 177) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 4175 (current 4080)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 161) Barrier(997) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 997) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 161) Barrier(997) END
s_waitcnt vmcnt(1)
ds_write_b64 v17, v[28:29] offset:16 // (op 177) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512, 1536, 1560, 1584, 1608, 1632, 1656, 1680, 1704, 1728, 1752, 1776, 1800, 1824, 1848, 1872, 1896, 1920, 1944, 1968, 1992, 2016, 2040, 2064, 2088, 2112, 2136, 2160, 2184, 2208, 2232, 2256, 2280, 2304, 2328, 2352, 2376, 2400, 2424, 2448, 2472, 2496, 2520, 2544, 2568, 2592, 2616, 2640, 2664, 2688, 2712, 2736, 2760, 2784, 2808, 2832, 2856, 2880, 2904, 2928, 2952, 2976, 3000, 3024, 3048]
// DSMEM: Expected complete at 4268 (current 4173)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 161) StoreLDSTile Value: FP6x16(177) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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

 // (op 161) Deallocate{}(1079) BEGIN
 // (op 1079) Deallocate 554
 // Deleting tag 554
 // (op 1079) Deallocate 418
 // Freeing : 6 VGPRs (Value: FP6x16) (op 179): v24, v25, v26, v27, v28, v29

 // Deleting tag 418
 // (op 161) Deallocate{}(1079) END
// Allocated : 6 VGPRs (Value: FP6x16) (op 169): v20, v21, v22, v23, v24, v25
ds_read_b128 v[20:23], v3 offset:3200 // (op 169) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 4269 (current 4174)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 161) Barrier(1021) BEGIN
s_barrier  // (op 1021) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 161) Barrier(1021) END
ds_read_b64 v[24:25], v3 offset:3216 // (op 169) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 4271 (current 4176)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 3200, Size: 1536)
 // (op 168) LoadLDSTile{Value: FP6x16}(169) END
 // (op 168) end: generate({169})
 // (op 161) SetCoordinate(168) END
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 3072
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
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = DataFlowTag568: VGPR Value: UInt32 x 1: v19
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
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

 // (op 161) Barrier(1011) BEGIN
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1011) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 161) Barrier(1011) END
s_waitcnt vmcnt(0)
ds_write_b8 v19, v13 offset:3072 // (op 178) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 4367 (current 4272)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 161) StoreLDSTile Value: E8M0(178) END
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

 // (op 161) Deallocate{}(1081) BEGIN
 // (op 1081) Deallocate 570
 // Deleting tag 570
 // (op 1081) Deallocate 420
 // Freeing : 1 VGPR (Value: E8M0) (op 180): v13

 // Deleting tag 420
 // (op 161) Deallocate{}(1081) END
// Allocated : 6 VGPRs (Value: FP6x16) (op 171): v26, v27, v28, v29, v30, v31
ds_read_b128 v[26:29], v4 // (op 171) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 4368 (current 4273)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
ds_read_b64 v[30:31], v4 offset:16 // (op 171) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304, 2400, 2496, 2592, 2688, 2784, 2880, 2976, 1560, 1656, 1752, 1848, 1944, 2040, 2136, 2232, 2328, 2424, 2520, 2616, 2712, 2808, 2904, 3000, 1584, 1680, 1776, 1872, 1968, 2064, 2160, 2256, 2352, 2448, 2544, 2640, 2736, 2832, 2928, 3024, 1608, 1704, 1800, 1896, 1992, 2088, 2184, 2280, 2376, 2472, 2568, 2664, 2760, 2856, 2952, 3048]
// DSMEM: Expected complete at 4369 (current 4274)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 3072)
 // (op 170) LoadLDSTile{Value: FP6x16}(171) END
 // (op 170) end: generate({171})
 // (op 161) SetCoordinate(170) END
 // (op 161) SetCoordinate(172) BEGIN
 // (op 172) generate({})
 // (op 172) end: generate({})
 // (op 172) generate({211})
 // (op 172) SetCoordinate(211) BEGIN
 // (op 211) generate({})
 // (op 211) end: generate({})
 // (op 211) generate({173})
 // (op 211) LoadTiled Value: E8M0(173) BEGIN
 // (op 173) GEN: loadMacroTileWAVE OP173 WaveTile 201
 // (op 173) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 173) Generate 1:U64 into nullptr
 // tag 426: v**UNALLOCATED**
 // (op 173) 	Dir = Load
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
// Allocated Offset492: 1 VGPR (Value: UInt32) (op 173): v10
v_mov_b32 v10, v15 // (op 173) 
 // (op 173) LSTInfo {
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
 // (op 173) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 173): v13
buffer_load_ubyte v13, v10, s[48:51], 0 offen offset: 0 // (op 173) Load value
// VMEM: Expected complete at 5418 (current 4938)
 // (op 211) LoadTiled Value: E8M0(173) END
 // (op 211) Deallocate{}(1075) BEGIN
 // (op 1075) Deallocate 492
 // Freeing Offset492: 1 VGPR (Value: UInt32) (op 173): v10

 // Deleting tag 492
 // (op 1075) Deallocate 494
 // Deleting tag 494
 // (op 211) Deallocate{}(1075) END
 // (op 211) Assign VGPR Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32(550) BEGIN
 // (op 550) Assign dim(488) = Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32
 // (op 550) Generate Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32 into DataFlowTag488: VGPR Value: UInt32 x 1: v15
 // (op 550) reg expression
 // (op 550) Add({DataFlowTag488: v15:U32}, 4:U32)U32
v_add_u32 v15, 4, v15 // (op 550) 
 // (op 211) Assign VGPR Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32(550) END
 // (op 211) end: generate({173})
 // (op 172) SetCoordinate(211) END
 // (op 172) end: generate({211})
 // (op 161) SetCoordinate(172) END
 // (op 161) Barrier(1025) BEGIN
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1025) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 161) Barrier(1025) END
 // (op 161) SetCoordinate(174) BEGIN
 // (op 174) generate({})
 // (op 174) end: generate({})
 // (op 174) generate({175})
 // (op 174) LoadLDSTile{Value: E8M0}(175) BEGIN
 // (op 175) GEN: loadMacroTileWAVELDS OP 175 LDS 32 WaveTile 261
 // (op 175) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 175) Generate 1:U32 into nullptr
 // tag 428: v**UNALLOCATED**
 // (op 175) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
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

 // (op 175) LSTInfo {
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
 // 	info.offset = Literal Value: UInt32 x 0: 3072
 // 	info.bufDesc = (nullptr)
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

// Allocated : 1 VGPR (Value: E8M0) (op 175): v10
ds_read_u8 v10, v0 offset:3072 // (op 175) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 4465 (current 4370)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3072, Size: 128)
 // (op 174) LoadLDSTile{Value: E8M0}(175) END
 // (op 174) end: generate({175})
 // (op 161) SetCoordinate(174) END
 // (op 161) Multiply(176) BEGIN
 // (op 176) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile16xFP6, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v13, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 176) 
 // (op 161) Multiply(176) END
 // (op 161) Deallocate{}(1077) BEGIN
 // (op 1077) Deallocate 422
 // Freeing : 6 VGPRs (Value: FP6x16) (op 169): v20, v21, v22, v23, v24, v25

 // Deleting tag 422
 // (op 1077) Deallocate 424
 // Freeing : 6 VGPRs (Value: FP6x16) (op 171): v26, v27, v28, v29, v30, v31

 // Deleting tag 424
 // (op 1077) Deallocate 426
 // Freeing : 1 VGPR (Value: E8M0) (op 173): v13

 // Deleting tag 426
 // (op 1077) Deallocate 428
 // Freeing : 1 VGPR (Value: E8M0) (op 175): v10

 // Deleting tag 428
 // (op 161) Deallocate{}(1077) END
 // (op 161) end: generate({163, 206})
 // (op 161) For Loop Increment
 // (op 161) generate({})
 // (op 161) end: generate({})
 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 if true)
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
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161:
 // (op 161) 
 // (op 161) Unlock For Loop
 // Unlock None
 // (op 422) ForLoopOp KLoopTail: LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL(161) END
 // (op 422) Deallocate{}(1069) BEGIN
 // (op 1069) Deallocate 450
 // Freeing DataFlowTag450: 1 VGPR (Value: UInt32) (op 427): v12

 // Deleting tag 450
 // (op 1069) Deallocate 451
 // Deleting tag 451
 // (op 1069) Deallocate 455
 // Deleting tag 455
 // (op 1069) Deallocate 469
 // Freeing DataFlowTag469: 1 VGPR (Value: UInt32) (op 494): v14

 // Deleting tag 469
 // (op 1069) Deallocate 507
 // Freeing DataFlowTag507: 1 VGPR (Value: UInt32) (op 622): v11

 // Deleting tag 507
 // (op 1069) Deallocate 553
 // Deleting tag 553
 // (op 1069) Deallocate 495
 // Deleting tag 495
 // (op 1069) Deallocate 568
 // Freeing DataFlowTag568: 1 VGPR (Value: UInt32) (op 821): v19

 // Deleting tag 568
 // (op 1069) Deallocate 470
 // Deleting tag 470
 // (op 1069) Deallocate 569
 // Deleting tag 569
 // (op 1069) Deallocate 414
 // Freeing DataFlowTag414: 1 SGPR (Value: UInt32) (op 162): s14

 // Deleting tag 414
 // (op 1069) Deallocate 571
 // Deleting tag 571
 // (op 1069) Deallocate 512
 // Deleting tag 512
 // (op 1069) Deallocate 536
 // Freeing DataFlowTag536: 1 VGPR (Value: UInt32) (op 715): v16

 // Deleting tag 536
 // (op 1069) Deallocate 457
 // Deleting tag 457
 // (op 1069) Deallocate 474
 // Deleting tag 474
 // (op 1069) Deallocate 476
 // Deleting tag 476
 // (op 1069) Deallocate 488
 // Freeing DataFlowTag488: 1 VGPR (Value: UInt32) (op 558): v15

 // Deleting tag 488
 // (op 1069) Deallocate 489
 // Deleting tag 489
 // (op 1069) Deallocate 508
 // Deleting tag 508
 // (op 1069) Deallocate 514
 // Deleting tag 514
 // (op 1069) Deallocate 537
 // Deleting tag 537
 // (op 1069) Deallocate 539
 // Deleting tag 539
 // (op 1069) Deallocate 493
 // Deleting tag 493
 // (op 1069) Deallocate 552
 // Freeing DataFlowTag552: 1 VGPR (Value: UInt32) (op 768): v17

 // Deleting tag 552
 // (op 1069) Deallocate 555
 // Deleting tag 555
 // (op 422) Deallocate{}(1069) END
 // (op 422) end: generate({416})
 // (op 422) Unlock Scope 422
 // Unlock None
 // (op 202) Scope(422) END
 // (op 202) end: generate({422})
 // (op 201) SetCoordinate(202) END
 // (op 201) end: generate({202})
 // (op 310) SetCoordinate(201) END
 // (op 310) Deallocate{}(1087) BEGIN
 // (op 1087) Deallocate 138
 // Deleting tag 138
 // (op 1087) Deallocate 395
 // Deleting tag 395
 // (op 1087) Deallocate 72
 // Deleting tag 72
 // (op 1087) Deallocate 472
 // Freeing DataFlowTag472: 4 SGPRs (Buffer: None) (op 519): s44, s45, s46, s47

 // Deleting tag 472
 // (op 1087) Deallocate 16
 // Deleting tag 16
 // (op 1087) Deallocate 397
 // Deleting tag 397
 // (op 1087) Deallocate 491
 // Freeing DataFlowTag491: 4 SGPRs (Buffer: None) (op 583): s48, s49, s50, s51

 // Deleting tag 491
 // (op 1087) Deallocate 234
 // Deleting tag 234
 // (op 1087) Deallocate 453
 // Freeing DataFlowTag453: 4 SGPRs (Buffer: None) (op 455): s40, s41, s42, s43

 // Deleting tag 453
 // (op 1087) Deallocate 200
 // Deleting tag 200
 // (op 1087) Deallocate 393
 // Deleting tag 393
 // (op 1087) Deallocate 399
 // Deleting tag 399
 // (op 1087) Deallocate 578
 // Deleting tag 578
 // (op 1087) Deallocate 562
 // Deleting tag 562
 // (op 1087) Deallocate 546
 // Deleting tag 546
 // (op 1087) Deallocate 32
 // Deleting tag 32
 // (op 1087) Deallocate 87
 // Deleting tag 87
 // (op 1087) Deallocate 249
 // Deleting tag 249
 // (op 1087) Deallocate 24
 // Deleting tag 24
 // (op 1087) Deallocate 153
 // Deleting tag 153
 // (op 1087) Deallocate 1
 // Deleting tag 1
 // (op 1087) Deallocate 4
 // Deleting tag 4
 // (op 1087) Deallocate 3
 // Deleting tag 3
 // (op 1087) Deallocate 2
 // Deleting tag 2
 // (op 1087) Deallocate 510
 // Freeing DataFlowTag510: 4 SGPRs (Buffer: None) (op 647): s52, s53, s54, s55

 // Deleting tag 510
 // (op 310) Deallocate{}(1087) END
 // (op 310) Deallocate{Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_2_stride_1_3, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_4_pointer, Tensor_7_pointer, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer}(1204) BEGIN
 // (op 1204) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1204) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1204) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 1204) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1204) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1204) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 1204) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1204) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1204) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1204) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 310) Deallocate{Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_2_stride_1_3, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_4_pointer, Tensor_7_pointer, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer}(1204) END
 // (op 310) Barrier(1089) BEGIN
s_barrier  // (op 1089) 
 // (op 310) Barrier(1089) END
 // (op 310) end: generate({315})
 // (op 310) Unlock Scope 310
 // Unlock None
 // (op 1) Scope(310) END
 // (op 1) Deallocate{}(1125) BEGIN
 // (op 1125) Deallocate 561
 // Deleting tag 561
 // (op 1125) Deallocate 595
 // Deleting tag 595
 // (op 1125) Deallocate 605
 // Deleting tag 605
 // (op 1125) Deallocate 610
 // Freeing DataFlowTag610: 1 VGPR (Value: UInt32) (op 955): v5

 // Deleting tag 610
 // (op 1125) Deallocate 594
 // Freeing DataFlowTag594: 1 VGPR (Value: UInt32) (op 907): v6

 // Deleting tag 594
 // (op 1125) Deallocate 544
 // Freeing DataFlowTag544: 1 VGPR (Value: UInt32) (op 743): v3

 // Deleting tag 544
 // (op 1125) Deallocate 563
 // Deleting tag 563
 // (op 1125) Deallocate 577
 // Deleting tag 577
 // (op 1125) Deallocate 613
 // Deleting tag 613
 // (op 1125) Deallocate 597
 // Deleting tag 597
 // (op 1125) Deallocate 611
 // Deleting tag 611
 // (op 1125) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 796): v4

 // Deleting tag 560
 // (op 1125) Deallocate 579
 // Deleting tag 579
 // (op 1125) Deallocate 547
 // Deleting tag 547
 // (op 1125) Deallocate 576
 // Freeing DataFlowTag576: 1 VGPR (Value: UInt32) (op 849): v0

 // Deleting tag 576
 // (op 1125) Deallocate 603
 // Deleting tag 603
 // (op 1125) Deallocate 545
 // Deleting tag 545
 // (op 1125) Deallocate 602
 // Freeing DataFlowTag602: 1 VGPR (Value: UInt32) (op 931): v7

 // Deleting tag 602
 // (op 1) Deallocate{}(1125) END
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
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123
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
 // (op 318) NOP(859) BEGIN
 // (op 859) generate({})
 // (op 859) end: generate({})
 // (op 318) NOP(859) END
 // (op 318) NOP(856) BEGIN
 // (op 856) generate({})
 // (op 856) end: generate({})
 // (op 318) NOP(856) END
 // (op 318) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(863) BEGIN
 // (op 863) Assign dim(580) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 580: v**UNALLOCATED**
 // (op 863) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag580: VGPR Value: UInt32 x 1: (unallocated)
 // (op 863) reg expression
 // (op 863) reg expression
 // (op 863) reg expression
 // (op 863) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 863) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 863): v0
v_lshrrev_b32 v0, 7, v1 // (op 863) 
 // (op 863) convert
 // (op 863) BEGIN: Tile[1]
 // (op 863) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 863): v3
v_and_b32 v3, 63, v1 // (op 863) 
 // (op 863) END: Tile[1]
 // (op 863) LogicalShiftR({Tile[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 863): v4
v_lshrrev_b32 v4, 4, v3 // (op 863) 
 // (op 863) BEGIN: Flatten
 // (op 863) {Flatten: ShiftL(v4:U32, 2:U32)U32}
 // (op 863) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 863)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 863)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 863): v5
v_lshlrev_b32 v5, 2, v4 // (op 863) 
 // (op 863) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 863): v4

 // (op 863) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 863): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 863) 
 // (op 863) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 863): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 863): v0

 // (op 863) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 863): v0
v_lshl_add_u32 v0, s16, 4, v4 // (op 863) 
 // (op 863) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 863): v4

 // (op 863) LogicalShiftR({Tile[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 863): v4
v_lshrrev_b32 v4, 2, v3 // (op 863) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 863): v3

 // (op 863) BEGIN: Tile[1]
 // (op 863) {Tile[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 863): v3
v_and_b32 v3, 3, v4 // (op 863) 
 // (op 863) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 863): v4

 // (op 863) BEGIN: Tile[1]
 // (op 863) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 863): v4
v_and_b32 v4, 3, v1 // (op 863) 
 // (op 863) END: Tile[1]
 // (op 863) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 863): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 863) 
 // (op 863) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 863): v4

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 863): v3

 // (op 863) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 863): v3
v_lshrrev_b32 v3, 6, v1 // (op 863) 
 // (op 863) BEGIN: Tile[1]
 // (op 863) {Tile[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 863): v4
v_and_b32 v4, 1, v3 // (op 863) 
 // (op 863) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 863): v3

 // (op 863) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 863): v3
v_lshl_add_u32 v3, v4, 4, v5 // (op 863) 
 // (op 863) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 863): v5

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 863): v4

 // (op 863) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 863): v4
v_lshl_add_u32 v4, s17, 5, v3 // (op 863) 
 // (op 863) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 863): v3

 // (op 863) Multiply({Flatten: v4:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 863): v3
v_mul_lo_u32 v3, v4, s30 // (op 863) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 863): v4

 // (op 863) BEGIN: Join
// Allocated DataFlowTag580: 1 VGPR (Value: UInt32) (op 863): v4
v_add_lshl_u32 v4, v0, v3, 2 // (op 863) 
 // (op 863) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 863): v3

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 863): v0

 // (op 318) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(863) END
 // (op 318) Assign VGPR 64:U64(864) BEGIN
 // (op 864) Assign dim(581) = 64:U64
 // (op 864) Assign stride expression
 // (op 318) Assign VGPR 64:U64(864) END
 // (op 318) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(865) BEGIN
 // (op 865) Assign dim(583) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 583: s**UNALLOCATED**
 // (op 865) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag583: SGPR Buffer: None x 1: (unallocated)
 // (op 865) reg expression
 // (op 865) reg expression
 // (op 865) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 865) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 865)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 865)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 865): s14, s15
s_mul_i32 s15, s30, s13 // (op 865) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 865): s0
s_mul_i32 s0, s31, s12 // (op 865) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 865) least significant
// Allocated : 1 SGPR (Value: Int32) (op 865): s1
s_mul_hi_u32 s1, s30, s12 // (op 865) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 865) most significant: sum
s_add_u32 s15, s15, s1 // (op 865) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 865): s1

 // Freeing : 1 SGPR (Value: Int32) (op 865): s0

 // (op 865) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 865) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 865)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 865)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 865): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 865) 
 // Freeing : 2 SGPRs (Value: Int64) (op 865): s14, s15

 // (op 865) reg expression
 // (op 865) reg expression
// Allocated DataFlowTag583: 4 SGPRs (Buffer: None) (op 865): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 865) reg expression
s_mov_b32 s9, s33 // (op 865) reg expression
s_mov_b32 s10, s0 // (op 865) reg expression
s_mov_b32 s11, 131072 // (op 865) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 865): s0, s1

 // (op 318) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(865) END
 // (op 318) NOP(857) BEGIN
 // (op 857) generate({})
 // (op 857) end: generate({})
 // (op 318) NOP(857) END
 // (op 318) Assign VGPR 4:U64(870) BEGIN
 // (op 870) Assign dim(585) = 4:U64
 // (op 870) Assign stride expression
 // (op 318) Assign VGPR 4:U64(870) END
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
 // 	info.rowOffsetReg = DataFlowTag580: VGPR Value: UInt32 x 1: v4
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

buffer_store_dwordx4 v[10:13], v4, s[8:11], 0 offen // (op 52) Store value
// VMEM: Expected complete at 5947 (current 5467)
 // (op 318) StoreTiled Value: Float(52) END
 // (op 318) Deallocate{}(1053) BEGIN
 // (op 1053) Deallocate 580
 // Freeing DataFlowTag580: 1 VGPR (Value: UInt32) (op 863): v4

 // Deleting tag 580
 // (op 1053) Deallocate 584
 // Deleting tag 584
 // (op 1053) Deallocate 585
 // Deleting tag 585
 // (op 1053) Deallocate 55
 // Freeing DataFlowTag55: 4 VGPRs (Value: Float) (op 46): v10, v11, v12, v13

 // Deleting tag 55
 // (op 1053) Deallocate 581
 // Deleting tag 581
 // (op 318) Deallocate{}(1053) END
 // (op 318) end: generate({321})
 // (op 318) Unlock Scope 318
 // Unlock None
 // (op 123) Scope(318) END
 // (op 123) end: generate({318})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 123) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 123) 
s_branch RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123:
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
 // (op 324) Scope(680) BEGIN
 // (op 680) Lock Scope 680
 // Lock Branch
 // (op 680) generate({677})
 // (op 680) NOP(677) BEGIN
 // (op 677) generate({})
 // (op 677) end: generate({})
 // (op 680) NOP(677) END
 // (op 680) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(685) BEGIN
 // (op 685) Assign dim(526) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}
 // tag 526: v**UNALLOCATED**
 // (op 685) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32} into DataFlowTag526: VGPR Value: UInt32 x 1: (unallocated)
 // (op 685) reg expression
 // (op 685) reg expression
 // (op 685) reg expression
 // (op 685) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 685) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 685): v0
v_lshrrev_b32 v0, 7, v1 // (op 685) 
 // (op 685) convert
 // (op 685) BEGIN: Flatten[1]
 // (op 685) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 685): v3
v_and_b32 v3, 63, v1 // (op 685) 
 // (op 685) END: Flatten[1]
 // (op 685) LogicalShiftR({Flatten[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 685): v4
v_lshrrev_b32 v4, 4, v3 // (op 685) 
 // (op 685) BEGIN: Tile
 // (op 685) {Tile: ShiftL(v4:U32, 2:U32)U32}
 // (op 685) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 685)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 685)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 685): v5
v_lshlrev_b32 v5, 2, v4 // (op 685) 
 // (op 685) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 685): v4

 // (op 685) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 685): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 685) 
 // (op 685) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 685): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 685): v0

 // (op 685) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 685): v0
v_lshl_add_u32 v0, s16, 4, v4 // (op 685) 
 // (op 685) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 685): v4

 // (op 685) LogicalShiftR({Flatten[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 685): v4
v_lshrrev_b32 v4, 2, v3 // (op 685) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 685): v3

 // (op 685) BEGIN: Flatten[1]
 // (op 685) {Flatten[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 685): v3
v_and_b32 v3, 3, v4 // (op 685) 
 // (op 685) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 685): v4

 // (op 685) BEGIN: Flatten[1]
 // (op 685) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 685): v4
v_and_b32 v4, 3, v1 // (op 685) 
 // (op 685) END: Flatten[1]
 // (op 685) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 685): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 685) 
 // (op 685) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 685): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 685): v3

 // (op 685) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 685): v3
v_lshrrev_b32 v3, 6, v1 // (op 685) 
 // (op 685) BEGIN: Tile
 // (op 685) {Tile: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 685): v4
v_and_b32 v4, 1, v3 // (op 685) 
 // (op 685) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 685): v3

 // (op 685) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 685): v3
v_lshl_add_u32 v3, v4, 4, v5 // (op 685) 
 // (op 685) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 685): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 685): v4

 // (op 685) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 685): v4
v_lshl_add_u32 v4, s17, 5, v3 // (op 685) 
 // (op 685) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 685): v3

 // (op 685) Multiply({Tile: v4:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 685): v3
v_mul_lo_u32 v3, v4, s34 // (op 685) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 685): v4

 // (op 685) BEGIN: Split
// Allocated DataFlowTag526: 1 VGPR (Value: UInt32) (op 685): v4
v_add_lshl_u32 v4, v0, v3, 2 // (op 685) 
 // (op 685) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 685): v3

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 685): v0

 // (op 680) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(685) END
 // (op 680) Deallocate{}(1143) BEGIN
 // (op 1143) Deallocate 403
 // Deleting tag 403
 // (op 1143) Deallocate 401
 // Deleting tag 401
 // (op 1143) Deallocate 290
 // Deleting tag 290
 // (op 680) Deallocate{}(1143) END
 // (op 680) Assign VGPR 64:U64(686) BEGIN
 // (op 686) Assign dim(527) = 64:U64
 // (op 686) Assign stride expression
 // (op 680) Assign VGPR 64:U64(686) END
 // (op 680) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(687) BEGIN
 // (op 687) Assign dim(529) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 529: s**UNALLOCATED**
 // (op 687) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag529: SGPR Buffer: None x 1: (unallocated)
 // (op 687) reg expression
 // (op 687) reg expression
 // (op 687) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 687) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 687)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 687)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 687): s14, s15
s_mul_i32 s15, s34, s13 // (op 687) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 687): s0
s_mul_i32 s0, s35, s12 // (op 687) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 687) least significant
// Allocated : 1 SGPR (Value: Int32) (op 687): s1
s_mul_hi_u32 s1, s34, s12 // (op 687) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 687) most significant: sum
s_add_u32 s15, s15, s1 // (op 687) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 687): s1

 // Freeing : 1 SGPR (Value: Int32) (op 687): s0

 // (op 687) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 687) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 687)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 687)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 687): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 687) 
 // Freeing : 2 SGPRs (Value: Int64) (op 687): s14, s15

 // (op 687) reg expression
 // (op 687) reg expression
// Allocated DataFlowTag529: 4 SGPRs (Buffer: None) (op 687): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 687) reg expression
s_mov_b32 s5, s37 // (op 687) reg expression
s_mov_b32 s6, s0 // (op 687) reg expression
s_mov_b32 s7, 131072 // (op 687) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 687): s0, s1

 // (op 680) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(687) END
 // (op 680) Deallocate{Tensor_10_pointer}(1208) BEGIN
 // (op 1208) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 680) Deallocate{Tensor_10_pointer}(1208) END
 // (op 680) NOP(678) BEGIN
 // (op 678) generate({})
 // (op 678) end: generate({})
 // (op 680) NOP(678) END
 // (op 680) Assign VGPR 4:U64(692) BEGIN
 // (op 692) Assign dim(531) = 4:U64
 // (op 692) Assign stride expression
 // (op 680) Assign VGPR 4:U64(692) END
 // (op 680) LoadTiled Value: Float(135) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag526: VGPR Value: UInt32 x 1: v4
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

// Allocated : 4 VGPRs (Value: Float) (op 135): v10, v11, v12, v13
buffer_load_dwordx4 v[10:13], v4, s[4:7], 0 offen // (op 135) Load value
// VMEM: Expected complete at 6458 (current 5978)
 // (op 680) LoadTiled Value: Float(135) END
 // (op 680) Deallocate{Tensor_10_stride_1_10}(1065) BEGIN
 // (op 1065) Deallocate 5
 // Deleting tag 5
 // (op 1065) Deallocate 530
 // Deleting tag 530
 // (op 1065) Deallocate 529
 // Freeing DataFlowTag529: 4 SGPRs (Buffer: None) (op 687): s4, s5, s6, s7

 // Deleting tag 529
 // (op 1065) Deallocate 531
 // Deleting tag 531
 // (op 1065) Deallocate 526
 // Freeing DataFlowTag526: 1 VGPR (Value: UInt32) (op 685): v4

 // Deleting tag 526
 // (op 1065) Deallocate 527
 // Deleting tag 527
 // (op 1065) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 680) Deallocate{Tensor_10_stride_1_10}(1065) END
 // (op 680) end: generate({677})
 // (op 680) Unlock Scope 680
 // Unlock None
 // (op 324) Scope(680) END
// Allocated : 4 VGPRs (Value: Float) (op 132): v4, v5, v6, v7
v_accvgpr_read v4, a0 // (op 132) 
 // (op 324) Assign VGPR Multiply(DataFlowTag(35)NA, DataFlowTag(385)NA)NA(136) BEGIN
 // (op 136) Assign dim(387) = Multiply(DataFlowTag(35)NA, DataFlowTag(385)NA)NA
 // (op 136) Generate Multiply(DataFlowTag(35)NA, DataFlowTag(385)NA)NA into nullptr
 // (op 136) reg expression
 // (op 136) reg expression
 // (op 136) Multiply(v8:S, v[10:13]:S)S
v_accvgpr_read v5, a1 // (op 132) 
v_accvgpr_read v6, a2 // (op 132) 
v_accvgpr_read v7, a3 // (op 132) 
// Allocated : 4 VGPRs (Value: Float) (op 132): v14, v15, v16, v17
v_mul_f32 v14, v9, v4 // (op 132) 
v_mul_f32 v15, v9, v5 // (op 132) 
v_mul_f32 v16, v9, v6 // (op 132) 
v_mul_f32 v17, v9, v7 // (op 132) 
 // Freeing : 4 VGPRs (Value: Float) (op 132): v4, v5, v6, v7

 // (op 324) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(132) END
// Allocated : 4 VGPRs (Value: Float) (op 136): v4, v5, v6, v7
s_waitcnt vmcnt(0)
v_mul_f32 v4, v8, v10 // (op 136) 
v_mul_f32 v5, v8, v11 // (op 136) 
v_mul_f32 v6, v8, v12 // (op 136) 
v_mul_f32 v7, v8, v13 // (op 136) 
 // (op 324) Assign VGPR Multiply(DataFlowTag(35)NA, DataFlowTag(385)NA)NA(136) END
 // (op 324) Deallocate{}(1067) BEGIN
 // (op 1067) Deallocate 385
 // Freeing : 4 VGPRs (Value: Float) (op 135): v10, v11, v12, v13

 // Deleting tag 385
 // (op 324) Deallocate{}(1067) END
 // (op 324) Assign VGPR Add(DataFlowTag(387)NA, DataFlowTag(389)NA)NA(133) BEGIN
 // (op 133) Assign dim(391) = Add(DataFlowTag(387)NA, DataFlowTag(389)NA)NA
 // (op 133) Generate Add(DataFlowTag(387)NA, DataFlowTag(389)NA)NA into nullptr
 // (op 133) reg expression
 // (op 133) reg expression
 // (op 133) Add({DataFlowTag387: v[4:7]:S}, {DataFlowTag389: v[14:17]:S})S
// Allocated : 4 VGPRs (Value: Float) (op 133): v10, v11, v12, v13
v_add_f32 v10, v4, v14 // (op 133) 
v_add_f32 v11, v5, v15 // (op 133) 
v_add_f32 v12, v6, v16 // (op 133) 
v_add_f32 v13, v7, v17 // (op 133) 
 // (op 324) Assign VGPR Add(DataFlowTag(387)NA, DataFlowTag(389)NA)NA(133) END
 // (op 324) Deallocate{}(1061) BEGIN
 // (op 1061) Deallocate 389
 // Freeing DataFlowTag389: 4 VGPRs (Value: Float) (op 132): v14, v15, v16, v17

 // Deleting tag 389
 // (op 1061) Deallocate 387
 // Freeing DataFlowTag387: 4 VGPRs (Value: Float) (op 136): v4, v5, v6, v7

 // Deleting tag 387
 // (op 324) Deallocate{}(1061) END
 // (op 324) NOP(876) BEGIN
 // (op 876) generate({})
 // (op 876) end: generate({})
 // (op 324) NOP(876) END
 // (op 324) NOP(873) BEGIN
 // (op 873) generate({})
 // (op 873) end: generate({})
 // (op 324) NOP(873) END
 // (op 324) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(880) BEGIN
 // (op 880) Assign dim(586) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 586: v**UNALLOCATED**
 // (op 880) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag586: VGPR Value: UInt32 x 1: (unallocated)
 // (op 880) reg expression
 // (op 880) reg expression
 // (op 880) reg expression
 // (op 880) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 880) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 880): v0
v_lshrrev_b32 v0, 7, v1 // (op 880) 
 // (op 880) convert
 // (op 880) BEGIN: Tile[1]
 // (op 880) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 880): v3
v_and_b32 v3, 63, v1 // (op 880) 
 // (op 880) END: Tile[1]
 // (op 880) LogicalShiftR({Tile[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 880): v4
v_lshrrev_b32 v4, 4, v3 // (op 880) 
 // (op 880) BEGIN: Flatten
 // (op 880) {Flatten: ShiftL(v4:U32, 2:U32)U32}
 // (op 880) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 880)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 880)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 880): v5
v_lshlrev_b32 v5, 2, v4 // (op 880) 
 // (op 880) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 880): v4

 // (op 880) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 880): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 880) 
 // (op 880) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 880): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 880): v0

 // (op 880) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 880): v0
v_lshl_add_u32 v0, s16, 4, v4 // (op 880) 
 // (op 880) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 880): v4

 // (op 880) LogicalShiftR({Tile[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 880): v4
v_lshrrev_b32 v4, 2, v3 // (op 880) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 880): v3

 // (op 880) BEGIN: Tile[1]
 // (op 880) {Tile[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 880): v3
v_and_b32 v3, 3, v4 // (op 880) 
 // (op 880) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 880): v4

 // (op 880) BEGIN: Tile[1]
 // (op 880) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 880): v4
v_and_b32 v4, 3, v1 // (op 880) 
 // (op 880) END: Tile[1]
 // (op 880) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 880): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 880) 
 // (op 880) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 880): v4

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 880): v3

 // (op 880) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 880): v3
v_lshrrev_b32 v3, 6, v1 // (op 880) 
 // (op 880) BEGIN: Tile[1]
 // (op 880) {Tile[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 880): v4
v_and_b32 v4, 1, v3 // (op 880) 
 // (op 880) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 880): v3

 // (op 880) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 880): v3
v_lshl_add_u32 v3, v4, 4, v5 // (op 880) 
 // (op 880) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 880): v5

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 880): v4

 // (op 880) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 880): v4
v_lshl_add_u32 v4, s17, 5, v3 // (op 880) 
 // (op 880) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 880): v3

 // (op 880) Multiply({Flatten: v4:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 880): v3
v_mul_lo_u32 v3, v4, s30 // (op 880) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 880): v4

 // (op 880) BEGIN: Join
// Allocated DataFlowTag586: 1 VGPR (Value: UInt32) (op 880): v4
v_add_lshl_u32 v4, v0, v3, 2 // (op 880) 
 // (op 880) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 880): v3

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 880): v0

 // (op 324) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(880) END
 // (op 324) Assign VGPR 64:U64(881) BEGIN
 // (op 881) Assign dim(587) = 64:U64
 // (op 881) Assign stride expression
 // (op 324) Assign VGPR 64:U64(881) END
 // (op 324) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(882) BEGIN
 // (op 882) Assign dim(583) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 882) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag583: SGPR Buffer: None x 1: s[8:11]
 // (op 882) reg expression
 // (op 882) reg expression
 // (op 882) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 882) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 882)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 882)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 882): s14, s15
s_mul_i32 s15, s30, s13 // (op 882) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 882): s0
s_mul_i32 s0, s31, s12 // (op 882) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 882) least significant
// Allocated : 1 SGPR (Value: Int32) (op 882): s1
s_mul_hi_u32 s1, s30, s12 // (op 882) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 882) most significant: sum
s_add_u32 s15, s15, s1 // (op 882) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 882): s1

 // Freeing : 1 SGPR (Value: Int32) (op 882): s0

 // (op 882) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 882) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 882)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 882)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 882): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 882) 
 // Freeing : 2 SGPRs (Value: Int64) (op 882): s14, s15

 // (op 882) reg expression
 // (op 882) reg expression
s_mov_b32 s8, s32 // (op 882) reg expression
s_mov_b32 s9, s33 // (op 882) reg expression
s_mov_b32 s10, s0 // (op 882) reg expression
s_mov_b32 s11, 131072 // (op 882) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 882): s0, s1

 // (op 324) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(882) END
 // (op 324) NOP(874) BEGIN
 // (op 874) generate({})
 // (op 874) end: generate({})
 // (op 324) NOP(874) END
 // (op 324) Assign VGPR 4:U64(887) BEGIN
 // (op 887) Assign dim(589) = 4:U64
 // (op 887) Assign stride expression
 // (op 324) Assign VGPR 4:U64(887) END
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
 // 	info.data = DataFlowTag391: VGPR Value: Float x 4: v[10:13]
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
 // 	info.data = DataFlowTag391: VGPR Value: Float x 4: v[10:13]
 // 	info.rowOffsetReg = DataFlowTag586: VGPR Value: UInt32 x 1: v4
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

buffer_store_dwordx4 v[10:13], v4, s[8:11], 0 offen // (op 134) Store value
// VMEM: Expected complete at 6975 (current 6495)
 // (op 324) StoreTiled Value: Float(134) END
 // (op 324) Deallocate{}(1063) BEGIN
 // (op 1063) Deallocate 586
 // Freeing DataFlowTag586: 1 VGPR (Value: UInt32) (op 880): v4

 // Deleting tag 586
 // (op 1063) Deallocate 587
 // Deleting tag 587
 // (op 1063) Deallocate 391
 // Freeing DataFlowTag391: 4 VGPRs (Value: Float) (op 133): v10, v11, v12, v13

 // Deleting tag 391
 // (op 1063) Deallocate 588
 // Deleting tag 588
 // (op 1063) Deallocate 589
 // Deleting tag 589
 // (op 324) Deallocate{}(1063) END
 // (op 324) end: generate({327})
 // (op 324) Unlock Scope 324
 // Unlock None
 // (op 123) Scope(324) END
 // (op 123) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1127) BEGIN
 // (op 1127) Deallocate 405
 // Deleting tag 405
 // (op 1127) Deallocate 407
 // Deleting tag 407
 // (op 1127) Deallocate 330
 // Deleting tag 330
 // (op 1127) Deallocate 583
 // Freeing DataFlowTag583: 4 SGPRs (Buffer: None) (op 865): s8, s9, s10, s11

 // Deleting tag 583
 // (op 1127) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 1127) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 123) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1127) END
 // (op 123) end: generate({324})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 123) 
RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123:
 // (op 123) 
 // (op 123) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(35)S)BL(123) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1059) BEGIN
 // (op 1059) Deallocate 51
 // Freeing DataFlowTag51: 4 ACCVGPRs (Value: Float) (op 59): a0, a1, a2, a3

 // Deleting tag 51
 // (op 1059) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 39): v9

 // Deleting tag 44
 // (op 1059) Deallocate 35
 // Freeing : 1 VGPR (Value: Float) (op 2): v8

 // Deleting tag 35
 // (op 1059) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1059) END
 // (op 1) end: generate({739, 792, 845, 903, 927, 951})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
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
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
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
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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

