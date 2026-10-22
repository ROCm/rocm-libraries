// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.FP6.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.FP8.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
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
 // Tensor_2_pointer: PointerGlobal: FP8(offset: 40, size: 8, read_write)
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
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP8, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
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
.globl RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
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
RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:
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
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP8, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
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
 // Tensor_2_pointer: SGPR PointerGlobal: FP8 x 1: s[10:11]
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
 // (op 1) generate({763, 815, 868, 926, 965, 989})
 // (op 1) NOP(763) BEGIN
 // (op 763) generate({})
 // (op 763) end: generate({})
 // (op 1) NOP(763) END
 // (op 1) NOP(815) BEGIN
 // (op 815) generate({})
 // (op 815) end: generate({})
 // (op 1) NOP(815) END
 // (op 1) NOP(868) BEGIN
 // (op 868) generate({})
 // (op 868) end: generate({})
 // (op 1) NOP(868) END
 // (op 1) NOP(926) BEGIN
 // (op 926) generate({})
 // (op 926) end: generate({})
 // (op 1) NOP(926) END
 // (op 1) NOP(965) BEGIN
 // (op 965) generate({})
 // (op 965) end: generate({})
 // (op 1) NOP(965) END
 // (op 1) NOP(989) BEGIN
 // (op 989) generate({})
 // (op 989) end: generate({})
 // (op 1) NOP(989) END
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(767) BEGIN
 // (op 767) Assign dim(544) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 544: v**UNALLOCATED**
 // (op 767) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag544: VGPR Value: UInt32 x 1: (unallocated)
 // (op 767) reg expression
 // (op 767) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4294967280:U32)U32})U32}(819) BEGIN
 // (op 819) Assign dim(560) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4294967280:U32)U32})U32}
 // tag 560: v**UNALLOCATED**
 // (op 819) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4294967280:U32)U32})U32} into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 819) reg expression
 // (op 819) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}(872) BEGIN
 // (op 872) Assign dim(576) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 576: v**UNALLOCATED**
 // (op 872) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag576: VGPR Value: UInt32 x 1: (unallocated)
 // (op 872) reg expression
 // (op 872) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(930) BEGIN
 // (op 930) Assign dim(594) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 594: v**UNALLOCATED**
 // (op 930) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag594: VGPR Value: UInt32 x 1: (unallocated)
 // (op 930) reg expression
 // (op 930) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4294967280:U32)U32})U32}(969) BEGIN
 // (op 969) Assign dim(606) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4294967280:U32)U32})U32}
 // tag 606: v**UNALLOCATED**
 // (op 969) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4294967280:U32)U32})U32} into DataFlowTag606: VGPR Value: UInt32 x 1: (unallocated)
 // (op 969) reg expression
 // (op 969) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}(993) BEGIN
 // (op 993) Assign dim(614) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 614: v**UNALLOCATED**
 // (op 993) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag614: VGPR Value: UInt32 x 1: (unallocated)
 // (op 993) reg expression
 // (op 993) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 767): v0
v_lshrrev_b32 v0, 7, v1 // (op 767) 
 // (op 767) BEGIN: Flatten[1]Tile
 // (op 767) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 767): v3
v_and_b32 v3, 15, v1 // (op 767) 
 // (op 767) END: Flatten[1]Tile
 // (op 767) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 767): v4
v_lshl_add_u32 v4, v0, 4, v3 // (op 767) 
 // (op 767) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 767): v3

 // Freeing : 1 VGPR (Value: UInt32) (op 767): v0

 // (op 767) BEGIN: Flatten[1]
 // (op 767) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 767): v0
v_and_b32 v0, 63, v1 // (op 767) 
 // (op 767) END: Flatten[1]
 // (op 767) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 767): v3
v_lshrrev_b32 v3, 4, v0 // (op 767) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 767): v0

 // (op 767) BEGIN: TileTile
 // (op 767) {TileTile: ShiftL(v3:U32, 5:U32)U32}
 // (op 767) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 767)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 767)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 767): v0
v_lshlrev_b32 v0, 5, v3 // (op 767) 
 // (op 767) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 767): v3

 // (op 767) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 767): v3
v_lshl_add_u32 v3, v4, 7, v0 // (op 767) 
 // (op 767) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 767): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 767): v4

 // (op 767) Multiply({Tile: v3:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 767): v0
v_mul_lo_u32 v0, v3, 6 // (op 767) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 767): v3

 // (op 767) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag544: 1 VGPR (Value: UInt32) (op 767): v3
v_lshrrev_b32 v3, 3, v0 // (op 767) 
 // Freeing : 1 VGPR (Value: UInt32) (op 767): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(767) END
// Allocated : 1 VGPR (Value: UInt32) (op 819): v0
v_lshrrev_b32 v0, 6, v1 // (op 819) 
 // (op 819) BEGIN: Tile
 // (op 819) {Tile: BitwiseAnd(v0:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 819): v4
v_and_b32 v4, 1, v0 // (op 819) 
 // (op 819) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 819): v0

 // (op 819) BEGIN: Flatten[1]Tile
 // (op 819) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 819): v0
v_and_b32 v0, 15, v1 // (op 819) 
 // (op 819) END: Flatten[1]Tile
 // (op 819) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 819): v5
v_lshl_add_u32 v5, v4, 4, v0 // (op 819) 
 // (op 819) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 819): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 819): v4

 // (op 819) BEGIN: TileTile
 // (op 819) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 48:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 819): v0
v_and_b32 v0, 48, v1 // (op 819) 
 // (op 819) END: TileTile
 // (op 819) BEGIN: Tile
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 819): v4
v_lshl_add_u32 v4, v5, 7, v0 // (op 819) 
 // (op 819) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 819): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 819): v5

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4294967280:U32)U32})U32}(819) END
// Allocated : 1 VGPR (Value: UInt32) (op 872): v0
v_lshrrev_b32 v0, 6, v1 // (op 872) 
 // (op 872) BEGIN: Tile
 // (op 872) {Tile: BitwiseAnd(v0:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 872): v5
v_and_b32 v5, 1, v0 // (op 872) 
 // (op 872) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 872): v0

 // (op 872) BEGIN: Flatten[1]
 // (op 872) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 872): v0
v_and_b32 v0, 15, v1 // (op 872) 
 // (op 872) END: Flatten[1]
 // (op 872) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 872): v6
v_lshl_add_u32 v6, v5, 4, v0 // (op 872) 
 // (op 872) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 872): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 872): v5

 // (op 872) BEGIN: Flatten[1]
 // (op 872) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 872): v0
v_and_b32 v0, 63, v1 // (op 872) 
 // (op 872) END: Flatten[1]
 // (op 872) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 872): v5
v_lshrrev_b32 v5, 4, v0 // (op 872) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 872): v0

 // (op 872) BEGIN: Tile
// Allocated DataFlowTag576: 1 VGPR (Value: UInt32) (op 872): v0
v_lshl_add_u32 v0, v6, 2, v5 // (op 872) 
 // (op 872) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 872): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 872): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}(872) END
// Allocated : 1 VGPR (Value: UInt32) (op 930): v5
v_lshrrev_b32 v5, 7, v1 // (op 930) 
 // (op 930) BEGIN: Flatten[1]Tile
 // (op 930) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 930): v6
v_and_b32 v6, 15, v1 // (op 930) 
 // (op 930) END: Flatten[1]Tile
 // (op 930) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 930): v7
v_lshl_add_u32 v7, v5, 4, v6 // (op 930) 
 // (op 930) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 930): v6

 // Freeing : 1 VGPR (Value: UInt32) (op 930): v5

 // (op 930) BEGIN: Flatten[1]
 // (op 930) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 930): v5
v_and_b32 v5, 63, v1 // (op 930) 
 // (op 930) END: Flatten[1]
 // (op 930) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 930): v6
v_lshrrev_b32 v6, 4, v5 // (op 930) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 930): v5

 // (op 930) BEGIN: TileTile
 // (op 930) {TileTile: ShiftL(v6:U32, 5:U32)U32}
 // (op 930) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 930)         value (VGPR Value: UInt32 x 1: v6) 
 // (op 930)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 930): v5
v_lshlrev_b32 v5, 5, v6 // (op 930) 
 // (op 930) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 930): v6

 // (op 930) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 930): v6
v_lshl_add_u32 v6, v7, 7, v5 // (op 930) 
 // (op 930) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 930): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 930): v7

 // (op 930) Multiply({Tile: v6:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 930): v5
v_mul_lo_u32 v5, v6, 6 // (op 930) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 930): v6

 // (op 930) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag594: 1 VGPR (Value: UInt32) (op 930): v6
v_lshrrev_b32 v6, 3, v5 // (op 930) 
 // Freeing : 1 VGPR (Value: UInt32) (op 930): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(98)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(98)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(98)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(930) END
// Allocated : 1 VGPR (Value: UInt32) (op 969): v5
v_lshrrev_b32 v5, 6, v1 // (op 969) 
 // (op 969) BEGIN: Tile
 // (op 969) {Tile: BitwiseAnd(v5:U32, 1:U32)U32}
 // (op 1) Deallocate{}(1180) BEGIN
 // (op 1180) Deallocate 98
 // Deleting tag 98
 // (op 1) Deallocate{}(1180) END
// Allocated : 1 VGPR (Value: UInt32) (op 969): v7
v_and_b32 v7, 1, v5 // (op 969) 
 // (op 969) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 969): v5

 // (op 969) BEGIN: Flatten[1]Tile
 // (op 969) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Assign VGPR 96:U32(768) BEGIN
 // (op 768) Assign dim(545) = 96:U32
 // (op 768) Assign stride expression
 // (op 1) Assign VGPR 96:U32(768) END
 // (op 1) Assign VGPR 96:U32(931) BEGIN
 // (op 931) Assign dim(595) = 96:U32
 // (op 931) Assign stride expression
 // (op 1) Assign VGPR 96:U32(931) END
 // (op 1) NOP(764) BEGIN
 // (op 764) generate({})
 // (op 764) end: generate({})
 // (op 1) NOP(764) END
 // (op 1) NOP(927) BEGIN
 // (op 927) generate({})
 // (op 927) end: generate({})
 // (op 1) NOP(927) END
 // (op 1) Assign VGPR 0:U32(772) BEGIN
 // (op 772) Assign dim(547) = 0:U32
 // (op 772) Assign stride expression
 // (op 1) Assign VGPR 0:U32(772) END
 // (op 1) Assign VGPR 0:U32(935) BEGIN
 // (op 935) Assign dim(597) = 0:U32
 // (op 935) Assign stride expression
 // (op 1) Assign VGPR 0:U32(935) END
// Allocated : 1 VGPR (Value: UInt32) (op 969): v5
v_and_b32 v5, 15, v1 // (op 969) 
 // (op 969) END: Flatten[1]Tile
 // (op 969) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 969): v8
v_lshl_add_u32 v8, v7, 4, v5 // (op 969) 
 // (op 969) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 969): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 969): v7

 // (op 969) BEGIN: TileTile
 // (op 969) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 48:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 969): v5
v_and_b32 v5, 48, v1 // (op 969) 
 // (op 969) END: TileTile
 // (op 969) BEGIN: Tile
// Allocated DataFlowTag606: 1 VGPR (Value: UInt32) (op 969): v7
v_lshl_add_u32 v7, v8, 7, v5 // (op 969) 
 // (op 969) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 969): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 969): v8

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(164)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(164)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(164)U32, 63:U32)U32}, 4294967280:U32)U32})U32}(969) END
// Allocated : 1 VGPR (Value: UInt32) (op 993): v5
v_lshrrev_b32 v5, 6, v1 // (op 993) 
 // (op 993) BEGIN: Tile
 // (op 993) {Tile: BitwiseAnd(v5:U32, 1:U32)U32}
 // (op 1) Deallocate{}(1183) BEGIN
 // (op 1183) Deallocate 164
 // Deleting tag 164
 // (op 1) Deallocate{}(1183) END
// Allocated : 1 VGPR (Value: UInt32) (op 993): v8
v_and_b32 v8, 1, v5 // (op 993) 
 // (op 993) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 993): v5

 // (op 993) BEGIN: Flatten[1]
 // (op 993) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Assign VGPR 128:U32(820) BEGIN
 // (op 820) Assign dim(561) = 128:U32
 // (op 820) Assign stride expression
 // (op 1) Assign VGPR 128:U32(820) END
 // (op 1) Assign VGPR 128:U32(970) BEGIN
 // (op 970) Assign dim(607) = 128:U32
 // (op 970) Assign stride expression
 // (op 1) Assign VGPR 128:U32(970) END
 // (op 1) NOP(816) BEGIN
 // (op 816) generate({})
 // (op 816) end: generate({})
 // (op 1) NOP(816) END
 // (op 1) NOP(966) BEGIN
 // (op 966) generate({})
 // (op 966) end: generate({})
 // (op 1) NOP(966) END
 // (op 1) Assign VGPR 1:U32(824) BEGIN
 // (op 824) Assign dim(563) = 1:U32
 // (op 824) Assign stride expression
 // (op 1) Assign VGPR 1:U32(824) END
 // (op 1) Assign VGPR 1:U32(974) BEGIN
 // (op 974) Assign dim(609) = 1:U32
 // (op 974) Assign stride expression
 // (op 1) Assign VGPR 1:U32(974) END
// Allocated : 1 VGPR (Value: UInt32) (op 993): v5
v_and_b32 v5, 15, v1 // (op 993) 
 // (op 993) END: Flatten[1]
 // (op 993) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 993): v9
v_lshl_add_u32 v9, v8, 4, v5 // (op 993) 
 // (op 993) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 993): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 993): v8

 // (op 993) BEGIN: Flatten[1]
 // (op 993) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 993): v5
v_and_b32 v5, 63, v1 // (op 993) 
 // (op 993) END: Flatten[1]
 // (op 993) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 993): v8
v_lshrrev_b32 v8, 4, v5 // (op 993) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 993): v5

 // (op 993) BEGIN: Tile
// Allocated DataFlowTag614: 1 VGPR (Value: UInt32) (op 993): v5
v_lshl_add_u32 v5, v9, 2, v8 // (op 993) 
 // (op 993) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 993): v8

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 993): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(260)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(260)U32, 63:U32)U32}, 4:U32)U32)U32}(993) END
 // (op 1) Deallocate{}(1186) BEGIN
 // (op 1186) Deallocate 260
 // Deleting tag 260
 // (op 1) Deallocate{}(1186) END
 // (op 1) Assign VGPR 4:U32(873) BEGIN
 // (op 873) Assign dim(577) = 4:U32
 // (op 873) Assign stride expression
 // (op 1) Assign VGPR 4:U32(873) END
 // (op 1) Assign VGPR 4:U32(994) BEGIN
 // (op 994) Assign dim(615) = 4:U32
 // (op 994) Assign stride expression
 // (op 1) Assign VGPR 4:U32(994) END
 // (op 1) NOP(869) BEGIN
 // (op 869) generate({})
 // (op 869) end: generate({})
 // (op 1) NOP(869) END
 // (op 1) NOP(990) BEGIN
 // (op 990) generate({})
 // (op 990) end: generate({})
 // (op 1) NOP(990) END
 // (op 1) Assign VGPR 1:U32(877) BEGIN
 // (op 877) Assign dim(579) = 1:U32
 // (op 877) Assign stride expression
 // (op 1) Assign VGPR 1:U32(877) END
 // (op 1) Assign VGPR 1:U32(998) BEGIN
 // (op 998) Assign dim(617) = 1:U32
 // (op 998) Assign stride expression
 // (op 1) Assign VGPR 1:U32(998) END
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
 // (op 1) Deallocate{user_Float_Value_14}(1226) BEGIN
 // (op 1226) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1226) END
 // (op 1) Deallocate{user_Float_Value_12}(1228) BEGIN
 // (op 1228) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1228) END
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
 // (op 310) Scope(469) BEGIN
 // (op 469) Lock Scope 469
 // Lock Branch
 // (op 469) generate({464})
 // (op 469) NOP(464) BEGIN
 // (op 464) generate({})
 // (op 464) end: generate({})
 // (op 469) NOP(464) END
 // (op 469) Assign VGPR 96:U64(474) BEGIN
 // (op 474) Assign dim(458) = 96:U64
 // (op 474) Assign stride expression
 // (op 469) Assign VGPR 96:U64(474) END
 // (op 469) NOP(465) BEGIN
 // (op 465) generate({})
 // (op 465) end: generate({})
 // (op 469) NOP(465) END
 // (op 469) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(477) BEGIN
 // (op 477) Assign dim(459) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 459: v**UNALLOCATED**
 // (op 477) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag459: VGPR Value: UInt32 x 1: (unallocated)
 // (op 477) reg expression
 // (op 477) reg expression
 // (op 477) reg expression
 // (op 477) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 477): v10
v_lshrrev_b32 v10, 3, v1 // (op 477) 
 // (op 477) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 477): v11
v_lshl_add_u32 v11, s16, 4, v10 // (op 477) 
 // (op 477) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 477): v10

 // (op 477) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 477): v10
v_and_b32 v10, -8, v1 // (op 477) 
 // (op 477) BEGIN: Flatten[1]
 // (op 477) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 477): v12
v_sub_u32 v12, v1, v10 // (op 477) 
 // (op 477) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 477): v10

 // (op 477) BEGIN: TileTile
 // (op 477) {TileTile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 477) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 477)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 477)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 477): v10
v_lshlrev_b32 v10, 4, v12 // (op 477) 
 // (op 477) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 477): v12

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 477): v12, v13
v_mov_b32 v13, 0 // (op 477) convert
v_mov_b32 v12, v10 // (op 477) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 477): v10

 // (op 477) BEGIN: Split
 // (op 477) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 477)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 477)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 477): v14, v15
v_mul_lo_u32 v15, v11, s3 // (op 477) most significant: low of low * high
 // (op 477) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 477): v10
v_mul_hi_u32 v10, v11, s2 // (op 477) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 477) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 477) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 477): v10

 // (op 477) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 477)      lhs (VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 477)      rhs (TileTile: VGPR Value: UInt64 x 1: v[12:13])
// Allocated : 2 SGPRs (Value: Bool64) (op 477): s0, s1
v_add_co_u32 v14, s[0:1], v14, v12 // (op 477) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 477) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 477): s0, s1

 // (op 477) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 477): v12, v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 477): v11

 // (op 477) Multiply({Split: v[14:15]:U64}, 6:U32)U64
 // (op 477) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 477)           lhs (Split: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 477)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 477) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 477): v10, v11
v_mul_lo_u32 v11, v15, 6 // (op 477) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 477): v12
v_mul_hi_u32 v12, v14, 6 // (op 477) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 477) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 477) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 477): v12

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 477): v14, v15

 // (op 477) LogicalShiftR(v[10:11]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 477): v12, v13
v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 477) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 477): v10, v11

// Allocated DataFlowTag459: 1 VGPR (Value: UInt32) (op 477): v10
v_mov_b32 v10, v12 // (op 477) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 477): v12, v13

 // (op 469) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(477) END
 // (op 469) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(478) BEGIN
 // (op 478) Assign dim(460) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 478) Assign stride expression
 // (op 469) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(478) END
 // (op 469) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(479) BEGIN
 // (op 479) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 453: s**UNALLOCATED**
 // (op 479) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: (unallocated)
 // (op 479) reg expression
 // (op 479) reg expression
 // (op 479) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 479) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 479)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 479)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 479) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 479): s0, s1
s_mul_i32 s1, s7, 6 // (op 479) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 479) least significant
// Allocated : 1 SGPR (Value: Int32) (op 479): s14
s_mul_hi_u32 s14, s6, 6 // (op 479) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 479) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 479): s14

 // (op 479) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 479) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 479)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 479)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 479): s14, s15
s_mul_i32 s15, s2, s1 // (op 479) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 479): s18
s_mul_i32 s18, s3, s0 // (op 479) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 479) least significant
// Allocated : 1 SGPR (Value: Int32) (op 479): s19
s_mul_hi_u32 s19, s2, s0 // (op 479) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 479) most significant: sum
s_add_u32 s15, s15, s19 // (op 479) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 479): s19

 // Freeing : 1 SGPR (Value: Int32) (op 479): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 479): s0, s1

 // (op 479) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 479): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 479) 
 // Freeing : 2 SGPRs (Value: Int64) (op 479): s14, s15

 // (op 479) reg expression
 // (op 479) reg expression
// Allocated DataFlowTag453: 4 SGPRs (Buffer: None) (op 479): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 479) reg expression
s_mov_b32 s41, s5 // (op 479) reg expression
s_mov_b32 s42, s0 // (op 479) reg expression
s_mov_b32 s43, 131072 // (op 479) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 479): s0, s1

 // (op 469) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(479) END
 // (op 469) NOP(466) BEGIN
 // (op 466) generate({})
 // (op 466) end: generate({})
 // (op 469) NOP(466) END
 // (op 469) Assign VGPR 0:U64(484) BEGIN
 // (op 484) Assign dim(462) = 0:U64
 // (op 484) Assign stride expression
 // (op 469) Assign VGPR 0:U64(484) END
 // (op 469) NOP(487) BEGIN
 // (op 487) generate({})
 // (op 487) end: generate({})
 // (op 469) NOP(487) END
 // (op 469) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(496) BEGIN
 // (op 496) Assign dim(463) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 463: v**UNALLOCATED**
 // (op 496) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag463: VGPR Value: UInt32 x 1: (unallocated)
 // (op 496) reg expression
 // (op 496) reg expression
 // (op 496) reg expression
 // (op 496) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 496): v11
v_lshrrev_b32 v11, 3, v1 // (op 496) 
 // (op 496) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 496): v12
v_lshl_add_u32 v12, s16, 4, v11 // (op 496) 
 // (op 496) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 496): v11

 // (op 496) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 496): v11
v_and_b32 v11, -8, v1 // (op 496) 
 // (op 496) BEGIN: Flatten[1]
 // (op 496) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 496): v13
v_sub_u32 v13, v1, v11 // (op 496) 
 // (op 496) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 496): v11

 // (op 496) BEGIN: TileTile
 // (op 496) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 496) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 496)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 496)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 496): v11
v_lshlrev_b32 v11, 4, v13 // (op 496) 
 // (op 496) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 496): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 496): v14, v15
v_mov_b32 v15, 0 // (op 496) convert
v_mov_b32 v14, v11 // (op 496) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 496): v11

 // (op 496) BEGIN: Split
 // (op 496) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 496)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 496)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 496): v16, v17
v_mul_lo_u32 v17, v12, s3 // (op 496) most significant: low of low * high
 // (op 496) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 496): v11
v_mul_hi_u32 v11, v12, s2 // (op 496) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 496) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 496) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 496): v11

 // (op 496) Add: dest (VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 496)      lhs (VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 496)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 496): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 496) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 496) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 496): s0, s1

 // (op 496) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 496): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 496): v12

 // (op 496) Multiply({Split: v[16:17]:U64}, 6:U32)U64
 // (op 496) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 496)           lhs (Split: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 496)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 496) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 496): v14, v15
v_mul_lo_u32 v15, v17, 6 // (op 496) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 496): v11
v_mul_hi_u32 v11, v16, 6 // (op 496) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 496) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 496) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 496): v11

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 496): v16, v17

 // (op 496) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 496): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 496) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 496): v14, v15

// Allocated DataFlowTag463: 1 VGPR (Value: UInt32) (op 496): v11
v_mov_b32 v11, v12 // (op 496) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 496): v12, v13

 // (op 469) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(496) END
 // (op 469) Assign VGPR 96:U64(497) BEGIN
 // (op 497) Assign dim(464) = 96:U64
 // (op 497) Assign stride expression
 // (op 469) Assign VGPR 96:U64(497) END
 // (op 469) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(498) BEGIN
 // (op 498) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 498) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // (op 498) reg expression
 // (op 498) reg expression
 // (op 498) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 498) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 498)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 498)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 498) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 498): s0, s1
s_mul_i32 s1, s7, 6 // (op 498) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 498) least significant
// Allocated : 1 SGPR (Value: Int32) (op 498): s14
s_mul_hi_u32 s14, s6, 6 // (op 498) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 498) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 498): s14

 // (op 498) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 498) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 498)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 498)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 498): s14, s15
s_mul_i32 s15, s2, s1 // (op 498) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 498): s18
s_mul_i32 s18, s3, s0 // (op 498) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 498) least significant
// Allocated : 1 SGPR (Value: Int32) (op 498): s19
s_mul_hi_u32 s19, s2, s0 // (op 498) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 498) most significant: sum
s_add_u32 s15, s15, s19 // (op 498) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 498): s19

 // Freeing : 1 SGPR (Value: Int32) (op 498): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 498): s0, s1

 // (op 498) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 498): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 498) 
 // Freeing : 2 SGPRs (Value: Int64) (op 498): s14, s15

 // (op 498) reg expression
 // (op 498) reg expression
s_mov_b32 s40, s4 // (op 498) reg expression
s_mov_b32 s41, s5 // (op 498) reg expression
s_mov_b32 s42, s0 // (op 498) reg expression
s_mov_b32 s43, 131072 // (op 498) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 498): s0, s1

 // (op 469) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(498) END
 // (op 469) NOP(489) BEGIN
 // (op 489) generate({})
 // (op 489) end: generate({})
 // (op 469) NOP(489) END
 // (op 469) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(503) BEGIN
 // (op 503) Assign dim(466) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 503) Assign stride expression
 // (op 469) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(503) END
 // (op 469) NOP(490) BEGIN
 // (op 490) generate({})
 // (op 490) end: generate({})
 // (op 469) NOP(490) END
 // (op 469) Assign VGPR 0:U64(506) BEGIN
 // (op 506) Assign dim(468) = 0:U64
 // (op 506) Assign stride expression
 // (op 469) Assign VGPR 0:U64(506) END
 // (op 469) NOP(531) BEGIN
 // (op 531) generate({})
 // (op 531) end: generate({})
 // (op 469) NOP(531) END
 // (op 469) Assign VGPR 128:U64(538) BEGIN
 // (op 538) Assign dim(477) = 128:U64
 // (op 538) Assign stride expression
 // (op 469) Assign VGPR 128:U64(538) END
 // (op 469) NOP(532) BEGIN
 // (op 532) generate({})
 // (op 532) end: generate({})
 // (op 469) NOP(532) END
 // (op 469) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(541) BEGIN
 // (op 541) Assign dim(478) = {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 478: v**UNALLOCATED**
 // (op 541) Generate {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag478: VGPR Value: UInt64 x 1: (unallocated)
 // (op 541) reg expression
 // (op 541) reg expression
 // (op 541) reg expression
 // (op 541) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 541): v12
v_and_b32 v12, -8, v1 // (op 541) 
 // (op 541) BEGIN: Flatten[1]
 // (op 541) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 541): v13
v_sub_u32 v13, v1, v12 // (op 541) 
 // (op 541) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 541): v12

 // (op 541) BEGIN: TileTile
 // (op 541) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 541) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 541)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 541)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 541): v12
v_lshlrev_b32 v12, 4, v13 // (op 541) 
 // (op 541) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 541): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 541): v14, v15
v_mov_b32 v15, 0 // (op 541) convert
v_mov_b32 v14, v12 // (op 541) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 541): v12

 // (op 541) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 541): v12
v_lshrrev_b32 v12, 3, v1 // (op 541) 
 // (op 541) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 541): v13
v_lshl_add_u32 v13, s17, 5, v12 // (op 541) 
 // (op 541) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 541): v12

 // (op 541) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 541) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 541)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 541)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 541): v16, v17
v_mul_lo_u32 v17, v13, s9 // (op 541) most significant: low of low * high
 // (op 541) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 541): v12
v_mul_hi_u32 v12, v13, s8 // (op 541) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 541) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 541) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 541): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 541): v13

 // (op 541) BEGIN: Split
 // (op 541) {Split: Add({TileTile: v[14:15]:U64}, v[16:17]:I64)U64}
 // (op 541) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 541)      lhs (TileTile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 541)      rhs (VGPR Value: Int64 x 1: v[16:17])
// Allocated DataFlowTag478: 2 VGPRs (Value: UInt64) (op 541): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 541): s0, s1
v_add_co_u32 v12, s[0:1], v14, v16 // (op 541) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 541) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 541): s0, s1

 // (op 541) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 541): v16, v17

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 541): v14, v15

 // (op 469) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(541) END
 // (op 469) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(542) BEGIN
 // (op 542) Assign dim(479) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 542) Assign stride expression
 // (op 469) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(542) END
 // (op 469) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(543) BEGIN
 // (op 543) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 472: s**UNALLOCATED**
 // (op 543) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: (unallocated)
 // (op 543) reg expression
 // (op 543) reg expression
 // (op 543) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 543) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 543)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 543)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 543): s0, s1
s_mul_i32 s1, s8, s13 // (op 543) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 543): s14
s_mul_i32 s14, s9, s12 // (op 543) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 543) least significant
// Allocated : 1 SGPR (Value: Int32) (op 543): s15
s_mul_hi_u32 s15, s8, s12 // (op 543) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 543) most significant: sum
s_add_u32 s1, s1, s15 // (op 543) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 543): s15

 // Freeing : 1 SGPR (Value: Int32) (op 543): s14

 // (op 543) reg expression
 // (op 543) reg expression
// Allocated DataFlowTag472: 4 SGPRs (Buffer: None) (op 543): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 543) reg expression
s_mov_b32 s45, s11 // (op 543) reg expression
s_mov_b32 s46, s0 // (op 543) reg expression
s_mov_b32 s47, 131072 // (op 543) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 543): s0, s1

 // (op 469) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(543) END
 // (op 469) NOP(533) BEGIN
 // (op 533) generate({})
 // (op 533) end: generate({})
 // (op 469) NOP(533) END
 // (op 469) Assign VGPR 1:U64(548) BEGIN
 // (op 548) Assign dim(481) = 1:U64
 // (op 548) Assign stride expression
 // (op 469) Assign VGPR 1:U64(548) END
 // (op 469) NOP(551) BEGIN
 // (op 551) generate({})
 // (op 551) end: generate({})
 // (op 469) NOP(551) END
 // (op 469) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(560) BEGIN
 // (op 560) Assign dim(482) = {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 482: v**UNALLOCATED**
 // (op 560) Generate {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag482: VGPR Value: UInt64 x 1: (unallocated)
 // (op 560) reg expression
 // (op 560) reg expression
 // (op 560) reg expression
 // (op 560) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 560): v14
v_and_b32 v14, -8, v1 // (op 560) 
 // (op 560) BEGIN: Flatten[1]
 // (op 560) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 560): v15
v_sub_u32 v15, v1, v14 // (op 560) 
 // (op 560) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 560): v14

 // (op 560) BEGIN: TileTile
 // (op 560) {TileTile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 560) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 560)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 560)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 560): v14
v_lshlrev_b32 v14, 4, v15 // (op 560) 
 // (op 560) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 560): v15

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 560): v16, v17
v_mov_b32 v17, 0 // (op 560) convert
v_mov_b32 v16, v14 // (op 560) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 560): v14

 // (op 560) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 560): v14
v_lshrrev_b32 v14, 3, v1 // (op 560) 
 // (op 560) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 560): v15
v_lshl_add_u32 v15, s17, 5, v14 // (op 560) 
 // (op 560) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 560): v14

 // (op 560) Multiply({Tile: v15:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 560) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 560)           lhs (Tile: VGPR Value: UInt32 x 1: v15) 
 // (op 560)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 560): v18, v19
v_mul_lo_u32 v19, v15, s9 // (op 560) most significant: low of low * high
 // (op 560) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 560): v14
v_mul_hi_u32 v14, v15, s8 // (op 560) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 560) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 560) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 560): v14

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 560): v15

 // (op 560) BEGIN: Split
 // (op 560) {Split: Add({TileTile: v[16:17]:U64}, v[18:19]:I64)U64}
 // (op 560) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 560)      lhs (TileTile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 560)      rhs (VGPR Value: Int64 x 1: v[18:19])
// Allocated DataFlowTag482: 2 VGPRs (Value: UInt64) (op 560): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 560): s0, s1
v_add_co_u32 v14, s[0:1], v16, v18 // (op 560) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 560) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 560): s0, s1

 // (op 560) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 560): v18, v19

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 560): v16, v17

 // (op 469) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(560) END
 // (op 469) Assign VGPR 128:U64(561) BEGIN
 // (op 561) Assign dim(483) = 128:U64
 // (op 561) Assign stride expression
 // (op 469) Assign VGPR 128:U64(561) END
 // (op 469) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(562) BEGIN
 // (op 562) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 562) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // (op 562) reg expression
 // (op 562) reg expression
 // (op 562) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 562) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 562)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 562)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 562): s0, s1
s_mul_i32 s1, s8, s13 // (op 562) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 562): s14
s_mul_i32 s14, s9, s12 // (op 562) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 562) least significant
// Allocated : 1 SGPR (Value: Int32) (op 562): s15
s_mul_hi_u32 s15, s8, s12 // (op 562) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 562) most significant: sum
s_add_u32 s1, s1, s15 // (op 562) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 562): s15

 // Freeing : 1 SGPR (Value: Int32) (op 562): s14

 // (op 562) reg expression
 // (op 562) reg expression
s_mov_b32 s44, s10 // (op 562) reg expression
s_mov_b32 s45, s11 // (op 562) reg expression
s_mov_b32 s46, s0 // (op 562) reg expression
s_mov_b32 s47, 131072 // (op 562) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 562): s0, s1

 // (op 469) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(562) END
 // (op 469) NOP(553) BEGIN
 // (op 553) generate({})
 // (op 553) end: generate({})
 // (op 469) NOP(553) END
 // (op 469) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(567) BEGIN
 // (op 567) Assign dim(485) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 567) Assign stride expression
 // (op 469) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(567) END
 // (op 469) NOP(554) BEGIN
 // (op 554) generate({})
 // (op 554) end: generate({})
 // (op 469) NOP(554) END
 // (op 469) Assign VGPR 1:U64(570) BEGIN
 // (op 570) Assign dim(487) = 1:U64
 // (op 570) Assign stride expression
 // (op 469) Assign VGPR 1:U64(570) END
 // (op 469) NOP(595) BEGIN
 // (op 595) generate({})
 // (op 595) end: generate({})
 // (op 469) NOP(595) END
 // (op 469) Assign VGPR 4:U64(602) BEGIN
 // (op 602) Assign dim(496) = 4:U64
 // (op 602) Assign stride expression
 // (op 469) Assign VGPR 4:U64(602) END
 // (op 469) NOP(596) BEGIN
 // (op 596) generate({})
 // (op 596) end: generate({})
 // (op 469) NOP(596) END
 // (op 469) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}(605) BEGIN
 // (op 605) Assign dim(497) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 497: v**UNALLOCATED**
 // (op 605) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag497: VGPR Value: UInt32 x 1: (unallocated)
 // (op 605) reg expression
 // (op 605) reg expression
 // (op 605) reg expression
 // (op 605) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 605) convert
// Allocated : 1 VGPR (Value: UInt32) (op 605): v16
v_lshrrev_b32 v16, 7, v1 // (op 605) 
 // (op 605) BEGIN: Flatten[1]
 // (op 605) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 605): v17
v_and_b32 v17, 15, v1 // (op 605) 
 // (op 605) END: Flatten[1]
 // (op 605) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 605): v18
v_lshl_add_u32 v18, v16, 4, v17 // (op 605) 
 // (op 605) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 605): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 605): v16

 // (op 605) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 605): v16
v_lshl_add_u32 v16, s16, 4, v18 // (op 605) 
 // (op 605) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 605): v18

 // (op 605) BEGIN: Flatten[1]
 // (op 605) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 605): v17
v_and_b32 v17, 63, v1 // (op 605) 
 // (op 605) END: Flatten[1]
 // (op 605) LogicalShiftR({Flatten[1]: v17:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 605): v18
v_lshrrev_b32 v18, 4, v17 // (op 605) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 605): v17

 // (op 605) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag497: 1 VGPR (Value: UInt32) (op 605): v17
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v17, v16, s20 // (op 605) 
v_add_u32 v17, v17, v18 // (op 605) 
 // (op 605) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 605): v18

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 605): v16

 // (op 469) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}(605) END
 // (op 469) Assign VGPR 4:U64(606) BEGIN
 // (op 606) Assign dim(498) = 4:U64
 // (op 606) Assign stride expression
 // (op 469) Assign VGPR 4:U64(606) END
 // (op 469) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(607) BEGIN
 // (op 607) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 491: s**UNALLOCATED**
 // (op 607) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: (unallocated)
 // (op 607) reg expression
 // (op 607) reg expression
 // (op 607) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 607) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 607)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 607)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 607): s0, s1
s_mul_i32 s1, s20, s7 // (op 607) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 607): s14
s_mul_i32 s14, s21, s6 // (op 607) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 607) least significant
// Allocated : 1 SGPR (Value: Int32) (op 607): s15
s_mul_hi_u32 s15, s20, s6 // (op 607) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 607) most significant: sum
s_add_u32 s1, s1, s15 // (op 607) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 607): s15

 // Freeing : 1 SGPR (Value: Int32) (op 607): s14

 // (op 607) reg expression
 // (op 607) reg expression
// Allocated DataFlowTag491: 4 SGPRs (Buffer: None) (op 607): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 607) reg expression
s_mov_b32 s49, s23 // (op 607) reg expression
s_mov_b32 s50, s0 // (op 607) reg expression
s_mov_b32 s51, 131072 // (op 607) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 607): s0, s1

 // (op 469) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(607) END
 // (op 469) NOP(597) BEGIN
 // (op 597) generate({})
 // (op 597) end: generate({})
 // (op 469) NOP(597) END
 // (op 469) Assign VGPR 1:U64(612) BEGIN
 // (op 612) Assign dim(500) = 1:U64
 // (op 612) Assign stride expression
 // (op 469) Assign VGPR 1:U64(612) END
 // (op 469) NOP(615) BEGIN
 // (op 615) generate({})
 // (op 615) end: generate({})
 // (op 469) NOP(615) END
 // (op 469) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}(624) BEGIN
 // (op 624) Assign dim(501) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 501: v**UNALLOCATED**
 // (op 624) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag501: VGPR Value: UInt32 x 1: (unallocated)
 // (op 624) reg expression
 // (op 624) reg expression
 // (op 624) reg expression
 // (op 624) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 624) convert
// Allocated : 1 VGPR (Value: UInt32) (op 624): v16
v_lshrrev_b32 v16, 7, v1 // (op 624) 
 // (op 624) BEGIN: Flatten[1]
 // (op 624) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 624): v18
v_and_b32 v18, 15, v1 // (op 624) 
 // (op 624) END: Flatten[1]
 // (op 624) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 624): v19
v_lshl_add_u32 v19, v16, 4, v18 // (op 624) 
 // (op 624) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 624): v18

 // Freeing : 1 VGPR (Value: UInt32) (op 624): v16

 // (op 624) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 624): v16
v_lshl_add_u32 v16, s16, 4, v19 // (op 624) 
 // (op 624) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 624): v19

 // (op 624) BEGIN: Flatten[1]
 // (op 624) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 624): v18
v_and_b32 v18, 63, v1 // (op 624) 
 // (op 624) END: Flatten[1]
 // (op 624) LogicalShiftR({Flatten[1]: v18:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 624): v19
v_lshrrev_b32 v19, 4, v18 // (op 624) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 624): v18

 // (op 624) BEGIN: Split
// Allocated DataFlowTag501: 1 VGPR (Value: UInt32) (op 624): v18
v_mul_lo_u32 v18, v16, s20 // (op 624) 
v_add_u32 v18, v18, v19 // (op 624) 
 // (op 624) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 624): v19

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 624): v16

 // (op 469) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32}(624) END
 // (op 469) Assign VGPR 4:U64(625) BEGIN
 // (op 625) Assign dim(502) = 4:U64
 // (op 625) Assign stride expression
 // (op 469) Assign VGPR 4:U64(625) END
 // (op 469) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(626) BEGIN
 // (op 626) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 626) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // (op 626) reg expression
 // (op 626) reg expression
 // (op 626) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 626) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 626)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 626)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 626): s0, s1
s_mul_i32 s1, s20, s7 // (op 626) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 626): s14
s_mul_i32 s14, s21, s6 // (op 626) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 626) least significant
// Allocated : 1 SGPR (Value: Int32) (op 626): s15
s_mul_hi_u32 s15, s20, s6 // (op 626) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 626) most significant: sum
s_add_u32 s1, s1, s15 // (op 626) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 626): s15

 // Freeing : 1 SGPR (Value: Int32) (op 626): s14

 // (op 626) reg expression
 // (op 626) reg expression
s_mov_b32 s48, s22 // (op 626) reg expression
s_mov_b32 s49, s23 // (op 626) reg expression
s_mov_b32 s50, s0 // (op 626) reg expression
s_mov_b32 s51, 131072 // (op 626) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 626): s0, s1

 // (op 469) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(626) END
 // (op 469) NOP(617) BEGIN
 // (op 617) generate({})
 // (op 617) end: generate({})
 // (op 469) NOP(617) END
 // (op 469) Assign VGPR 4:U64(631) BEGIN
 // (op 631) Assign dim(504) = 4:U64
 // (op 631) Assign stride expression
 // (op 469) Assign VGPR 4:U64(631) END
 // (op 469) NOP(618) BEGIN
 // (op 618) generate({})
 // (op 618) end: generate({})
 // (op 469) NOP(618) END
 // (op 469) Assign VGPR 1:U64(634) BEGIN
 // (op 634) Assign dim(506) = 1:U64
 // (op 634) Assign stride expression
 // (op 469) Assign VGPR 1:U64(634) END
 // (op 469) NOP(659) BEGIN
 // (op 659) generate({})
 // (op 659) end: generate({})
 // (op 469) NOP(659) END
 // (op 469) Assign VGPR 4:U64(666) BEGIN
 // (op 666) Assign dim(515) = 4:U64
 // (op 666) Assign stride expression
 // (op 469) Assign VGPR 4:U64(666) END
 // (op 469) NOP(660) BEGIN
 // (op 660) generate({})
 // (op 660) end: generate({})
 // (op 469) NOP(660) END
 // (op 469) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(669) BEGIN
 // (op 669) Assign dim(516) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 516: v**UNALLOCATED**
 // (op 669) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag516: VGPR Value: UInt32 x 1: (unallocated)
 // (op 669) reg expression
 // (op 669) reg expression
 // (op 669) reg expression
 // (op 669) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
 // (op 669) convert
// Allocated : 1 VGPR (Value: UInt32) (op 669): v16
v_and_b32 v16, -4, v1 // (op 669) 
 // (op 669) BEGIN: Flatten[1]TileTile
 // (op 669) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 669): v19
v_sub_u32 v19, v1, v16 // (op 669) 
 // (op 669) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 669): v16

 // (op 669) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 669): v16
v_lshrrev_b32 v16, 2, v1 // (op 669) 
 // (op 669) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 669): v20
v_lshl_add_u32 v20, s17, 5, v16 // (op 669) 
 // (op 669) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 669): v16

 // (op 669) Multiply({Tile: v20:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 669): v16
v_mul_lo_u32 v16, v20, s24 // (op 669) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 669): v20

 // (op 669) BEGIN: Split
 // (op 669) {Split: Add({Flatten[1]TileTile: v19:U32}, v16:U32)U32}
// Allocated DataFlowTag516: 1 VGPR (Value: UInt32) (op 669): v20
v_add_u32 v20, v19, v16 // (op 669) 
 // (op 669) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 669): v16

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 669): v19

 // (op 469) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(669) END
 // (op 469) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(670) BEGIN
 // (op 670) Assign dim(517) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 670) Assign stride expression
 // (op 469) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(670) END
 // (op 469) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(671) BEGIN
 // (op 671) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 510: s**UNALLOCATED**
 // (op 671) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: (unallocated)
 // (op 671) reg expression
 // (op 671) reg expression
 // (op 671) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 671) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 671)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 671)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 671): s0, s1
s_mul_i32 s1, s24, s13 // (op 671) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 671): s14
s_mul_i32 s14, s25, s12 // (op 671) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 671) least significant
// Allocated : 1 SGPR (Value: Int32) (op 671): s15
s_mul_hi_u32 s15, s24, s12 // (op 671) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 671) most significant: sum
s_add_u32 s1, s1, s15 // (op 671) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 671): s15

 // Freeing : 1 SGPR (Value: Int32) (op 671): s14

 // (op 671) reg expression
 // (op 671) reg expression
// Allocated DataFlowTag510: 4 SGPRs (Buffer: None) (op 671): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 671) reg expression
s_mov_b32 s53, s27 // (op 671) reg expression
s_mov_b32 s54, s0 // (op 671) reg expression
s_mov_b32 s55, 131072 // (op 671) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 671): s0, s1

 // (op 469) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(671) END
 // (op 469) NOP(661) BEGIN
 // (op 661) generate({})
 // (op 661) end: generate({})
 // (op 469) NOP(661) END
 // (op 469) Assign VGPR 1:U64(676) BEGIN
 // (op 676) Assign dim(519) = 1:U64
 // (op 676) Assign stride expression
 // (op 469) Assign VGPR 1:U64(676) END
 // (op 469) NOP(679) BEGIN
 // (op 679) generate({})
 // (op 679) end: generate({})
 // (op 469) NOP(679) END
 // (op 469) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(688) BEGIN
 // (op 688) Assign dim(520) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 520: v**UNALLOCATED**
 // (op 688) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag520: VGPR Value: UInt32 x 1: (unallocated)
 // (op 688) reg expression
 // (op 688) reg expression
 // (op 688) reg expression
 // (op 688) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
 // (op 688) convert
// Allocated : 1 VGPR (Value: UInt32) (op 688): v16
v_and_b32 v16, -4, v1 // (op 688) 
 // (op 688) BEGIN: Flatten[1]TileTile
 // (op 688) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 688): v19
v_sub_u32 v19, v1, v16 // (op 688) 
 // (op 688) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 688): v16

 // (op 688) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 688): v16
v_lshrrev_b32 v16, 2, v1 // (op 688) 
 // (op 688) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 688): v21
v_lshl_add_u32 v21, s17, 5, v16 // (op 688) 
 // (op 688) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 688): v16

 // (op 688) Multiply({Tile: v21:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 688): v16
v_mul_lo_u32 v16, v21, s24 // (op 688) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 688): v21

 // (op 688) BEGIN: Split
 // (op 688) {Split: Add({Flatten[1]TileTile: v19:U32}, v16:U32)U32}
// Allocated DataFlowTag520: 1 VGPR (Value: UInt32) (op 688): v21
v_add_u32 v21, v19, v16 // (op 688) 
 // (op 688) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 688): v16

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 688): v19

 // (op 469) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(688) END
 // (op 469) Assign VGPR 4:U64(689) BEGIN
 // (op 689) Assign dim(521) = 4:U64
 // (op 689) Assign stride expression
 // (op 469) Assign VGPR 4:U64(689) END
 // (op 469) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(690) BEGIN
 // (op 690) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 690) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // (op 690) reg expression
 // (op 690) reg expression
 // (op 690) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 690) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 690)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 690)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 690): s0, s1
s_mul_i32 s1, s24, s13 // (op 690) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 690): s14
s_mul_i32 s14, s25, s12 // (op 690) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 690) least significant
// Allocated : 1 SGPR (Value: Int32) (op 690): s15
s_mul_hi_u32 s15, s24, s12 // (op 690) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 690) most significant: sum
s_add_u32 s1, s1, s15 // (op 690) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 690): s15

 // Freeing : 1 SGPR (Value: Int32) (op 690): s14

 // (op 690) reg expression
 // (op 690) reg expression
s_mov_b32 s52, s26 // (op 690) reg expression
s_mov_b32 s53, s27 // (op 690) reg expression
s_mov_b32 s54, s0 // (op 690) reg expression
s_mov_b32 s55, 131072 // (op 690) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 690): s0, s1

 // (op 469) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(690) END
 // (op 469) NOP(681) BEGIN
 // (op 681) generate({})
 // (op 681) end: generate({})
 // (op 469) NOP(681) END
 // (op 469) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(695) BEGIN
 // (op 695) Assign dim(523) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 695) Assign stride expression
 // (op 469) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(695) END
 // (op 469) NOP(682) BEGIN
 // (op 682) generate({})
 // (op 682) end: generate({})
 // (op 469) NOP(682) END
 // (op 469) Assign VGPR 1:U64(698) BEGIN
 // (op 698) Assign dim(525) = 1:U64
 // (op 698) Assign stride expression
 // (op 469) Assign VGPR 1:U64(698) END
 // (op 469) NOP(330) BEGIN
 // (op 330) generate({})
 // (op 330) end: generate({})
 // (op 469) NOP(330) END
 // (op 469) SetCoordinate(335) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag497: VGPR Value: UInt32 x 1: v17
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

// Allocated : 1 VGPR (Value: E8M0) (op 337): v16
buffer_load_ubyte v16, v17, s[48:51], 0 offen offset: 0 // (op 337) Load value
// VMEM: Expected complete at 789 (current 309)
 // (op 336) LoadTiled Value: E8M0(337) END
 // (op 336) end: generate({337})
 // (op 335) SetCoordinate(336) END
 // (op 335) end: generate({336})
 // (op 469) SetCoordinate(335) END
 // (op 469) SetCoordinate(340) BEGIN
 // (op 340) generate({})
 // (op 340) end: generate({})
 // (op 340) generate({803})
 // (op 340) Scope(803) BEGIN
 // (op 803) Lock Scope 803
 // Lock Branch
 // (op 803) generate({800})
 // (op 803) NOP(800) BEGIN
 // (op 800) generate({})
 // (op 800) end: generate({})
 // (op 803) NOP(800) END
 // (op 803) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(807) BEGIN
 // (op 807) Assign dim(556) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 556: s**UNALLOCATED**
 // (op 807) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag556: SGPR Value: UInt32 x 1: (unallocated)
 // (op 807) reg expression
 // (op 807) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 807): v19
v_and_b32 v19, -8, v1 // (op 807) 
 // (op 807) BEGIN: Tile[1]
 // (op 807) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v19:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 807): v22
v_sub_u32 v22, v1, v19 // (op 807) 
 // (op 807) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 807): v19

 // (op 807) BEGIN: Flatten
 // (op 807) {Flatten: ShiftL({Tile[1]: v22:U32}, 4:U32)U32}
 // (op 807) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 807)         value (Tile[1]: VGPR Value: UInt32 x 1: v22) 
 // (op 807)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 807): v19
v_lshlrev_b32 v19, 4, v22 // (op 807) 
 // (op 807) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 807): v22

 // (op 807) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 807): v22
v_lshrrev_b32 v22, 3, v1 // (op 807) 
 // (op 807) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 807): v23
v_lshl_add_u32 v23, v22, 7, v19 // (op 807) 
 // (op 807) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 807): v19

 // Freeing : 1 VGPR (Value: UInt32) (op 807): v22

// Allocated DataFlowTag556: 1 SGPR (Value: UInt32) (op 807): s0
s_nop 0
v_readfirstlane_b32 s0, v23 // (op 807) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 807): v23

 // (op 803) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(807) END
 // (op 803) Assign VGPR 2048:U32(808) BEGIN
 // (op 808) Assign dim(557) = 2048:U32
 // (op 808) Assign stride expression
 // (op 803) Assign VGPR 2048:U32(808) END
 // (op 803) NOP(801) BEGIN
 // (op 801) generate({})
 // (op 801) end: generate({})
 // (op 803) NOP(801) END
 // (op 803) Assign VGPR 1:U32(812) BEGIN
 // (op 812) Assign dim(559) = 1:U32
 // (op 812) Assign stride expression
 // (op 803) Assign VGPR 1:U32(812) END
 // (op 803) LoadTileDirect2LDS Value: FP8x4(433) BEGIN
 // (op 433) GEN: LoadTileDirect2LDS
 // (op 433) GEN: loadMacroTileDirect2LDS OP 433 LDS 24 MacroTile 9
 // (op 433) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 433) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride479: VGPR Value: UInt64 x 1: (unallocated)
 // (op 433) reg expression
 // (op 433) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 433) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 433)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 433)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 433): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 433) 
// Allocated Stride479: 2 VGPRs (Value: UInt64) (op 433): v22, v23
v_mov_b64 v[22:23], s[14:15] // (op 433) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 433): s14, s15

 // (op 433) Generate 1:U64 into nullptr
 // (op 433) Generate 2048:U32 into nullptr
 // (op 433) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride479: VGPR Value: UInt64 x 1: v[22:23]
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 433): v24, v25
v_mov_b64 v[24:25], v[12:13] // (op 433) 
 // (op 433) Generate Add(0:U32, {DataFlowTag556Flatten: s0:U32})U32 into nullptr
 // (op 433)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride479: v[22:23]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 433) Generate {DataFlowTag556Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 433) reg expression
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 433) Load value direct to lds
// VMEM: Expected complete at 801 (current 321)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 433) Generate Add(v[24:25]:U64, v22:R)U64 into VGPR Value: UInt64 x 1: v[24:25]
 // (op 433) reg expression
 // (op 433) reg expression
 // (op 433) Add(v[24:25]:U64, v22:R)U64
 // (op 433) Add: dest (VGPR Value: UInt64 x 1: v[24:25]) = 
 // (op 433)      lhs (VGPR Value: UInt64 x 1: v[24:25]) 
 // (op 433)      rhs (VGPR Value: Raw32 x 1: v22)
// Allocated : 2 SGPRs (Value: Bool64) (op 433): s14, s15
v_add_co_u32 v24, s[14:15], v24, v22 // (op 433) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 433) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 433): s14, s15

 // (op 433) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 433) reg expression
 // (op 433) Add(m0:U32, 2048:U32)U32
 // (op 433) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 433) 
 // (op 433) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 433) Load value direct to lds
// VMEM: Expected complete at 805 (current 325)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // Freeing Stride479: 2 VGPRs (Value: UInt64) (op 433): v22, v23

 // Freeing : 2 VGPRs (Value: UInt64) (op 433): v24, v25

 // (op 803) LoadTileDirect2LDS Value: FP8x4(433) END
 // (op 803) Deallocate{}(1174) BEGIN
 // (op 1174) Deallocate 559
 // Deleting tag 559
 // (op 1174) Deallocate 557
 // Deleting tag 557
 // (op 1174) Deallocate 556
 // Freeing DataFlowTag556: 1 SGPR (Value: UInt32) (op 807): s0

 // Deleting tag 556
 // (op 1174) Deallocate 558
 // Deleting tag 558
 // (op 803) Deallocate{}(1174) END
 // (op 803) end: generate({800})
 // (op 803) Unlock Scope 803
 // Unlock None
 // (op 340) Scope(803) END
 // (op 340) end: generate({803})
 // (op 469) SetCoordinate(340) END
 // (op 469) SetCoordinate(343) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag516: VGPR Value: UInt32 x 1: v20
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

// Allocated : 1 VGPR (Value: E8M0) (op 344): v19
buffer_load_ubyte v19, v20, s[52:55], 0 offen offset: 0 // (op 344) Load value
// VMEM: Expected stall of 464. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1270 (current 790)
 // (op 343) LoadTiled Value: E8M0(344) END
 // (op 343) end: generate({344})
 // (op 469) SetCoordinate(343) END
 // (op 469) SetCoordinate(346) BEGIN
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

// Allocated : 3 VGPRs (Value: FP6x16) (op 347): v22, v23, v24
buffer_load_dwordx3 v[22:24], v10, s[40:43], 0 offen // (op 347) Load value
// VMEM: Expected stall of 11. CBNW: 1, Inc: 3
// VMEM: Expected complete at 1282 (current 802)
 // (op 346) LoadTiled Value: FP6x16(347) END
 // (op 346) end: generate({347})
 // (op 469) SetCoordinate(346) END
 // (op 469) SetCoordinate(349) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag497: VGPR Value: UInt32 x 1: v17
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

// Allocated : 1 VGPR (Value: E8M0) (op 351): v25
buffer_load_ubyte v25, v17, s[48:51], 0 offen offset:4 // (op 351) Load value
// VMEM: Expected stall of 3. CBNW: 2, Inc: 3
// VMEM: Expected complete at 1286 (current 806)
 // (op 350) LoadTiled Value: E8M0(351) END
 // (op 350) end: generate({351})
 // (op 349) SetCoordinate(350) END
 // (op 349) end: generate({350})
 // (op 469) SetCoordinate(349) END
 // (op 469) Deallocate{}(1156) BEGIN
 // (op 1156) Deallocate 498
 // Deleting tag 498
 // (op 1156) Deallocate 500
 // Deleting tag 500
 // (op 1156) Deallocate 496
 // Deleting tag 496
 // (op 1156) Deallocate 497
 // Freeing DataFlowTag497: 1 VGPR (Value: UInt32) (op 605): v17

 // Deleting tag 497
 // (op 1156) Deallocate 499
 // Deleting tag 499
 // (op 469) Deallocate{}(1156) END
 // (op 469) SetCoordinate(354) BEGIN
 // (op 354) generate({})
 // (op 354) end: generate({})
 // (op 354) generate({953})
 // (op 354) Scope(953) BEGIN
 // (op 953) Lock Scope 953
 // Lock Branch
 // (op 953) generate({950})
 // (op 953) NOP(950) BEGIN
 // (op 950) generate({})
 // (op 950) end: generate({})
 // (op 953) NOP(950) END
 // (op 953) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(957) BEGIN
 // (op 957) Assign dim(602) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 602: s**UNALLOCATED**
 // (op 957) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag602: SGPR Value: UInt32 x 1: (unallocated)
 // (op 957) reg expression
 // (op 957) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 957): v17
v_and_b32 v17, -8, v1 // (op 957) 
 // (op 957) BEGIN: Tile[1]
 // (op 957) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v17:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 957): v26
v_sub_u32 v26, v1, v17 // (op 957) 
 // (op 957) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 957): v17

 // (op 957) BEGIN: Flatten
 // (op 957) {Flatten: ShiftL({Tile[1]: v26:U32}, 4:U32)U32}
 // (op 957) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 957)         value (Tile[1]: VGPR Value: UInt32 x 1: v26) 
 // (op 957)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 957): v17
v_lshlrev_b32 v17, 4, v26 // (op 957) 
 // (op 957) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 957): v26

 // (op 957) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 957): v26
v_lshrrev_b32 v26, 3, v1 // (op 957) 
 // (op 957) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 957): v27
v_lshl_add_u32 v27, v26, 7, v17 // (op 957) 
 // (op 957) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 957): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 957): v26

// Allocated DataFlowTag602: 1 SGPR (Value: UInt32) (op 957): s0
s_nop 0
v_readfirstlane_b32 s0, v27 // (op 957) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 957): v27

 // (op 953) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(957) END
 // (op 953) Assign VGPR 2048:U32(958) BEGIN
 // (op 958) Assign dim(603) = 2048:U32
 // (op 958) Assign stride expression
 // (op 953) Assign VGPR 2048:U32(958) END
 // (op 953) NOP(951) BEGIN
 // (op 951) generate({})
 // (op 951) end: generate({})
 // (op 953) NOP(951) END
 // (op 953) Assign VGPR 1:U32(962) BEGIN
 // (op 962) Assign dim(605) = 1:U32
 // (op 962) Assign stride expression
 // (op 953) Assign VGPR 1:U32(962) END
 // (op 953) LoadTileDirect2LDS Value: FP8x4(438) BEGIN
 // (op 438) GEN: LoadTileDirect2LDS
 // (op 438) GEN: loadMacroTileDirect2LDS OP 438 LDS 436 MacroTile 434
 // (op 438) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 438) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride479: VGPR Value: UInt64 x 1: (unallocated)
 // (op 438) reg expression
 // (op 438) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 438) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 438)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 438)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 438): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 438) 
// Allocated Stride479: 2 VGPRs (Value: UInt64) (op 438): v26, v27
v_mov_b64 v[26:27], s[14:15] // (op 438) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 438): s14, s15

 // (op 438) Generate 1:U64 into nullptr
 // (op 438) Generate 2048:U32 into nullptr
 // (op 438) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 4096
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride479: VGPR Value: UInt64 x 1: v[26:27]
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 438): v28, v29
v_mov_b64 v[28:29], v[12:13] // (op 438) 
 // (op 438) Generate Add(4096:U32, {DataFlowTag602Flatten: s0:U32})U32 into nullptr
 // (op 438) reg expression
 // (op 438) Add(4096:U32, {DataFlowTag602Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 438): s1
s_add_u32 s1, 4096, s0 // (op 438) 
 // (op 438)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride479: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 438) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 438) reg expression
 // (op 438) Generate 128:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 128 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 438): s14
s_mov_b32 s14, 128 // (op 438) call()
// 128 is not a supported value!
buffer_load_dwordx4 v28, s[44:47], s14 offen offset: 0 lds // (op 438) Load value direct to lds
// VMEM: Expected stall of 451. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1751 (current 1271)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 438) Generate Add(v[28:29]:U64, v26:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 438) reg expression
 // (op 438) reg expression
 // (op 438) Add(v[28:29]:U64, v26:R)U64
 // (op 438) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 438)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 438)      rhs (VGPR Value: Raw32 x 1: v26)
// Allocated : 2 SGPRs (Value: Bool64) (op 438): s18, s19
v_add_co_u32 v28, s[18:19], v28, v26 // (op 438) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 438) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 438): s18, s19

 // (op 438) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 438) reg expression
 // (op 438) Add(m0:U32, 2048:U32)U32
 // (op 438) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 438) 
 // (op 438) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[44:47], s14 offen offset: 0 lds // (op 438) Load value direct to lds
// VMEM: Expected stall of 8. CBNW: 4, Inc: 3
// VMEM: Expected complete at 1763 (current 1283)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 438): s14

 // Freeing Stride479: 2 VGPRs (Value: UInt64) (op 438): v26, v27

 // Freeing : 2 VGPRs (Value: UInt64) (op 438): v28, v29

 // Freeing : 1 SGPR (Value: UInt32) (op 438): s1

 // (op 953) LoadTileDirect2LDS Value: FP8x4(438) END
 // (op 953) Deallocate{}(1176) BEGIN
 // (op 1176) Deallocate 604
 // Deleting tag 604
 // (op 1176) Deallocate 605
 // Deleting tag 605
 // (op 1176) Deallocate 602
 // Freeing DataFlowTag602: 1 SGPR (Value: UInt32) (op 957): s0

 // Deleting tag 602
 // (op 1176) Deallocate 603
 // Deleting tag 603
 // (op 953) Deallocate{}(1176) END
 // (op 953) end: generate({950})
 // (op 953) Unlock Scope 953
 // Unlock None
 // (op 354) Scope(953) END
 // (op 354) end: generate({953})
 // (op 469) SetCoordinate(354) END
 // (op 469) Deallocate{}(1158) BEGIN
 // (op 1158) Deallocate 477
 // Deleting tag 477
 // (op 1158) Deallocate 478
 // Freeing DataFlowTag478: 2 VGPRs (Value: UInt64) (op 541): v12, v13

 // Deleting tag 478
 // (op 1158) Deallocate 481
 // Deleting tag 481
 // (op 1158) Deallocate 479
 // Deleting tag 479
 // (op 1158) Deallocate 480
 // Deleting tag 480
 // (op 469) Deallocate{}(1158) END
 // (op 469) SetCoordinate(357) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag516: VGPR Value: UInt32 x 1: v20
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

// Allocated : 1 VGPR (Value: E8M0) (op 358): v17
buffer_load_ubyte v17, v20, s[52:55], 0 offen offset:4 // (op 358) Load value
// VMEM: Expected stall of 3. CBNW: 5, Inc: 3
// VMEM: Expected complete at 1767 (current 1287)
 // (op 357) LoadTiled Value: E8M0(358) END
 // (op 357) end: generate({358})
 // (op 469) SetCoordinate(357) END
 // (op 469) Deallocate{}(1160) BEGIN
 // (op 1160) Deallocate 519
 // Deleting tag 519
 // (op 1160) Deallocate 516
 // Freeing DataFlowTag516: 1 VGPR (Value: UInt32) (op 669): v20

 // Deleting tag 516
 // (op 1160) Deallocate 517
 // Deleting tag 517
 // (op 1160) Deallocate 518
 // Deleting tag 518
 // (op 1160) Deallocate 515
 // Deleting tag 515
 // (op 469) Deallocate{}(1160) END
 // (op 469) SetCoordinate(360) BEGIN
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

// Allocated : 3 VGPRs (Value: FP6x16) (op 361): v26, v27, v28
buffer_load_dwordx3 v[26:28], v10, s[40:43], 0 offen offset:96 // (op 361) Load value
// VMEM: Expected stall of 464. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2232 (current 1752)
 // (op 360) LoadTiled Value: FP6x16(361) END
 // (op 360) end: generate({361})
 // (op 469) SetCoordinate(360) END
 // (op 469) Deallocate{}(1162) BEGIN
 // (op 1162) Deallocate 462
 // Deleting tag 462
 // (op 1162) Deallocate 458
 // Deleting tag 458
 // (op 1162) Deallocate 460
 // Deleting tag 460
 // (op 1162) Deallocate 459
 // Freeing DataFlowTag459: 1 VGPR (Value: UInt32) (op 477): v10

 // Deleting tag 459
 // (op 1162) Deallocate 461
 // Deleting tag 461
 // (op 469) Deallocate{}(1162) END
 // (op 469) NOP(435) BEGIN
 // (op 435) generate({})
 // (op 435) end: generate({})
 // (op 469) NOP(435) END
 // (op 469) Scope(855) BEGIN
 // (op 855) Lock Scope 855
 // Lock Branch
 // (op 855) generate({852})
 // (op 855) NOP(852) BEGIN
 // (op 852) generate({})
 // (op 852) end: generate({})
 // (op 855) NOP(852) END
 // (op 855) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(860) BEGIN
 // (op 860) Assign dim(572) = {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}
 // tag 572: v**UNALLOCATED**
 // (op 860) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32} into DataFlowTag572: VGPR Value: UInt32 x 1: (unallocated)
 // (op 860) reg expression
 // (op 860) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 860): v10
v_and_b32 v10, -4, v1 // (op 860) 
 // (op 860) BEGIN: Tile[1]Flatten
 // (op 860) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 860): v20
v_sub_u32 v20, v1, v10 // (op 860) 
 // (op 860) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 860): v10

 // (op 860) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 860): v10
v_and_b32 v10, 4294967292, v1 // (op 860) 
 // (op 860) BEGIN: Flatten
 // (op 860) {Flatten: Add(v10:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag572: 1 VGPR (Value: UInt32) (op 860): v12
v_add_u32 v12, v10, v20 // (op 860) 
 // (op 860) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 860): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 860): v10

 // (op 855) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(860) END
 // (op 855) Assign VGPR 4:U32(861) BEGIN
 // (op 861) Assign dim(573) = 4:U32
 // (op 861) Assign stride expression
 // (op 855) Assign VGPR 4:U32(861) END
 // (op 855) NOP(853) BEGIN
 // (op 853) generate({})
 // (op 853) end: generate({})
 // (op 855) NOP(853) END
 // (op 855) Assign VGPR 1:U32(865) BEGIN
 // (op 865) Assign dim(575) = 1:U32
 // (op 865) Assign stride expression
 // (op 855) Assign VGPR 1:U32(865) END
 // (op 855) StoreLDSTile Value: E8M0(364) BEGIN
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

 // (op 364) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag572: VGPR Value: UInt32 x 1: v12
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
ds_write_b8 v12, v19 offset:8192 // (op 364) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 1853 (current 1758)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 855) StoreLDSTile Value: E8M0(364) END
 // (op 855) Deallocate{}(1164) BEGIN
 // (op 1164) Deallocate 573
 // Deleting tag 573
 // (op 1164) Deallocate 572
 // Freeing DataFlowTag572: 1 VGPR (Value: UInt32) (op 860): v12

 // Deleting tag 572
 // (op 1164) Deallocate 575
 // Deleting tag 575
 // (op 1164) Deallocate 574
 // Deleting tag 574
 // (op 855) Deallocate{}(1164) END
 // (op 855) end: generate({852})
 // (op 855) Unlock Scope 855
 // Unlock None
 // (op 469) Scope(855) END
 // (op 469) Scope(750) BEGIN
 // (op 750) Lock Scope 750
 // Lock Branch
 // (op 750) generate({747})
 // (op 750) NOP(747) BEGIN
 // (op 747) generate({})
 // (op 747) end: generate({})
 // (op 750) NOP(747) END
 // (op 750) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(755) BEGIN
 // (op 755) Assign dim(540) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 540: v**UNALLOCATED**
 // (op 755) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag540: VGPR Value: UInt32 x 1: (unallocated)
 // (op 755) reg expression
 // (op 755) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 755): v10
v_and_b32 v10, -8, v1 // (op 755) 
 // (op 755) BEGIN: Tile[1]
 // (op 755) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 755): v20
v_sub_u32 v20, v1, v10 // (op 755) 
 // (op 755) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 755): v10

 // (op 755) BEGIN: Flatten
 // (op 755) {Flatten: ShiftL({Tile[1]: v20:U32}, 4:U32)U32}
 // (op 755) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 755)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 755)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 755): v10
v_lshlrev_b32 v10, 4, v20 // (op 755) 
 // (op 755) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 755): v20

 // (op 755) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 755): v20
v_lshrrev_b32 v20, 3, v1 // (op 755) 
 // (op 755) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 755): v12
v_lshl_add_u32 v12, v20, 7, v10 // (op 755) 
 // (op 755) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 755): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 755): v20

 // (op 755) Multiply({Flatten: v12:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 755): v10
v_mul_lo_u32 v10, v12, 6 // (op 755) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 755): v12

 // (op 755) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag540: 1 VGPR (Value: UInt32) (op 755): v20
v_lshrrev_b32 v20, 3, v10 // (op 755) 
 // Freeing : 1 VGPR (Value: UInt32) (op 755): v10

 // (op 750) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(755) END
 // (op 750) Assign VGPR 96:U32(756) BEGIN
 // (op 756) Assign dim(541) = 96:U32
 // (op 756) Assign stride expression
 // (op 750) Assign VGPR 96:U32(756) END
 // (op 750) NOP(748) BEGIN
 // (op 748) generate({})
 // (op 748) end: generate({})
 // (op 750) NOP(748) END
 // (op 750) Assign VGPR 0:U32(760) BEGIN
 // (op 760) Assign dim(543) = 0:U32
 // (op 760) Assign stride expression
 // (op 750) Assign VGPR 0:U32(760) END
 // (op 750) StoreLDSTile Value: FP6x16(365) BEGIN
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
 // 	info.data = VGPR Value: FP6x16 x 1: v[22:24]
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

 // (op 365) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[22:24]
 // 	info.rowOffsetReg = DataFlowTag540: VGPR Value: UInt32 x 1: v20
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
ds_write_b96 v20, v[22:24] offset:8320 // (op 365) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 1862 (current 1767)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 750) StoreLDSTile Value: FP6x16(365) END
 // (op 750) Deallocate{}(1166) BEGIN
 // (op 1166) Deallocate 543
 // Deleting tag 543
 // (op 1166) Deallocate 540
 // Freeing DataFlowTag540: 1 VGPR (Value: UInt32) (op 755): v20

 // Deleting tag 540
 // (op 1166) Deallocate 541
 // Deleting tag 541
 // (op 1166) Deallocate 542
 // Deleting tag 542
 // (op 750) Deallocate{}(1166) END
 // (op 750) end: generate({747})
 // (op 750) Unlock Scope 750
 // Unlock None
 // (op 469) Scope(750) END
 // (op 469) NOP(333) BEGIN
 // (op 333) generate({})
 // (op 333) end: generate({})
 // (op 469) NOP(333) END
 // (op 469) NOP(722) BEGIN
 // (op 722) generate({})
 // (op 722) end: generate({})
 // (op 469) NOP(722) END
 // (op 469) NOP(719) BEGIN
 // (op 719) generate({})
 // (op 719) end: generate({})
 // (op 469) NOP(719) END
 // (op 469) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(726) BEGIN
 // (op 726) Assign dim(532) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 532: v**UNALLOCATED**
 // (op 726) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag532: VGPR Value: UInt32 x 1: (unallocated)
 // (op 726) reg expression
 // (op 726) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 726): v10
v_and_b32 v10, -8, v1 // (op 726) 
 // (op 726) BEGIN: Tile[1]
 // (op 726) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 726): v20
v_sub_u32 v20, v1, v10 // (op 726) 
 // (op 726) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 726): v10

 // (op 726) BEGIN: Flatten
 // (op 726) {Flatten: ShiftL({Tile[1]: v20:U32}, 4:U32)U32}
 // (op 726) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 726)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 726)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 726): v10
v_lshlrev_b32 v10, 4, v20 // (op 726) 
 // (op 726) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 726): v20

 // (op 726) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 726): v20
v_lshrrev_b32 v20, 3, v1 // (op 726) 
 // (op 726) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 726): v12
v_lshl_add_u32 v12, v20, 7, v10 // (op 726) 
 // (op 726) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 726): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 726): v20

 // (op 726) Multiply({Flatten: v12:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 726): v10
v_mul_lo_u32 v10, v12, 6 // (op 726) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 726): v12

 // (op 726) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag532: 1 VGPR (Value: UInt32) (op 726): v20
v_lshrrev_b32 v20, 3, v10 // (op 726) 
 // Freeing : 1 VGPR (Value: UInt32) (op 726): v10

 // (op 469) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(726) END
 // (op 469) Assign VGPR 96:U32(727) BEGIN
 // (op 727) Assign dim(533) = 96:U32
 // (op 727) Assign stride expression
 // (op 469) Assign VGPR 96:U32(727) END
 // (op 469) NOP(720) BEGIN
 // (op 720) generate({})
 // (op 720) end: generate({})
 // (op 469) NOP(720) END
 // (op 469) Assign VGPR 0:U32(731) BEGIN
 // (op 731) Assign dim(535) = 0:U32
 // (op 731) Assign stride expression
 // (op 469) Assign VGPR 0:U32(731) END
 // (op 469) NOP(774) BEGIN
 // (op 774) generate({})
 // (op 774) end: generate({})
 // (op 469) NOP(774) END
 // (op 469) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(779) BEGIN
 // (op 779) Assign dim(548) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 548: s**UNALLOCATED**
 // (op 779) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag548: SGPR Value: UInt32 x 1: (unallocated)
 // (op 779) reg expression
 // (op 779) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 779): v10
v_and_b32 v10, -8, v1 // (op 779) 
 // (op 779) BEGIN: Tile[1]
 // (op 779) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 779): v12
v_sub_u32 v12, v1, v10 // (op 779) 
 // (op 779) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 779): v10

 // (op 779) BEGIN: Flatten
 // (op 779) {Flatten: ShiftL({Tile[1]: v12:U32}, 4:U32)U32}
 // (op 779) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 779)         value (Tile[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 779)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 779): v10
v_lshlrev_b32 v10, 4, v12 // (op 779) 
 // (op 779) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 779): v12

 // (op 779) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 779): v12
v_lshrrev_b32 v12, 3, v1 // (op 779) 
 // (op 779) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 779): v13
v_lshl_add_u32 v13, v12, 7, v10 // (op 779) 
 // (op 779) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 779): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 779): v12

// Allocated DataFlowTag548: 1 SGPR (Value: UInt32) (op 779): s0
s_nop 0
v_readfirstlane_b32 s0, v13 // (op 779) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 779): v13

 // (op 469) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(779) END
 // (op 469) Assign VGPR 2048:U32(780) BEGIN
 // (op 780) Assign dim(549) = 2048:U32
 // (op 780) Assign stride expression
 // (op 469) Assign VGPR 2048:U32(780) END
 // (op 469) NOP(775) BEGIN
 // (op 775) generate({})
 // (op 775) end: generate({})
 // (op 469) NOP(775) END
 // (op 469) Assign VGPR 1:U32(784) BEGIN
 // (op 784) Assign dim(551) = 1:U32
 // (op 784) Assign stride expression
 // (op 469) Assign VGPR 1:U32(784) END
 // (op 469) NOP(826) BEGIN
 // (op 826) generate({})
 // (op 826) end: generate({})
 // (op 469) NOP(826) END
 // (op 469) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(831) BEGIN
 // (op 831) Assign dim(564) = {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}
 // tag 564: v**UNALLOCATED**
 // (op 831) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32} into DataFlowTag564: VGPR Value: UInt32 x 1: (unallocated)
 // (op 831) reg expression
 // (op 831) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 831): v10
v_and_b32 v10, -4, v1 // (op 831) 
 // (op 831) BEGIN: Tile[1]Flatten
 // (op 831) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 831): v12
v_sub_u32 v12, v1, v10 // (op 831) 
 // (op 831) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 831): v10

 // (op 831) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 831): v10
v_and_b32 v10, 4294967292, v1 // (op 831) 
 // (op 831) BEGIN: Flatten
 // (op 831) {Flatten: Add(v10:U32, {Tile[1]Flatten: v12:U32})U32}
// Allocated DataFlowTag564: 1 VGPR (Value: UInt32) (op 831): v13
v_add_u32 v13, v10, v12 // (op 831) 
 // (op 831) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 831): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 831): v10

 // (op 469) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(831) END
 // (op 469) Assign VGPR 4:U32(832) BEGIN
 // (op 832) Assign dim(565) = 4:U32
 // (op 832) Assign stride expression
 // (op 469) Assign VGPR 4:U32(832) END
 // (op 469) NOP(827) BEGIN
 // (op 827) generate({})
 // (op 827) end: generate({})
 // (op 469) NOP(827) END
 // (op 469) Assign VGPR 1:U32(836) BEGIN
 // (op 836) Assign dim(567) = 1:U32
 // (op 836) Assign stride expression
 // (op 469) Assign VGPR 1:U32(836) END
 // (op 469) NOP(913) BEGIN
 // (op 913) generate({})
 // (op 913) end: generate({})
 // (op 469) NOP(913) END
 // (op 469) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(918) BEGIN
 // (op 918) Assign dim(590) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 590: v**UNALLOCATED**
 // (op 918) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag590: VGPR Value: UInt32 x 1: (unallocated)
 // (op 918) reg expression
 // (op 918) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 918): v10
v_and_b32 v10, -8, v1 // (op 918) 
 // (op 918) BEGIN: Tile[1]
 // (op 918) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 918): v12
v_sub_u32 v12, v1, v10 // (op 918) 
 // (op 918) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 918): v10

 // (op 918) BEGIN: Flatten
 // (op 918) {Flatten: ShiftL({Tile[1]: v12:U32}, 4:U32)U32}
 // (op 918) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 918)         value (Tile[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 918)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 918): v10
v_lshlrev_b32 v10, 4, v12 // (op 918) 
 // (op 918) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 918): v12

 // (op 918) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 918): v12
v_lshrrev_b32 v12, 3, v1 // (op 918) 
 // (op 918) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 918): v29
v_lshl_add_u32 v29, v12, 7, v10 // (op 918) 
 // (op 918) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 918): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 918): v12

 // (op 918) Multiply({Flatten: v29:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 918): v10
v_mul_lo_u32 v10, v29, 6 // (op 918) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 918): v29

 // (op 918) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag590: 1 VGPR (Value: UInt32) (op 918): v12
v_lshrrev_b32 v12, 3, v10 // (op 918) 
 // Freeing : 1 VGPR (Value: UInt32) (op 918): v10

 // (op 469) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(918) END
 // (op 469) Assign VGPR 96:U32(919) BEGIN
 // (op 919) Assign dim(591) = 96:U32
 // (op 919) Assign stride expression
 // (op 469) Assign VGPR 96:U32(919) END
 // (op 469) NOP(914) BEGIN
 // (op 914) generate({})
 // (op 914) end: generate({})
 // (op 469) NOP(914) END
 // (op 469) Assign VGPR 0:U32(923) BEGIN
 // (op 923) Assign dim(593) = 0:U32
 // (op 923) Assign stride expression
 // (op 469) Assign VGPR 0:U32(923) END
 // (op 469) NOP(937) BEGIN
 // (op 937) generate({})
 // (op 937) end: generate({})
 // (op 469) NOP(937) END
 // (op 469) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(942) BEGIN
 // (op 942) Assign dim(598) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 598: s**UNALLOCATED**
 // (op 942) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag598: SGPR Value: UInt32 x 1: (unallocated)
 // (op 942) reg expression
 // (op 942) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 942): v10
v_and_b32 v10, -8, v1 // (op 942) 
 // (op 942) BEGIN: Tile[1]
 // (op 942) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 942): v29
v_sub_u32 v29, v1, v10 // (op 942) 
 // (op 942) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 942): v10

 // (op 942) BEGIN: Flatten
 // (op 942) {Flatten: ShiftL({Tile[1]: v29:U32}, 4:U32)U32}
 // (op 942) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 942)         value (Tile[1]: VGPR Value: UInt32 x 1: v29) 
 // (op 942)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 942): v10
v_lshlrev_b32 v10, 4, v29 // (op 942) 
 // (op 942) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 942): v29

 // (op 942) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 942): v29
v_lshrrev_b32 v29, 3, v1 // (op 942) 
 // (op 942) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 942): v30
v_lshl_add_u32 v30, v29, 7, v10 // (op 942) 
 // (op 942) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 942): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 942): v29

// Allocated DataFlowTag598: 1 SGPR (Value: UInt32) (op 942): s1
s_nop 0
v_readfirstlane_b32 s1, v30 // (op 942) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 942): v30

 // (op 469) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(942) END
 // (op 469) Assign VGPR 2048:U32(943) BEGIN
 // (op 943) Assign dim(599) = 2048:U32
 // (op 943) Assign stride expression
 // (op 469) Assign VGPR 2048:U32(943) END
 // (op 469) NOP(938) BEGIN
 // (op 938) generate({})
 // (op 938) end: generate({})
 // (op 469) NOP(938) END
 // (op 469) Assign VGPR 1:U32(947) BEGIN
 // (op 947) Assign dim(601) = 1:U32
 // (op 947) Assign stride expression
 // (op 469) Assign VGPR 1:U32(947) END
 // (op 469) NOP(976) BEGIN
 // (op 976) generate({})
 // (op 976) end: generate({})
 // (op 469) NOP(976) END
 // (op 469) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(981) BEGIN
 // (op 981) Assign dim(610) = {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}
 // tag 610: v**UNALLOCATED**
 // (op 981) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32} into DataFlowTag610: VGPR Value: UInt32 x 1: (unallocated)
 // (op 981) reg expression
 // (op 981) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 981): v10
v_and_b32 v10, -4, v1 // (op 981) 
 // (op 981) BEGIN: Tile[1]Flatten
 // (op 981) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 981): v29
v_sub_u32 v29, v1, v10 // (op 981) 
 // (op 981) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 981): v10

 // (op 981) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 981): v10
v_and_b32 v10, 4294967292, v1 // (op 981) 
 // (op 981) BEGIN: Flatten
 // (op 981) {Flatten: Add(v10:U32, {Tile[1]Flatten: v29:U32})U32}
// Allocated DataFlowTag610: 1 VGPR (Value: UInt32) (op 981): v30
v_add_u32 v30, v10, v29 // (op 981) 
 // (op 981) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 981): v29

 // Freeing : 1 VGPR (Value: UInt32) (op 981): v10

 // (op 469) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(981) END
 // (op 469) Assign VGPR 4:U32(982) BEGIN
 // (op 982) Assign dim(611) = 4:U32
 // (op 982) Assign stride expression
 // (op 469) Assign VGPR 4:U32(982) END
 // (op 469) NOP(977) BEGIN
 // (op 977) generate({})
 // (op 977) end: generate({})
 // (op 469) NOP(977) END
 // (op 469) Assign VGPR 1:U32(986) BEGIN
 // (op 986) Assign dim(613) = 1:U32
 // (op 986) Assign stride expression
 // (op 469) Assign VGPR 1:U32(986) END
 // (op 469) ForLoopOp KLoop: LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL(54) BEGIN
 // (op 54) Initialize For Loop
 // (op 54) generate({55})
 // (op 54) Assign SGPR 0:U32(55) BEGIN
 // (op 55) Assign dim(361) = 0:U32
 // tag 361: s**UNALLOCATED**
 // (op 55) Generate 0:U32 into DataFlowTag361: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag361: 1 SGPR (Value: UInt32) (op 55): s14
s_mov_b32 s14, 0 // (op 55) call()
 // (op 54) Assign SGPR 0:U32(55) END
 // (op 54) end: generate({55})
 // (op 54) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 54) 
 // (op 54) Generate LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 54) LessThan({DataFlowTag361: s14:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 54) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32
 // (op 54) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 54): s18, s19
s_ashr_i64 s[18:19], s[28:29], 7 // (op 54) 
// Allocated : 1 SGPR (Value: UInt32) (op 54): s15
s_mov_b32 s15, s18 // (op 54) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 54): s18, s19

// Allocated : 1 SGPR (Value: UInt32) (op 54): s18
s_and_b32 s18, s15, 4294967294 // (op 54) 
 // Freeing : 1 SGPR (Value: UInt32) (op 54): s15

s_cmp_lt_u32 s14, s18 // (op 54) 
 // Freeing : 1 SGPR (Value: UInt32) (op 54): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 54) 
s_cbranch_scc0 RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 // (op 54) Condition: Top (jump to RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54 if false)
RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54:
 // (op 54) 
 // (op 54) generate({56, 1026})
 // (op 54) Assign SGPR Add(DataFlowTag(361)U32, 2:U32)U32(56) BEGIN
 // (op 56) Assign dim(361) = Add(DataFlowTag(361)U32, 2:U32)U32
 // (op 56) Generate Add(DataFlowTag(361)U32, 2:U32)U32 into DataFlowTag361: SGPR Value: UInt32 x 1: s14
 // (op 56) reg expression
 // (op 56) Add({DataFlowTag361: s14:U32}, 2:U32)U32
 // (op 54) Barrier(1026) BEGIN
s_add_u32 s14, s14, 2 // (op 56) 
 // (op 54) Assign SGPR Add(DataFlowTag(361)U32, 2:U32)U32(56) END
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1026) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 54) Barrier(1026) END
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
 // (op 65) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 65): v32, v33, v34, v35, v36, v37
ds_read_b128 v[32:35], v3 offset:8320 // (op 65) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 1960 (current 1865)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
ds_read_b64 v[36:37], v3 offset:8336 // (op 65) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 1961 (current 1866)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 63) LoadLDSTile{Value: FP6x16}(65) END
 // (op 63) end: generate({65})
 // (op 54) SetCoordinate(63) END
 // (op 54) Barrier(1039) BEGIN
s_barrier  // (op 1039) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 54) Barrier(1039) END
 // (op 54) SetCoordinate(68) BEGIN
 // (op 68) generate({})
 // (op 68) end: generate({})
 // (op 68) generate({70})
 // (op 68) LoadLDSTile{Value: FP8x4}(70) BEGIN
 // (op 70) GEN: loadMacroTileWAVELDS OP 70 LDS 24 WaveTile 165
 // (op 70) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 70) Generate 1:U32 into nullptr
 // tag 22: v**UNALLOCATED**
 // (op 70) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v4
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 64

// Allocated : 8 VGPRs (Value: FP8x4) (op 70): v38, v39, v40, v41, v42, v43, v44, v45
ds_read_b128 v[38:41], v4 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected complete at 1963 (current 1868)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
ds_read_b128 v[42:45], v4 offset:64 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected stall of 92. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2056 (current 1961)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 68) LoadLDSTile{Value: FP8x4}(70) END
 // (op 68) end: generate({70})
 // (op 54) SetCoordinate(68) END
 // (op 54) Barrier(1052) BEGIN
s_barrier  // (op 1052) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 54) Barrier(1052) END
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
ds_read_u8 v10, v0 offset:8192 // (op 79) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 2058 (current 1963)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 77) LoadLDSTile{Value: E8M0}(79) END
 // (op 77) end: generate({79})
 // (op 54) SetCoordinate(77) END
 // (op 54) Multiply(82) BEGIN
 // (op 82) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile4xFP8, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[38:45], a[0:3], v16, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 82) 
 // (op 54) Multiply(82) END
 // (op 54) Deallocate{}(1094) BEGIN
 // (op 1094) Deallocate 14
 // Freeing : 6 VGPRs (Value: FP6x16) (op 65): v32, v33, v34, v35, v36, v37

 // Deleting tag 14
 // (op 1094) Deallocate 22
 // Freeing : 8 VGPRs (Value: FP8x4) (op 70): v38, v39, v40, v41, v42, v43, v44, v45

 // Deleting tag 22
 // (op 1094) Deallocate 30
 // Freeing : 1 VGPR (Value: E8M0) (op 79): v10

 // Deleting tag 30
 // (op 54) Deallocate{}(1094) END
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
v_mov_b32 v10, v18 // (op 75) 
 // (op 75) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
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

buffer_load_ubyte v16, v10, s[48:51], 0 offen offset:8 // (op 75) Load value
// VMEM: Expected complete at 2396 (current 1916)
 // (op 264) LoadTiled Value: E8M0(75) END
 // (op 264) end: generate({75})
 // (op 73) SetCoordinate(264) END
 // (op 73) end: generate({264})
 // (op 54) SetCoordinate(73) END
 // (op 54) Barrier(1042) BEGIN
s_barrier  // (op 1042) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 54) Barrier(1042) END
 // (op 54) SetCoordinate(267) BEGIN
 // (op 267) generate({})
 // (op 267) end: generate({})
 // (op 267) generate({416})
 // (op 267) LoadTileDirect2LDS Value: FP8x4(416) BEGIN
 // (op 416) GEN: LoadTileDirect2LDS
 // (op 416) GEN: loadMacroTileDirect2LDS OP 416 LDS 24 MacroTile 9
 // (op 416) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 416) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride485: VGPR Value: UInt64 x 1: (unallocated)
 // (op 416) reg expression
 // (op 416) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 416) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 416)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 416)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 416): s18, s19
s_lshl_b64 s[18:19], s[8:9], 4 // (op 416) 
// Allocated Stride485: 2 VGPRs (Value: UInt64) (op 416): v32, v33
v_mov_b64 v[32:33], s[18:19] // (op 416) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 416): s18, s19

 // (op 416) Generate 1:U64 into nullptr
 // (op 416) Generate 2048:U32 into nullptr
 // (op 416) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride485: VGPR Value: UInt64 x 1: v[32:33]
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 484: v**UNALLOCATED**
// Allocated Offset484: 2 VGPRs (Value: UInt64) (op 416): v34, v35
v_mov_b64 v[34:35], v[14:15] // (op 416) 
// Allocated : 2 VGPRs (Value: UInt64) (op 416): v36, v37
v_mov_b64 v[36:37], v[34:35] // (op 416) 
 // (op 416) Generate Add(0:U32, {DataFlowTag548Flatten: s0:U32})U32 into nullptr
 // (op 416)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride485: v[32:33]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 416) Generate {DataFlowTag548Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 416) reg expression
 // (op 416) Generate 256:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 256 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 416): s15
s_mov_b32 s15, 256 // (op 416) call()
// 256 is not a supported value!
buffer_load_dwordx4 v36, s[44:47], s15 offen offset: 0 lds // (op 416) Load value direct to lds
// VMEM: Expected complete at 2404 (current 1924)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 416) Generate Add(v[36:37]:U64, v32:R)U64 into VGPR Value: UInt64 x 1: v[36:37]
 // (op 416) reg expression
 // (op 416) reg expression
 // (op 416) Add(v[36:37]:U64, v32:R)U64
 // (op 416) Add: dest (VGPR Value: UInt64 x 1: v[36:37]) = 
 // (op 416)      lhs (VGPR Value: UInt64 x 1: v[36:37]) 
 // (op 416)      rhs (VGPR Value: Raw32 x 1: v32)
// Allocated : 2 SGPRs (Value: Bool64) (op 416): s18, s19
v_add_co_u32 v36, s[18:19], v36, v32 // (op 416) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 416) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 416): s18, s19

 // (op 416) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 416) reg expression
 // (op 416) Add(m0:U32, 2048:U32)U32
 // (op 416) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 416) 
 // (op 416) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v36, s[44:47], s15 offen offset: 0 lds // (op 416) Load value direct to lds
// VMEM: Expected stall of 305. CBNW: 4, Inc: 3
// VMEM: Expected complete at 2713 (current 2233)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 416): s15

 // Freeing Stride485: 2 VGPRs (Value: UInt64) (op 416): v32, v33

 // Freeing : 2 VGPRs (Value: UInt64) (op 416): v36, v37

 // (op 267) LoadTileDirect2LDS Value: FP8x4(416) END
 // (op 267) Deallocate{}(1168) BEGIN
 // (op 1168) Deallocate 550
 // Deleting tag 550
 // (op 267) Deallocate{}(1168) END
 // (op 267) end: generate({416})
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
// Allocated Offset522: 1 VGPR (Value: UInt32) (op 21): v29
v_mov_b32 v29, v21 // (op 21) 
 // (op 21) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = Offset522: VGPR Value: UInt32 x 1: v29
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

buffer_load_ubyte v19, v29, s[52:55], 0 offen offset:8 // (op 21) Load value
// VMEM: Expected stall of 162. CBNW: 5, Inc: 3
// VMEM: Expected complete at 2877 (current 2397)
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
 // 	info.data = VGPR Value: FP6x16 x 1: v[22:24]
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
// Allocated Offset465: 1 VGPR (Value: UInt32) (op 4): v31
v_mov_b32 v31, v11 // (op 4) 
 // (op 4) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[22:24]
 // 	info.rowOffsetReg = Offset465: VGPR Value: UInt32 x 1: v31
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

buffer_load_dwordx3 v[22:24], v31, s[40:43], 0 offen offset:192 // (op 4) Load value
// VMEM: Expected stall of 6. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2885 (current 2405)
 // (op 259) LoadTiled Value: FP6x16(4) END
 // (op 259) end: generate({4})
 // (op 54) SetCoordinate(259) END
 // (op 54) NOP(430) BEGIN
 // (op 430) generate({})
 // (op 430) end: generate({})
 // (op 54) NOP(430) END
 // (op 54) Barrier(1081) BEGIN
s_barrier  // (op 1081) 
 // (op 54) Barrier(1081) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 9856
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
 // 	info.data = VGPR Value: E8M0 x 1: v17
 // 	info.rowOffsetReg = DataFlowTag610: VGPR Value: UInt32 x 1: v30
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
 // 	info.offset = Literal Value: UInt32 x 0: 9856
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
ds_write_b8 v30, v17 offset:9856 // (op 236) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 2649 (current 2554)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 9856, Size: 128)
 // (op 54) StoreLDSTile Value: E8M0(236) END
 // (op 54) Deallocate{}(1144) BEGIN
 // (op 1144) Deallocate 612
 // Deleting tag 612
 // (op 54) Deallocate{}(1144) END
 // (op 54) Barrier(1069) BEGIN
s_barrier  // (op 1069) 
 // (op 54) Barrier(1069) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 9984
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
 // 	info.data = VGPR Value: FP6x16 x 1: v[26:28]
 // 	info.rowOffsetReg = DataFlowTag590: VGPR Value: UInt32 x 1: v12
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
 // 	info.offset = Literal Value: UInt32 x 0: 9984
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
ds_write_b96 v12, v[26:28] offset:9984 // (op 225) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 2652 (current 2557)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 9984, Size: 1536)
 // (op 54) StoreLDSTile Value: FP6x16(225) END
 // (op 54) Deallocate{}(1124) BEGIN
 // (op 1124) Deallocate 592
 // Deleting tag 592
 // (op 54) Deallocate{}(1124) END
 // (op 54) NOP(378) BEGIN
 // (op 378) generate({})
 // (op 378) end: generate({})
 // (op 54) NOP(378) END
 // (op 54) Barrier(1066) BEGIN
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1066) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 9984, Size: 1536)
 // (op 54) Barrier(1066) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 9984
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
 // 	info.offset = Literal Value: UInt32 x 0: 9984
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

// Allocated : 6 VGPRs (Value: FP6x16) (op 227): v36, v37, v38, v39, v40, v41
ds_read_b128 v[36:39], v6 offset:9984 // (op 227) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 2750 (current 2655)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 9984, Size: 1536)
ds_read_b64 v[40:41], v6 offset:10000 // (op 227) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 2751 (current 2656)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 9984, Size: 1536)
 // (op 226) LoadLDSTile{Value: FP6x16}(227) END
 // (op 226) Deallocate{}(1130) BEGIN
 // (op 1130) Deallocate 596
 // Deleting tag 596
 // (op 226) Deallocate{}(1130) END
 // (op 226) end: generate({227})
 // (op 54) SetCoordinate(226) END
 // (op 54) Deallocate{}(1126) BEGIN
 // (op 1126) Deallocate 432
 // Deleting tag 432
 // (op 54) Deallocate{}(1126) END
 // (op 54) Barrier(1072) BEGIN
 // (op 54) Barrier(1128) BEGIN
s_barrier  // (op 1072) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 54) Barrier(1072) END
s_barrier  // (op 1128) 
 // (op 54) Barrier(1128) END
 // (op 54) SetCoordinate(228) BEGIN
 // (op 228) generate({})
 // (op 228) end: generate({})
 // (op 228) generate({229})
 // (op 228) LoadLDSTile{Value: FP8x4}(229) BEGIN
 // (op 229) GEN: loadMacroTileWAVELDS OP 229 LDS 436 WaveTile 165
 // (op 229) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 229) Generate 1:U32 into nullptr
 // tag 444: v**UNALLOCATED**
 // (op 229) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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

 // (op 229) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag606: VGPR Value: UInt32 x 1: v7
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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
 // (op 229) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 64

// Allocated : 8 VGPRs (Value: FP8x4) (op 229): v42, v43, v44, v45, v46, v47, v48, v49
ds_read_b128 v[42:45], v7 offset:4096 // (op 229) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected complete at 2754 (current 2659)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 4096)
ds_read_b128 v[46:49], v7 offset:4160 // (op 229) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected stall of 91. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2846 (current 2751)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 228) LoadLDSTile{Value: FP8x4}(229) END
 // (op 228) Deallocate{}(1132) BEGIN
 // (op 1132) Deallocate 608
 // Deleting tag 608
 // (op 228) Deallocate{}(1132) END
 // (op 228) end: generate({229})
 // (op 54) SetCoordinate(228) END
 // (op 54) Barrier(1078) BEGIN
s_barrier  // (op 1078) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 9856, Size: 128)
 // (op 54) Barrier(1078) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 9856
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
 // 	info.rowOffsetReg = DataFlowTag614: VGPR Value: UInt32 x 1: v5
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
 // 	info.offset = Literal Value: UInt32 x 0: 9856
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

// Allocated : 1 VGPR (Value: E8M0) (op 233): v32
ds_read_u8 v32, v5 offset:9856 // (op 233) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 2848 (current 2753)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 9856, Size: 128)
 // (op 232) LoadLDSTile{Value: E8M0}(233) END
 // (op 232) Deallocate{}(1140) BEGIN
 // (op 1140) Deallocate 616
 // Deleting tag 616
 // (op 232) Deallocate{}(1140) END
 // (op 232) end: generate({233})
 // (op 54) SetCoordinate(232) END
 // (op 54) Deallocate{}(1136) BEGIN
 // (op 1136) Deallocate 440
 // Deleting tag 440
 // (op 54) Deallocate{}(1136) END
 // (op 54) Multiply(234) BEGIN
 // (op 234) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile4xFP8, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
 // (op 54) Barrier(1138) BEGIN
s_barrier  // (op 1138) 
 // (op 54) Barrier(1138) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v25, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 234) 
 // (op 54) Multiply(234) END
 // (op 54) Deallocate{}(1142) BEGIN
 // (op 1142) Deallocate 448
 // Freeing : 1 VGPR (Value: E8M0) (op 233): v32

 // Deleting tag 448
 // (op 1142) Deallocate 444
 // Freeing : 8 VGPRs (Value: FP8x4) (op 229): v42, v43, v44, v45, v46, v47, v48, v49

 // Deleting tag 444
 // (op 1142) Deallocate 442
 // Freeing : 6 VGPRs (Value: FP6x16) (op 227): v36, v37, v38, v39, v40, v41

 // Deleting tag 442
 // (op 54) Deallocate{}(1142) END
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
 // 	info.data = VGPR Value: E8M0 x 1: v25
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
 // 	info.data = VGPR Value: E8M0 x 1: v25
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

buffer_load_ubyte v25, v10, s[48:51], 0 offen offset:12 // (op 231) Load value
// VMEM: Expected stall of 198. CBNW: 2, Inc: 3
// VMEM: Expected complete at 3194 (current 2714)
 // (op 283) LoadTiled Value: E8M0(231) END
 // (op 283) end: generate({231})
 // (op 230) SetCoordinate(283) END
 // (op 230) end: generate({283})
 // (op 54) SetCoordinate(230) END
 // (op 54) Deallocate{}(1134) BEGIN
 // (op 1134) Deallocate 503
 // Freeing Offset503: 1 VGPR (Value: UInt32) (op 75): v10

 // Deleting tag 503
 // (op 1134) Deallocate 505
 // Deleting tag 505
 // (op 54) Deallocate{}(1134) END
 // (op 54) Assign VGPR Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32(616) BEGIN
 // (op 616) Assign dim(501) = Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32
 // (op 616) Generate Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32 into DataFlowTag501: VGPR Value: UInt32 x 1: v18
 // (op 616) reg expression
 // (op 616) Add({DataFlowTag501: v18:U32}, 8:U32)U32
 // (op 54) Barrier(1075) BEGIN
s_barrier  // (op 1075) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 54) Barrier(1075) END
v_add_u32 v18, 8, v18 // (op 616) 
 // (op 54) Assign VGPR Add(DataFlowTag(501)U32, ShiftL(Convert(DataFlowTag(502)U64)U32, 1:U32)U32)U32(616) END
 // (op 54) SetCoordinate(286) BEGIN
 // (op 286) generate({})
 // (op 286) end: generate({})
 // (op 286) generate({428})
 // (op 286) LoadTileDirect2LDS Value: FP8x4(428) BEGIN
 // (op 428) GEN: LoadTileDirect2LDS
 // (op 428) GEN: loadMacroTileDirect2LDS OP 428 LDS 436 MacroTile 434
 // (op 428) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 428) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride485: VGPR Value: UInt64 x 1: (unallocated)
 // (op 428) reg expression
 // (op 428) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 428) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 428)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 428)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 428): s18, s19
s_lshl_b64 s[18:19], s[8:9], 4 // (op 428) 
// Allocated Stride485: 2 VGPRs (Value: UInt64) (op 428): v32, v33
v_mov_b64 v[32:33], s[18:19] // (op 428) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 428): s18, s19

 // (op 428) Generate 1:U64 into nullptr
 // (op 428) Generate 2048:U32 into nullptr
 // (op 428) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 4096
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride485: VGPR Value: UInt64 x 1: v[32:33]
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 428): v36, v37
v_mov_b64 v[36:37], v[34:35] // (op 428) 
 // (op 428) Generate Add(4096:U32, {DataFlowTag598Flatten: s1:U32})U32 into nullptr
 // (op 428) reg expression
 // (op 428) Add(4096:U32, {DataFlowTag598Flatten: s1:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 428): s15
s_add_u32 s15, 4096, s1 // (op 428) 
 // (op 428)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride485: v[32:33]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 428) Generate s15:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s15 // (op 428) reg expression
 // (op 428) Generate 384:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 384 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 428): s18
s_mov_b32 s18, 384 // (op 428) call()
// 384 is not a supported value!
buffer_load_dwordx4 v36, s[44:47], s18 offen offset: 0 lds // (op 428) Load value direct to lds
// VMEM: Expected stall of 155. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3358 (current 2878)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 428) Generate Add(v[36:37]:U64, v32:R)U64 into VGPR Value: UInt64 x 1: v[36:37]
 // (op 428) reg expression
 // (op 428) reg expression
 // (op 428) Add(v[36:37]:U64, v32:R)U64
 // (op 428) Add: dest (VGPR Value: UInt64 x 1: v[36:37]) = 
 // (op 428)      lhs (VGPR Value: UInt64 x 1: v[36:37]) 
 // (op 428)      rhs (VGPR Value: Raw32 x 1: v32)
// Allocated : 2 SGPRs (Value: Bool64) (op 428): s38, s39
v_add_co_u32 v36, s[38:39], v36, v32 // (op 428) least significant half
v_addc_co_u32 v37, s[38:39], 0, v37, s[38:39] // (op 428) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 428): s38, s39

 // (op 428) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 428) reg expression
 // (op 428) Add(m0:U32, 2048:U32)U32
 // (op 428) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 428) 
 // (op 428) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v36, s[44:47], s18 offen offset: 0 lds // (op 428) Load value direct to lds
// VMEM: Expected stall of 4. CBNW: 4, Inc: 3
// VMEM: Expected complete at 3366 (current 2886)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 428): s18

 // Freeing Stride485: 2 VGPRs (Value: UInt64) (op 428): v32, v33

 // Freeing : 2 VGPRs (Value: UInt64) (op 428): v36, v37

 // Freeing : 1 SGPR (Value: UInt32) (op 428): s15

 // (op 286) LoadTileDirect2LDS Value: FP8x4(428) END
 // (op 286) Deallocate{}(1172) BEGIN
 // (op 1172) Deallocate 600
 // Deleting tag 600
 // (op 286) Deallocate{}(1172) END
 // (op 286) end: generate({428})
 // (op 54) SetCoordinate(286) END
 // (op 54) Deallocate{}(1148) BEGIN
 // (op 1148) Deallocate 484
 // Freeing Offset484: 2 VGPRs (Value: UInt64) (op 416): v34, v35

 // Deleting tag 484
 // (op 1148) Deallocate 486
 // Deleting tag 486
 // (op 54) Deallocate{}(1148) END
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

 // (op 238) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
 // 	info.rowOffsetReg = Offset522: VGPR Value: UInt32 x 1: v29
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

 // (op 54) Assign VGPR Add(DataFlowTag(482)U64, ShiftL(DataFlowTag(483)U64, 1:U32)U64)U64(552) BEGIN
 // (op 552) Assign dim(482) = Add(DataFlowTag(482)U64, ShiftL(DataFlowTag(483)U64, 1:U32)U64)U64
 // (op 552) Generate Add(DataFlowTag(482)U64, ShiftL(DataFlowTag(483)U64, 1:U32)U64)U64 into DataFlowTag482: VGPR Value: UInt64 x 1: v[14:15]
 // (op 552) reg expression
 // (op 552) Add({DataFlowTag482: v[14:15]:U64}, 256:U64)U64
 // (op 552) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 552)      lhs (DataFlowTag482: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 552)      rhs (Literal Value: UInt64 x 0: 256)
v_add_co_u32 v14, vcc, 256, v14 // (op 552) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 552) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 54) Assign VGPR Add(DataFlowTag(482)U64, ShiftL(DataFlowTag(483)U64, 1:U32)U64)U64(552) END
buffer_load_ubyte v17, v29, s[52:55], 0 offen offset:12 // (op 238) Load value
// VMEM: Expected stall of 306. CBNW: 5, Inc: 3
// VMEM: Expected complete at 3675 (current 3195)
 // (op 291) LoadTiled Value: E8M0(238) END
 // (op 291) end: generate({238})
 // (op 54) SetCoordinate(291) END
 // (op 54) Deallocate{}(1150) BEGIN
 // (op 1150) Deallocate 522
 // Freeing Offset522: 1 VGPR (Value: UInt32) (op 21): v29

 // Deleting tag 522
 // (op 1150) Deallocate 524
 // Deleting tag 524
 // (op 54) Deallocate{}(1150) END
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

 // (op 224) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[26:28]
 // 	info.rowOffsetReg = Offset465: VGPR Value: UInt32 x 1: v31
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

 // (op 54) Assign VGPR Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32(680) BEGIN
 // (op 680) Assign dim(520) = Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32
 // (op 680) Generate Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32 into DataFlowTag520: VGPR Value: UInt32 x 1: v21
 // (op 680) reg expression
 // (op 680) Add({DataFlowTag520: v21:U32}, 8:U32)U32
v_add_u32 v21, 8, v21 // (op 680) 
 // (op 54) Assign VGPR Add(DataFlowTag(520)U32, ShiftL(Convert(DataFlowTag(521)U64)U32, 1:U32)U32)U32(680) END
buffer_load_dwordx3 v[26:28], v31, s[40:43], 0 offen offset:288 // (op 224) Load value
// VMEM: Expected stall of 162. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3839 (current 3359)
 // (op 278) LoadTiled Value: FP6x16(224) END
 // (op 278) end: generate({224})
 // (op 54) SetCoordinate(278) END
 // (op 54) Deallocate{}(1146) BEGIN
 // (op 1146) Deallocate 467
 // Deleting tag 467
 // (op 1146) Deallocate 465
 // Freeing Offset465: 1 VGPR (Value: UInt32) (op 4): v31

 // Deleting tag 465
 // (op 54) Deallocate{}(1146) END
 // (op 54) NOP(418) BEGIN
 // (op 418) generate({})
 // (op 418) end: generate({})
 // (op 54) NOP(418) END
 // (op 54) Assign VGPR Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32(488) BEGIN
 // (op 488) Assign dim(463) = Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32
 // (op 488) Generate Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32 into DataFlowTag463: VGPR Value: UInt32 x 1: v11
 // (op 488) reg expression
 // (op 488) Add({DataFlowTag463: v11:U32}, 192:U32)U32
 // (op 54) Barrier(1055) BEGIN
s_barrier  // (op 1055) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 54) Barrier(1055) END
v_add_u32 v11, 192, v11 // (op 488) 
 // (op 54) Assign VGPR Add(DataFlowTag(463)U32, ShiftL(Convert(DataFlowTag(464)U64)U32, 1:U32)U32)U32(488) END
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

 // (op 26) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag564: VGPR Value: UInt32 x 1: v13
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
ds_write_b8 v13, v19 offset:8192 // (op 26) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 3793 (current 3698)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 54) StoreLDSTile Value: E8M0(26) END
 // (op 54) Deallocate{}(1086) BEGIN
 // (op 1086) Deallocate 566
 // Deleting tag 566
 // (op 54) Deallocate{}(1086) END
 // (op 54) Barrier(1029) BEGIN
s_barrier  // (op 1029) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 54) Barrier(1029) END
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
 // 	info.data = VGPR Value: FP6x16 x 1: v[22:24]
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

 // (op 6) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[22:24]
 // 	info.rowOffsetReg = DataFlowTag532: VGPR Value: UInt32 x 1: v20
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
ds_write_b96 v20, v[22:24] offset:8320 // (op 6) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 3796 (current 3701)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 54) StoreLDSTile Value: FP6x16(6) END
 // (op 54) Deallocate{}(1084) BEGIN
 // (op 1084) Deallocate 534
 // Deleting tag 534
 // (op 54) Deallocate{}(1084) END
 // (op 54) NOP(379) BEGIN
 // (op 379) generate({})
 // (op 379) end: generate({})
 // (op 54) NOP(379) END
 // (op 54) end: generate({56, 1026})
 // (op 54) For Loop Increment
 // (op 54) generate({})
 // (op 54) end: generate({})
 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 if true)
 // (op 54) Generate LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 54) LessThan({DataFlowTag361: s14:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 54) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32
 // (op 54) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 54): s18, s19
s_ashr_i64 s[18:19], s[28:29], 7 // (op 54) 
// Allocated : 1 SGPR (Value: UInt32) (op 54): s15
s_mov_b32 s15, s18 // (op 54) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 54): s18, s19

// Allocated : 1 SGPR (Value: UInt32) (op 54): s18
s_and_b32 s18, s15, 4294967294 // (op 54) 
 // Freeing : 1 SGPR (Value: UInt32) (op 54): s15

s_cmp_lt_u32 s14, s18 // (op 54) 
 // Freeing : 1 SGPR (Value: UInt32) (op 54): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 54) 
s_cbranch_scc1 RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 // (op 54) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_54 if true)
RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_54:
 // (op 54) 
 // (op 54) Unlock For Loop
 // Unlock None
 // (op 469) ForLoopOp KLoop: LessThan(DataFlowTag(361)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL(54) END
 // (op 469) Deallocate{}(1090) BEGIN
 // (op 1090) Deallocate 434
 // Deleting tag 434
 // (op 1090) Deallocate 502
 // Deleting tag 502
 // (op 1090) Deallocate 590
 // Freeing DataFlowTag590: 1 VGPR (Value: UInt32) (op 918): v12

 // Deleting tag 590
 // (op 1090) Deallocate 463
 // Freeing DataFlowTag463: 1 VGPR (Value: UInt32) (op 496): v11

 // Deleting tag 463
 // (op 1090) Deallocate 593
 // Deleting tag 593
 // (op 1090) Deallocate 466
 // Deleting tag 466
 // (op 1090) Deallocate 468
 // Deleting tag 468
 // (op 1090) Deallocate 599
 // Deleting tag 599
 // (op 1090) Deallocate 482
 // Freeing DataFlowTag482: 2 VGPRs (Value: UInt64) (op 560): v14, v15

 // Deleting tag 482
 // (op 1090) Deallocate 610
 // Freeing DataFlowTag610: 1 VGPR (Value: UInt32) (op 981): v30

 // Deleting tag 610
 // (op 1090) Deallocate 483
 // Deleting tag 483
 // (op 1090) Deallocate 487
 // Deleting tag 487
 // (op 1090) Deallocate 361
 // Freeing DataFlowTag361: 1 SGPR (Value: UInt32) (op 55): s14

 // Deleting tag 361
 // (op 1090) Deallocate 501
 // Freeing DataFlowTag501: 1 VGPR (Value: UInt32) (op 624): v18

 // Deleting tag 501
 // (op 1090) Deallocate 485
 // Deleting tag 485
 // (op 1090) Deallocate 565
 // Deleting tag 565
 // (op 1090) Deallocate 506
 // Deleting tag 506
 // (op 1090) Deallocate 533
 // Deleting tag 533
 // (op 1090) Deallocate 535
 // Deleting tag 535
 // (op 1090) Deallocate 548
 // Freeing DataFlowTag548: 1 SGPR (Value: UInt32) (op 779): s0

 // Deleting tag 548
 // (op 1090) Deallocate 549
 // Deleting tag 549
 // (op 1090) Deallocate 551
 // Deleting tag 551
 // (op 1090) Deallocate 564
 // Freeing DataFlowTag564: 1 VGPR (Value: UInt32) (op 831): v13

 // Deleting tag 564
 // (op 1090) Deallocate 567
 // Deleting tag 567
 // (op 1090) Deallocate 598
 // Freeing DataFlowTag598: 1 SGPR (Value: UInt32) (op 942): s1

 // Deleting tag 598
 // (op 1090) Deallocate 601
 // Deleting tag 601
 // (op 1090) Deallocate 611
 // Deleting tag 611
 // (op 1090) Deallocate 613
 // Deleting tag 613
 // (op 1090) Deallocate 438
 // Deleting tag 438
 // (op 1090) Deallocate 446
 // Deleting tag 446
 // (op 1090) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 344): v19

 // Deleting tag 10
 // (op 1090) Deallocate 532
 // Freeing DataFlowTag532: 1 VGPR (Value: UInt32) (op 726): v20

 // Deleting tag 532
 // (op 1090) Deallocate 430
 // Deleting tag 430
 // (op 1090) Deallocate 49
 // Freeing : 1 VGPR (Value: E8M0) (op 337): v16

 // Deleting tag 49
 // (op 1090) Deallocate 525
 // Deleting tag 525
 // (op 1090) Deallocate 523
 // Deleting tag 523
 // (op 1090) Deallocate 9
 // Deleting tag 9
 // (op 1090) Deallocate 521
 // Deleting tag 521
 // (op 1090) Deallocate 8
 // Freeing : 3 VGPRs (Value: FP6x16) (op 347): v22, v23, v24

 // Deleting tag 8
 // (op 1090) Deallocate 520
 // Freeing DataFlowTag520: 1 VGPR (Value: UInt32) (op 688): v21

 // Deleting tag 520
 // (op 1090) Deallocate 436
 // Deleting tag 436
 // (op 1090) Deallocate 504
 // Deleting tag 504
 // (op 1090) Deallocate 591
 // Deleting tag 591
 // (op 1090) Deallocate 464
 // Deleting tag 464
 // (op 469) Deallocate{}(1090) END
 // (op 469) Barrier(1092) BEGIN
s_barrier  // (op 1092) 
 // (op 469) Barrier(1092) END
 // (op 469) end: generate({464})
 // (op 469) Unlock Scope 469
 // Unlock None
 // (op 310) Scope(469) END
 // (op 310) SetCoordinate(201) BEGIN
 // (op 201) generate({})
 // (op 201) end: generate({})
 // (op 201) generate({202})
 // (op 201) SetCoordinate(202) BEGIN
 // (op 202) generate({})
 // (op 202) end: generate({})
 // (op 202) generate({446})
 // (op 202) Scope(446) BEGIN
 // (op 446) Lock Scope 446
 // Lock Branch
 // (op 446) generate({440})
 // (op 446) NOP(440) BEGIN
 // (op 440) generate({})
 // (op 440) end: generate({})
 // (op 446) NOP(440) END
 // (op 446) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(451) BEGIN
 // (op 451) Assign dim(450) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 450: v**UNALLOCATED**
 // (op 451) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag450: VGPR Value: UInt32 x 1: (unallocated)
 // (op 451) reg expression
 // (op 451) reg expression
 // (op 451) reg expression
 // (op 451) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
 // (op 451) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 451): v10
v_lshrrev_b32 v10, 3, v1 // (op 451) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 451): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 451) 
 // (op 451) BEGIN: Tile
 // (op 451) convert
// Allocated : 1 VGPR (Value: UInt32) (op 451): v11
v_lshl_add_u32 v11, s16, 4, v10 // (op 451) 
 // (op 451) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 451): v10

 // (op 451) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 451) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 451): s14
s_lshr_b32 s14, s0, 1 // (op 451) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 451): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 451): v10
v_and_b32 v10, -8, v1 // (op 451) 
 // (op 451) reg expression
 // (op 451) BEGIN: Flatten[1]
 // (op 451) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 451): v12
v_sub_u32 v12, v1, v10 // (op 451) 
 // (op 451) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 451): v10

 // (op 451) BEGIN: Tile
 // (op 451) {Tile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 451) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 451)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 451)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 451): v10
v_lshlrev_b32 v10, 4, v12 // (op 451) 
 // (op 451) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 451): v12

 // (op 451) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 451): v12
v_lshl_add_u32 v12, s14, 8, v10 // (op 451) 
 // (op 451) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 451): v10

 // Freeing : 1 SGPR (Value: UInt32) (op 451): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 451): v18, v19
v_mov_b32 v19, 0 // (op 451) convert
v_mov_b32 v18, v12 // (op 451) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 451): v12

 // (op 451) BEGIN: Split
 // (op 451) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 451)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 451)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 451): v12, v13
v_mul_lo_u32 v13, v11, s3 // (op 451) most significant: low of low * high
 // (op 451) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 451): v10
v_mul_hi_u32 v10, v11, s2 // (op 451) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 451) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 451) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 451): v10

 // (op 451) Add: dest (VGPR Value: UInt64 x 1: v[12:13]) = 
 // (op 451)      lhs (VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 451)      rhs (Tile: VGPR Value: UInt64 x 1: v[18:19])
// Allocated : 2 SGPRs (Value: Bool64) (op 451): s0, s1
v_add_co_u32 v12, s[0:1], v12, v18 // (op 451) least significant half
v_addc_co_u32 v13, s[0:1], v13, v19, s[0:1] // (op 451) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 451): s0, s1

 // (op 451) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 451): v18, v19

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 451): v11

 // (op 451) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 451) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 451)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 451)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 451) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 451): v10, v11
v_mul_lo_u32 v11, v13, 6 // (op 451) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 451): v14
v_mul_hi_u32 v14, v12, 6 // (op 451) most significant: high of low * low
v_mul_lo_u32 v10, v12, 6 // (op 451) least significant: low of low * low
v_add_u32 v11, v11, v14 // (op 451) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 451): v14

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 451): v12, v13

 // (op 451) LogicalShiftR(v[10:11]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 451): v12, v13
v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 451) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 451): v10, v11

// Allocated DataFlowTag450: 1 VGPR (Value: UInt32) (op 451): v10
v_mov_b32 v10, v12 // (op 451) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 451): v12, v13

 // (op 446) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(393)U32, 4:U32, LogicalShiftR(DataFlowTag(72)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(72)U32, BitwiseAnd(DataFlowTag(72)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(451) END
 // (op 446) Assign VGPR 96:U64(452) BEGIN
 // (op 452) Assign dim(451) = 96:U64
 // (op 452) Assign stride expression
 // (op 446) Assign VGPR 96:U64(452) END
 // (op 446) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(453) BEGIN
 // (op 453) Assign dim(453) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 453) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag453: SGPR Buffer: None x 1: s[40:43]
 // (op 453) reg expression
 // (op 453) reg expression
 // (op 453) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 453) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 453)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 453)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 453) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 453): s0, s1
s_mul_i32 s1, s7, 6 // (op 453) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 453) least significant
// Allocated : 1 SGPR (Value: Int32) (op 453): s14
s_mul_hi_u32 s14, s6, 6 // (op 453) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 453) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 453): s14

 // (op 453) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 453) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 453)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 453)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 453): s14, s15
s_mul_i32 s15, s2, s1 // (op 453) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 453): s18
s_mul_i32 s18, s3, s0 // (op 453) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 453) least significant
// Allocated : 1 SGPR (Value: Int32) (op 453): s19
s_mul_hi_u32 s19, s2, s0 // (op 453) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 453) most significant: sum
s_add_u32 s15, s15, s19 // (op 453) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 453): s19

 // Freeing : 1 SGPR (Value: Int32) (op 453): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 453): s0, s1

 // (op 453) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 453): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 453) 
 // Freeing : 2 SGPRs (Value: Int64) (op 453): s14, s15

 // (op 453) reg expression
 // (op 453) reg expression
s_mov_b32 s40, s4 // (op 453) reg expression
s_mov_b32 s41, s5 // (op 453) reg expression
s_mov_b32 s42, s0 // (op 453) reg expression
s_mov_b32 s43, 131072 // (op 453) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 453): s0, s1

 // (op 446) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(453) END
 // (op 446) NOP(442) BEGIN
 // (op 442) generate({})
 // (op 442) end: generate({})
 // (op 446) NOP(442) END
 // (op 446) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(458) BEGIN
 // (op 458) Assign dim(455) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 458) Assign stride expression
 // (op 446) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(458) END
 // (op 446) NOP(443) BEGIN
 // (op 443) generate({})
 // (op 443) end: generate({})
 // (op 446) NOP(443) END
 // (op 446) Assign VGPR 0:U64(461) BEGIN
 // (op 461) Assign dim(457) = 0:U64
 // (op 461) Assign stride expression
 // (op 446) Assign VGPR 0:U64(461) END
 // (op 446) NOP(509) BEGIN
 // (op 509) generate({})
 // (op 509) end: generate({})
 // (op 446) NOP(509) END
 // (op 446) Assign VGPR {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(518) BEGIN
 // (op 518) Assign dim(469) = {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 469: v**UNALLOCATED**
 // (op 518) Generate {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag469: VGPR Value: UInt64 x 1: (unallocated)
 // (op 518) reg expression
 // (op 518) reg expression
 // (op 518) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
 // (op 518) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 518): v11
v_and_b32 v11, -8, v1 // (op 518) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 518): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 518) 
 // (op 518) BEGIN: Flatten[1]
 // (op 518) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
 // (op 518) convert
// Allocated : 1 VGPR (Value: UInt32) (op 518): v12
v_sub_u32 v12, v1, v11 // (op 518) 
 // (op 518) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 518): v11

 // (op 518) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 518) BEGIN: Tile
 // (op 518) {Tile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 518) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 518)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 518)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 SGPR (Value: UInt32) (op 518): s14
s_lshr_b32 s14, s0, 1 // (op 518) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 518): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 518): v11
v_lshlrev_b32 v11, 4, v12 // (op 518) 
 // (op 518) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 518): v12

 // (op 518) reg expression
 // (op 518) BEGIN: Tile
 // (op 518) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 518): v12
v_lshl_add_u32 v12, s14, 8, v11 // (op 518) 
 // (op 518) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 518): v11

 // Freeing : 1 SGPR (Value: UInt32) (op 518): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 518): v18, v19
v_mov_b32 v19, 0 // (op 518) convert
v_mov_b32 v18, v12 // (op 518) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 518): v12

 // (op 518) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 518): v11
v_lshrrev_b32 v11, 3, v1 // (op 518) 
 // (op 518) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 518): v12
v_lshl_add_u32 v12, s17, 5, v11 // (op 518) 
 // (op 518) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 518): v11

 // (op 518) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 518) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 518)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 518)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 518): v20, v21
v_mul_lo_u32 v21, v12, s9 // (op 518) most significant: low of low * high
 // (op 518) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 518): v11
v_mul_hi_u32 v11, v12, s8 // (op 518) most significant: high of low * low
v_mul_lo_u32 v20, v12, s8 // (op 518) least significant: low of low * low
v_add_u32 v21, v21, v11 // (op 518) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 518): v11

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 518): v12

 // (op 518) BEGIN: Split
 // (op 518) {Split: Add({Tile: v[18:19]:U64}, v[20:21]:I64)U64}
 // (op 518) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 518)      lhs (Tile: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 518)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated DataFlowTag469: 2 VGPRs (Value: UInt64) (op 518): v22, v23
// Allocated : 2 SGPRs (Value: Bool64) (op 518): s0, s1
v_add_co_u32 v22, s[0:1], v18, v20 // (op 518) least significant half
v_addc_co_u32 v23, s[0:1], v19, v21, s[0:1] // (op 518) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 518): s0, s1

 // (op 518) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 518): v20, v21

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 518): v18, v19

 // (op 446) Assign VGPR {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(138)U32, BitwiseAnd(DataFlowTag(138)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(395)U32, 5:U32, LogicalShiftR(DataFlowTag(138)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(518) END
 // (op 446) Assign VGPR 128:U64(519) BEGIN
 // (op 519) Assign dim(470) = 128:U64
 // (op 519) Assign stride expression
 // (op 446) Assign VGPR 128:U64(519) END
 // (op 446) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(520) BEGIN
 // (op 520) Assign dim(472) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 520) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // (op 520) reg expression
 // (op 520) reg expression
 // (op 520) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 520) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 520)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 520)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 520): s0, s1
s_mul_i32 s1, s8, s13 // (op 520) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 520): s14
s_mul_i32 s14, s9, s12 // (op 520) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 520) least significant
// Allocated : 1 SGPR (Value: Int32) (op 520): s15
s_mul_hi_u32 s15, s8, s12 // (op 520) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 520) most significant: sum
s_add_u32 s1, s1, s15 // (op 520) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 520): s15

 // Freeing : 1 SGPR (Value: Int32) (op 520): s14

 // (op 520) reg expression
 // (op 520) reg expression
s_mov_b32 s44, s10 // (op 520) reg expression
s_mov_b32 s45, s11 // (op 520) reg expression
s_mov_b32 s46, s0 // (op 520) reg expression
s_mov_b32 s47, 131072 // (op 520) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 520): s0, s1

 // (op 446) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(520) END
 // (op 446) NOP(511) BEGIN
 // (op 511) generate({})
 // (op 511) end: generate({})
 // (op 446) NOP(511) END
 // (op 446) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(525) BEGIN
 // (op 525) Assign dim(474) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 525) Assign stride expression
 // (op 446) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(525) END
 // (op 446) NOP(512) BEGIN
 // (op 512) generate({})
 // (op 512) end: generate({})
 // (op 446) NOP(512) END
 // (op 446) Assign VGPR 1:U64(528) BEGIN
 // (op 528) Assign dim(476) = 1:U64
 // (op 528) Assign stride expression
 // (op 446) Assign VGPR 1:U64(528) END
 // (op 446) NOP(573) BEGIN
 // (op 573) generate({})
 // (op 573) end: generate({})
 // (op 446) NOP(573) END
 // (op 446) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32})U32}(582) BEGIN
 // (op 582) Assign dim(488) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32})U32}
 // tag 488: v**UNALLOCATED**
 // (op 582) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32})U32} into DataFlowTag488: VGPR Value: UInt32 x 1: (unallocated)
 // (op 582) reg expression
 // (op 582) reg expression
 // (op 582) reg expression
 // (op 582) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 582) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 582): v24
v_lshrrev_b32 v24, 7, v1 // (op 582) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 582): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 582) 
 // (op 582) BEGIN: Flatten[1]
 // (op 582) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 582) convert
// Allocated : 1 VGPR (Value: UInt32) (op 582): v11
v_and_b32 v11, 15, v1 // (op 582) 
 // (op 582) END: Flatten[1]
 // (op 582) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 582) BEGIN: Tile
// Allocated : 1 SGPR (Value: UInt32) (op 582): s14
s_lshr_b32 s14, s0, 1 // (op 582) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 582): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 582): v12
v_lshl_add_u32 v12, v24, 4, v11 // (op 582) 
 // (op 582) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 582): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 582): v24

 // (op 582) reg expression
 // (op 582) BEGIN: Tile
 // (op 582) convert
// Allocated : 1 VGPR (Value: UInt32) (op 582): v11
v_lshl_add_u32 v11, s16, 4, v12 // (op 582) 
 // (op 582) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 582): v12

 // (op 582) BEGIN: Flatten[1]
 // (op 582) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 582): v24
v_and_b32 v24, 63, v1 // (op 582) 
 // (op 582) END: Flatten[1]
 // (op 582) LogicalShiftR({Flatten[1]: v24:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 582): v12
v_lshrrev_b32 v12, 4, v24 // (op 582) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 582): v24

 // (op 582) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 582): v24
v_lshl_add_u32 v24, s14, 3, v12 // (op 582) 
 // (op 582) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 582): v12

 // Freeing : 1 SGPR (Value: UInt32) (op 582): s14

 // (op 582) BEGIN: Split
// Allocated DataFlowTag488: 1 VGPR (Value: UInt32) (op 582): v12
v_mul_lo_u32 v12, v11, s20 // (op 582) 
v_add_u32 v12, v12, v24 // (op 582) 
 // (op 582) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 582): v24

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 582): v11

 // (op 446) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(200)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(200)U32, 63:U32)U32}, 4:U32)U32)U32})U32}(582) END
 // (op 446) Assign VGPR 4:U64(583) BEGIN
 // (op 583) Assign dim(489) = 4:U64
 // (op 583) Assign stride expression
 // (op 446) Assign VGPR 4:U64(583) END
 // (op 446) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(584) BEGIN
 // (op 584) Assign dim(491) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 584) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag491: SGPR Buffer: None x 1: s[48:51]
 // (op 584) reg expression
 // (op 584) reg expression
 // (op 584) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 584) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 584)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 584)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 584): s0, s1
s_mul_i32 s1, s20, s7 // (op 584) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 584): s14
s_mul_i32 s14, s21, s6 // (op 584) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 584) least significant
// Allocated : 1 SGPR (Value: Int32) (op 584): s15
s_mul_hi_u32 s15, s20, s6 // (op 584) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 584) most significant: sum
s_add_u32 s1, s1, s15 // (op 584) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 584): s15

 // Freeing : 1 SGPR (Value: Int32) (op 584): s14

 // (op 584) reg expression
 // (op 584) reg expression
s_mov_b32 s48, s22 // (op 584) reg expression
s_mov_b32 s49, s23 // (op 584) reg expression
s_mov_b32 s50, s0 // (op 584) reg expression
s_mov_b32 s51, 131072 // (op 584) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 584): s0, s1

 // (op 446) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(584) END
 // (op 446) NOP(575) BEGIN
 // (op 575) generate({})
 // (op 575) end: generate({})
 // (op 446) NOP(575) END
 // (op 446) Assign VGPR 4:U64(589) BEGIN
 // (op 589) Assign dim(493) = 4:U64
 // (op 589) Assign stride expression
 // (op 446) Assign VGPR 4:U64(589) END
 // (op 446) NOP(576) BEGIN
 // (op 576) generate({})
 // (op 576) end: generate({})
 // (op 446) NOP(576) END
 // (op 446) Assign VGPR 1:U64(592) BEGIN
 // (op 592) Assign dim(495) = 1:U64
 // (op 592) Assign stride expression
 // (op 446) Assign VGPR 1:U64(592) END
 // (op 446) NOP(637) BEGIN
 // (op 637) generate({})
 // (op 637) end: generate({})
 // (op 446) NOP(637) END
 // (op 446) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(646) BEGIN
 // (op 646) Assign dim(507) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 507: v**UNALLOCATED**
 // (op 646) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag507: VGPR Value: UInt32 x 1: (unallocated)
 // (op 646) reg expression
 // (op 646) reg expression
 // (op 646) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
 // (op 646) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 646): v11
v_and_b32 v11, -4, v1 // (op 646) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 646): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 646) 
 // (op 646) BEGIN: Flatten[1]Tile
 // (op 646) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
 // (op 646) convert
// Allocated : 1 VGPR (Value: UInt32) (op 646): v24
v_sub_u32 v24, v1, v11 // (op 646) 
 // (op 646) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 646): v11

 // (op 646) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 646) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 646): s14
s_lshr_b32 s14, s0, 1 // (op 646) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 646): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 646): v11
v_lshrrev_b32 v11, 2, v1 // (op 646) 
 // (op 646) reg expression
 // (op 646) BEGIN: Tile
 // (op 646) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 646): v13
v_lshl_add_u32 v13, s14, 3, v24 // (op 646) 
 // (op 646) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 646): v24

 // Freeing : 1 SGPR (Value: UInt32) (op 646): s14

 // (op 646) convert
 // (op 646) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 646): v24
v_lshl_add_u32 v24, s17, 5, v11 // (op 646) 
 // (op 646) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 646): v11

 // (op 646) Multiply({Tile: v24:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 646): v11
v_mul_lo_u32 v11, v24, s24 // (op 646) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 646): v24

 // (op 646) BEGIN: Split
 // (op 646) {Split: Add({Tile: v13:U32}, v11:U32)U32}
// Allocated DataFlowTag507: 1 VGPR (Value: UInt32) (op 646): v24
v_add_u32 v24, v13, v11 // (op 646) 
 // (op 646) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 646): v11

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 646): v13

 // (op 446) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(234)U32, BitwiseAnd(DataFlowTag(234)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(234)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(646) END
 // (op 446) Assign VGPR 4:U64(647) BEGIN
 // (op 647) Assign dim(508) = 4:U64
 // (op 647) Assign stride expression
 // (op 446) Assign VGPR 4:U64(647) END
 // (op 446) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(648) BEGIN
 // (op 648) Assign dim(510) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 648) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag510: SGPR Buffer: None x 1: s[52:55]
 // (op 648) reg expression
 // (op 648) reg expression
 // (op 648) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 648) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 648)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 648)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 648): s0, s1
s_mul_i32 s1, s24, s13 // (op 648) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 648): s14
s_mul_i32 s14, s25, s12 // (op 648) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 648) least significant
// Allocated : 1 SGPR (Value: Int32) (op 648): s15
s_mul_hi_u32 s15, s24, s12 // (op 648) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 648) most significant: sum
s_add_u32 s1, s1, s15 // (op 648) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 648): s15

 // Freeing : 1 SGPR (Value: Int32) (op 648): s14

 // (op 648) reg expression
 // (op 648) reg expression
s_mov_b32 s52, s26 // (op 648) reg expression
s_mov_b32 s53, s27 // (op 648) reg expression
s_mov_b32 s54, s0 // (op 648) reg expression
s_mov_b32 s55, 131072 // (op 648) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 648): s0, s1

 // (op 446) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(648) END
 // (op 446) NOP(639) BEGIN
 // (op 639) generate({})
 // (op 639) end: generate({})
 // (op 446) NOP(639) END
 // (op 446) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(653) BEGIN
 // (op 653) Assign dim(512) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 653) Assign stride expression
 // (op 446) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(653) END
 // (op 446) NOP(640) BEGIN
 // (op 640) generate({})
 // (op 640) end: generate({})
 // (op 446) NOP(640) END
 // (op 446) Assign VGPR 1:U64(656) BEGIN
 // (op 656) Assign dim(514) = 1:U64
 // (op 656) Assign stride expression
 // (op 446) Assign VGPR 1:U64(656) END
 // (op 446) NOP(734) BEGIN
 // (op 734) generate({})
 // (op 734) end: generate({})
 // (op 446) NOP(734) END
 // (op 446) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(739) BEGIN
 // (op 739) Assign dim(536) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 536: v**UNALLOCATED**
 // (op 739) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag536: VGPR Value: UInt32 x 1: (unallocated)
 // (op 739) reg expression
 // (op 739) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 739): v11
v_and_b32 v11, -8, v1 // (op 739) 
 // (op 739) BEGIN: Tile[1]
 // (op 739) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 739): v13
v_sub_u32 v13, v1, v11 // (op 739) 
 // (op 739) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 739): v11

 // (op 739) BEGIN: Flatten
 // (op 739) {Flatten: ShiftL({Tile[1]: v13:U32}, 4:U32)U32}
 // (op 739) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 739)         value (Tile[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 739)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 739): v11
v_lshlrev_b32 v11, 4, v13 // (op 739) 
 // (op 739) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 739): v13

 // (op 739) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 739): v13
v_lshrrev_b32 v13, 3, v1 // (op 739) 
 // (op 739) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 739): v14
v_lshl_add_u32 v14, v13, 7, v11 // (op 739) 
 // (op 739) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 739): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 739): v13

 // (op 739) Multiply({Flatten: v14:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 739): v11
v_mul_lo_u32 v11, v14, 6 // (op 739) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 739): v14

 // (op 739) LogicalShiftR(v11:U32, 3:U32)U32
// Allocated DataFlowTag536: 1 VGPR (Value: UInt32) (op 739): v13
v_lshrrev_b32 v13, 3, v11 // (op 739) 
 // Freeing : 1 VGPR (Value: UInt32) (op 739): v11

 // (op 446) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(87)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(87)U32, BitwiseAnd(DataFlowTag(87)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(739) END
 // (op 446) Assign VGPR 96:U32(740) BEGIN
 // (op 740) Assign dim(537) = 96:U32
 // (op 740) Assign stride expression
 // (op 446) Assign VGPR 96:U32(740) END
 // (op 446) NOP(735) BEGIN
 // (op 735) generate({})
 // (op 735) end: generate({})
 // (op 446) NOP(735) END
 // (op 446) Assign VGPR 0:U32(744) BEGIN
 // (op 744) Assign dim(539) = 0:U32
 // (op 744) Assign stride expression
 // (op 446) Assign VGPR 0:U32(744) END
 // (op 446) NOP(787) BEGIN
 // (op 787) generate({})
 // (op 787) end: generate({})
 // (op 446) NOP(787) END
 // (op 446) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(792) BEGIN
 // (op 792) Assign dim(552) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 552: s**UNALLOCATED**
 // (op 792) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag552: SGPR Value: UInt32 x 1: (unallocated)
 // (op 792) reg expression
 // (op 792) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 792): v11
v_and_b32 v11, -8, v1 // (op 792) 
 // (op 792) BEGIN: Tile[1]
 // (op 792) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 792): v14
v_sub_u32 v14, v1, v11 // (op 792) 
 // (op 792) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 792): v11

 // (op 792) BEGIN: Flatten
 // (op 792) {Flatten: ShiftL({Tile[1]: v14:U32}, 4:U32)U32}
 // (op 792) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 792)         value (Tile[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 792)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 792): v11
v_lshlrev_b32 v11, 4, v14 // (op 792) 
 // (op 792) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 792): v14

 // (op 792) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 792): v14
v_lshrrev_b32 v14, 3, v1 // (op 792) 
 // (op 792) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 792): v15
v_lshl_add_u32 v15, v14, 7, v11 // (op 792) 
 // (op 792) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 792): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 792): v14

// Allocated DataFlowTag552: 1 SGPR (Value: UInt32) (op 792): s0
s_nop 0
v_readfirstlane_b32 s0, v15 // (op 792) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 792): v15

 // (op 446) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(153)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(153)U32, BitwiseAnd(DataFlowTag(153)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(792) END
 // (op 446) Assign VGPR 2048:U32(793) BEGIN
 // (op 793) Assign dim(553) = 2048:U32
 // (op 793) Assign stride expression
 // (op 446) Assign VGPR 2048:U32(793) END
 // (op 446) NOP(788) BEGIN
 // (op 788) generate({})
 // (op 788) end: generate({})
 // (op 446) NOP(788) END
 // (op 446) Assign VGPR 1:U32(797) BEGIN
 // (op 797) Assign dim(555) = 1:U32
 // (op 797) Assign stride expression
 // (op 446) Assign VGPR 1:U32(797) END
 // (op 446) NOP(839) BEGIN
 // (op 839) generate({})
 // (op 839) end: generate({})
 // (op 446) NOP(839) END
 // (op 446) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(844) BEGIN
 // (op 844) Assign dim(568) = {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}
 // tag 568: v**UNALLOCATED**
 // (op 844) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32} into DataFlowTag568: VGPR Value: UInt32 x 1: (unallocated)
 // (op 844) reg expression
 // (op 844) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 844): v11
v_and_b32 v11, -4, v1 // (op 844) 
 // (op 844) BEGIN: Tile[1]Flatten
 // (op 844) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 844): v14
v_sub_u32 v14, v1, v11 // (op 844) 
 // (op 844) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 844): v11

 // (op 844) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 844): v11
v_and_b32 v11, 4294967292, v1 // (op 844) 
 // (op 844) BEGIN: Flatten
 // (op 844) {Flatten: Add(v11:U32, {Tile[1]Flatten: v14:U32})U32}
// Allocated DataFlowTag568: 1 VGPR (Value: UInt32) (op 844): v15
v_add_u32 v15, v11, v14 // (op 844) 
 // (op 844) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 844): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 844): v11

 // (op 446) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(249)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(249)U32, BitwiseAnd(DataFlowTag(249)U32, -4:I)U32)U32})U32}(844) END
 // (op 446) Assign VGPR 4:U32(845) BEGIN
 // (op 845) Assign dim(569) = 4:U32
 // (op 845) Assign stride expression
 // (op 446) Assign VGPR 4:U32(845) END
 // (op 446) NOP(840) BEGIN
 // (op 840) generate({})
 // (op 840) end: generate({})
 // (op 446) NOP(840) END
 // (op 446) Assign VGPR 1:U32(849) BEGIN
 // (op 849) Assign dim(571) = 1:U32
 // (op 849) Assign stride expression
 // (op 446) Assign VGPR 1:U32(849) END
 // (op 446) ForLoopOp KLoopTail: LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL(161) BEGIN
 // (op 161) Initialize For Loop
 // (op 161) generate({162})
 // (op 161) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32(162) BEGIN
 // (op 162) Assign dim(414) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32
 // tag 414: s**UNALLOCATED**
 // (op 162) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag414: SGPR Value: UInt32 x 1: (unallocated)
 // (op 162) reg expression
 // (op 162) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 162): s14, s15
s_ashr_i64 s[14:15], s[28:29], 7 // (op 162) 
 // (op 162) convert
 // (op 162) BitwiseAnd({ convertForwardSubset: s14:U32}, 4294967294:U32)U32
// Allocated DataFlowTag414: 1 SGPR (Value: UInt32) (op 162): s1
s_and_b32 s1, s14, 4294967294 // (op 162) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 162): s14, s15

 // (op 161) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32(162) END
 // (op 161) end: generate({162})
 // (op 161) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 161) 
 // (op 161) Generate LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 161) LessThan({DataFlowTag414: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32)BL
 // (op 161) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 161): s14, s15
s_ashr_i64 s[14:15], s[28:29], 7 // (op 161) 
// Allocated : 1 SGPR (Value: UInt32) (op 161): s18
s_mov_b32 s18, s14 // (op 161) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 161): s14, s15

s_cmp_lt_u32 s1, s18 // (op 161) 
 // Freeing : 1 SGPR (Value: UInt32) (op 161): s18

 // Freeing : 1 VGPR (Value: E8M0) (op 351): v25

 // Freeing : 1 VGPR (Value: E8M0) (op 358): v17

 // Freeing : 3 VGPRs (Value: FP6x16) (op 361): v26, v27, v28

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 161) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 161) 
s_cbranch_scc0 RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 // (op 161) Condition: Top (jump to RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 161) 
RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161:
 // (op 161) 
 // (op 161) generate({163, 206})
 // (op 161) Assign SGPR Add(DataFlowTag(414)U32, 1:U32)U32(163) BEGIN
 // (op 163) Assign dim(414) = Add(DataFlowTag(414)U32, 1:U32)U32
 // (op 163) Generate Add(DataFlowTag(414)U32, 1:U32)U32 into DataFlowTag414: SGPR Value: UInt32 x 1: s1
 // (op 163) reg expression
 // (op 163) Add({DataFlowTag414: s1:U32}, 1:U32)U32
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
s_add_u32 s1, s1, 1 // (op 163) 
 // (op 161) Assign SGPR Add(DataFlowTag(414)U32, 1:U32)U32(163) END
// Allocated Offset454: 1 VGPR (Value: UInt32) (op 166): v11
v_mov_b32 v11, v10 // (op 166) 
 // (op 166) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset454: VGPR Value: UInt32 x 1: v11
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

// Allocated : 3 VGPRs (Value: FP6x16) (op 166): v16, v17, v18
buffer_load_dwordx3 v[16:18], v11, s[40:43], 0 offen // (op 166) Load value
// VMEM: Expected complete at 4326 (current 3846)
 // (op 206) LoadTiled Value: FP6x16(166) END
 // (op 206) Deallocate{}(1108) BEGIN
 // (op 1108) Deallocate 454
 // Freeing Offset454: 1 VGPR (Value: UInt32) (op 166): v11

 // Deleting tag 454
 // (op 1108) Deallocate 456
 // Deleting tag 456
 // (op 206) Deallocate{}(1108) END
 // (op 206) Assign VGPR Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32(441) BEGIN
 // (op 441) Assign dim(450) = Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32
 // (op 441) Generate Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32 into DataFlowTag450: VGPR Value: UInt32 x 1: v10
 // (op 441) reg expression
 // (op 441) Add({DataFlowTag450: v10:U32}, 96:U32)U32
v_add_u32 v10, 96, v10 // (op 441) 
 // (op 206) Assign VGPR Add(DataFlowTag(450)U32, Convert(DataFlowTag(451)U64)U32)U32(441) END
 // (op 206) end: generate({166})
 // (op 161) SetCoordinate(206) END
 // (op 161) Barrier(1032) BEGIN
 // (op 161) Barrier(1045) BEGIN
s_barrier  // (op 1032) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 161) Barrier(1032) END
s_barrier  // (op 1045) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 161) Barrier(1045) END
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
 // 	info.data = VGPR Value: FP6x16 x 1: v[16:18]
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

 // (op 167) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 1: v[16:18]
 // 	info.rowOffsetReg = DataFlowTag536: VGPR Value: UInt32 x 1: v13
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
 // (op 167) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 161) SetCoordinate(214) BEGIN
 // (op 214) generate({})
 // (op 214) end: generate({})
 // (op 214) generate({421})
 // (op 214) LoadTileDirect2LDS Value: FP8x4(421) BEGIN
 // (op 421) GEN: LoadTileDirect2LDS
 // (op 421) GEN: loadMacroTileDirect2LDS OP 421 LDS 24 MacroTile 418
 // (op 421) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 421) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride474: VGPR Value: UInt64 x 1: (unallocated)
 // (op 421) reg expression
 // (op 421) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 421) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 421)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 421)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 421): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 421) 
// Allocated Stride474: 2 VGPRs (Value: UInt64) (op 421): v20, v21
v_mov_b64 v[20:21], s[14:15] // (op 421) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 421): s14, s15

 // (op 421) Generate 1:U64 into nullptr
 // (op 421) Generate 2048:U32 into nullptr
 // (op 421) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride474: VGPR Value: UInt64 x 1: v[20:21]
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
 // 	info.bufDesc = DataFlowTag472: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 473: v**UNALLOCATED**
// Allocated Offset473: 2 VGPRs (Value: UInt64) (op 421): v26, v27
v_mov_b64 v[26:27], v[22:23] // (op 421) 
// Allocated : 2 VGPRs (Value: UInt64) (op 421): v28, v29
v_mov_b64 v[28:29], v[26:27] // (op 421) 
 // (op 421) Generate Add(0:U32, {DataFlowTag552Flatten: s0:U32})U32 into nullptr
 // (op 421)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride474: v[20:21]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 421) Generate {DataFlowTag552Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 421) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 421) Load value direct to lds
// VMEM: Expected complete at 4335 (current 3855)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 421) Generate Add(v[28:29]:U64, v20:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 421) reg expression
 // (op 421) reg expression
 // (op 421) Add(v[28:29]:U64, v20:R)U64
 // (op 421) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 421)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 421)      rhs (VGPR Value: Raw32 x 1: v20)
// Allocated : 2 SGPRs (Value: Bool64) (op 421): s14, s15
v_add_co_u32 v28, s[14:15], v28, v20 // (op 421) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 421) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 421): s14, s15

 // (op 421) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 421) reg expression
 // (op 421) Add(m0:U32, 2048:U32)U32
 // (op 421) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 421) 
 // (op 421) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 421) Load value direct to lds
// VMEM: Expected complete at 4339 (current 3859)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // Freeing Stride474: 2 VGPRs (Value: UInt64) (op 421): v20, v21

 // Freeing : 2 VGPRs (Value: UInt64) (op 421): v28, v29

 // (op 214) LoadTileDirect2LDS Value: FP8x4(421) END
 // (op 214) Deallocate{}(1170) BEGIN
 // (op 1170) Deallocate 475
 // Deleting tag 475
 // (op 1170) Deallocate 554
 // Deleting tag 554
 // (op 1170) Deallocate 473
 // Freeing Offset473: 2 VGPRs (Value: UInt64) (op 421): v26, v27

 // Deleting tag 473
 // (op 1170) Deallocate 418
 // Deleting tag 418
 // (op 214) Deallocate{}(1170) END
 // (op 214) Assign VGPR Add(DataFlowTag(469)U64, DataFlowTag(470)U64)U64(510) BEGIN
 // (op 510) Assign dim(469) = Add(DataFlowTag(469)U64, DataFlowTag(470)U64)U64
 // (op 510) Generate Add(DataFlowTag(469)U64, DataFlowTag(470)U64)U64 into DataFlowTag469: VGPR Value: UInt64 x 1: v[22:23]
 // (op 510) reg expression
 // (op 510) Add({DataFlowTag469: v[22:23]:U64}, 128:U64)U64
 // (op 510) Add: dest (VGPR Value: UInt64 x 1: v[22:23]) = 
 // (op 510)      lhs (DataFlowTag469: VGPR Value: UInt64 x 1: v[22:23]) 
 // (op 510)      rhs (Literal Value: UInt64 x 0: 128)
v_add_co_u32 v22, vcc, 128, v22 // (op 510) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 510) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 214) Assign VGPR Add(DataFlowTag(469)U64, DataFlowTag(470)U64)U64(510) END
 // (op 214) end: generate({421})
 // (op 161) SetCoordinate(214) END
s_waitcnt vmcnt(2)
ds_write_b96 v13, v[16:18] offset:8320 // (op 167) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 3953 (current 3858)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
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
 // (op 161) Deallocate{}(1110) BEGIN
 // (op 1110) Deallocate 416
 // Freeing : 3 VGPRs (Value: FP6x16) (op 166): v16, v17, v18

 // Deleting tag 416
 // (op 1110) Deallocate 538
 // Deleting tag 538
 // (op 161) Deallocate{}(1110) END
// Allocated Offset511: 1 VGPR (Value: UInt32) (op 180): v11
v_mov_b32 v11, v24 // (op 180) 
 // (op 180) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset511: VGPR Value: UInt32 x 1: v11
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

 // (op 161) NOP(423) BEGIN
 // (op 423) generate({})
 // (op 423) end: generate({})
 // (op 161) NOP(423) END
// Allocated : 1 VGPR (Value: E8M0) (op 180): v14
buffer_load_ubyte v14, v11, s[52:55], 0 offen offset: 0 // (op 180) Load value
// VMEM: Expected complete at 4810 (current 4330)
 // (op 219) LoadTiled Value: E8M0(180) END
 // (op 219) Deallocate{}(1118) BEGIN
 // (op 1118) Deallocate 511
 // Freeing Offset511: 1 VGPR (Value: UInt32) (op 180): v11

 // Deleting tag 511
 // (op 1118) Deallocate 513
 // Deleting tag 513
 // (op 219) Deallocate{}(1118) END
 // (op 219) Assign VGPR Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32(638) BEGIN
 // (op 638) Assign dim(507) = Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32
 // (op 638) Generate Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32 into DataFlowTag507: VGPR Value: UInt32 x 1: v24
 // (op 638) reg expression
 // (op 638) Add({DataFlowTag507: v24:U32}, 4:U32)U32
v_add_u32 v24, 4, v24 // (op 638) 
 // (op 219) Assign VGPR Add(DataFlowTag(507)U32, Convert(DataFlowTag(508)U64)U32)U32(638) END
 // (op 219) end: generate({180})
 // (op 161) SetCoordinate(219) END
 // (op 161) Barrier(1035) BEGIN
 // (op 161) Barrier(1058) BEGIN
s_barrier  // (op 1058) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 161) Barrier(1058) END
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1035) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 161) Barrier(1035) END
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
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

 // (op 178) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = DataFlowTag568: VGPR Value: UInt32 x 1: v15
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
 // (op 178) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
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
 // (op 169) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 169): v16, v17, v18, v19, v20, v21
ds_read_b128 v[16:19], v3 offset:8320 // (op 169) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 4051 (current 3956)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
ds_read_b64 v[20:21], v3 offset:8336 // (op 169) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 4052 (current 3957)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 168) LoadLDSTile{Value: FP6x16}(169) END
 // (op 168) end: generate({169})
 // (op 161) SetCoordinate(168) END
s_waitcnt vmcnt(0)
ds_write_b8 v15, v14 offset:8192 // (op 178) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 4054 (current 3959)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 161) StoreLDSTile Value: E8M0(178) END
 // (op 161) Barrier(1048) BEGIN
 // (op 161) Deallocate{}(1116) BEGIN
 // (op 1116) Deallocate 570
 // Deleting tag 570
 // (op 1116) Deallocate 420
 // Freeing : 1 VGPR (Value: E8M0) (op 180): v14

 // Deleting tag 420
 // (op 161) Deallocate{}(1116) END
s_barrier  // (op 1048) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 161) Barrier(1048) END
 // (op 161) SetCoordinate(170) BEGIN
 // (op 170) generate({})
 // (op 170) end: generate({})
 // (op 170) generate({171})
 // (op 170) LoadLDSTile{Value: FP8x4}(171) BEGIN
 // (op 171) GEN: loadMacroTileWAVELDS OP 171 LDS 24 WaveTile 165
 // (op 171) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 171) Generate 1:U32 into nullptr
 // tag 424: v**UNALLOCATED**
 // (op 171) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v4
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 64

// Allocated : 8 VGPRs (Value: FP8x4) (op 171): v26, v27, v28, v29, v30, v31, v32, v33
ds_read_b128 v[26:29], v4 // (op 171) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected stall of 91. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4147 (current 4052)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
ds_read_b128 v[30:33], v4 offset:64 // (op 171) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected complete at 4148 (current 4053)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 170) LoadLDSTile{Value: FP8x4}(171) END
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
// Allocated Offset492: 1 VGPR (Value: UInt32) (op 173): v11
v_mov_b32 v11, v12 // (op 173) 
 // (op 173) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset492: VGPR Value: UInt32 x 1: v11
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

// Allocated : 1 VGPR (Value: E8M0) (op 173): v14
buffer_load_ubyte v14, v11, s[48:51], 0 offen offset: 0 // (op 173) Load value
// VMEM: Expected complete at 5388 (current 4908)
 // (op 211) LoadTiled Value: E8M0(173) END
 // (op 211) Deallocate{}(1112) BEGIN
 // (op 1112) Deallocate 492
 // Freeing Offset492: 1 VGPR (Value: UInt32) (op 173): v11

 // Deleting tag 492
 // (op 1112) Deallocate 494
 // Deleting tag 494
 // (op 211) Deallocate{}(1112) END
 // (op 211) Assign VGPR Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32(574) BEGIN
 // (op 574) Assign dim(488) = Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32
 // (op 574) Generate Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32 into DataFlowTag488: VGPR Value: UInt32 x 1: v12
 // (op 574) reg expression
 // (op 574) Add({DataFlowTag488: v12:U32}, 4:U32)U32
v_add_u32 v12, 4, v12 // (op 574) 
 // (op 211) Assign VGPR Add(DataFlowTag(488)U32, Convert(DataFlowTag(489)U64)U32)U32(574) END
 // (op 211) end: generate({173})
 // (op 172) SetCoordinate(211) END
 // (op 172) end: generate({211})
 // (op 161) SetCoordinate(172) END
 // (op 161) Barrier(1062) BEGIN
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1062) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 161) Barrier(1062) END
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
 // (op 175) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 175): v11
ds_read_u8 v11, v0 offset:8192 // (op 175) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 4154 (current 4059)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 174) LoadLDSTile{Value: E8M0}(175) END
 // (op 174) end: generate({175})
 // (op 161) SetCoordinate(174) END
 // (op 161) Multiply(176) BEGIN
 // (op 176) Generate ScaledMatrixMultiply(WaveTile16xFP6, WaveTile4xFP8, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[26:33], a[0:3], v14, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 176) 
 // (op 161) Multiply(176) END
 // (op 161) Deallocate{}(1114) BEGIN
 // (op 1114) Deallocate 422
 // Freeing : 6 VGPRs (Value: FP6x16) (op 169): v16, v17, v18, v19, v20, v21

 // Deleting tag 422
 // (op 1114) Deallocate 424
 // Freeing : 8 VGPRs (Value: FP8x4) (op 171): v26, v27, v28, v29, v30, v31, v32, v33

 // Deleting tag 424
 // (op 1114) Deallocate 426
 // Freeing : 1 VGPR (Value: E8M0) (op 173): v14

 // Deleting tag 426
 // (op 1114) Deallocate 428
 // Freeing : 1 VGPR (Value: E8M0) (op 175): v11

 // Deleting tag 428
 // (op 161) Deallocate{}(1114) END
 // (op 161) end: generate({163, 206})
 // (op 161) For Loop Increment
 // (op 161) generate({})
 // (op 161) end: generate({})
 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 if true)
 // (op 161) Generate LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 161) LessThan({DataFlowTag414: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32)BL
 // (op 161) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 161): s14, s15
s_ashr_i64 s[14:15], s[28:29], 7 // (op 161) 
// Allocated : 1 SGPR (Value: UInt32) (op 161): s18
s_mov_b32 s18, s14 // (op 161) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 161): s14, s15

s_cmp_lt_u32 s1, s18 // (op 161) 
 // Freeing : 1 SGPR (Value: UInt32) (op 161): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 161) 
s_cbranch_scc1 RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 // (op 161) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_161 if true)
RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_161:
 // (op 161) 
 // (op 161) Unlock For Loop
 // Unlock None
 // (op 446) ForLoopOp KLoopTail: LessThan(DataFlowTag(414)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL(161) END
 // (op 446) Deallocate{}(1106) BEGIN
 // (op 1106) Deallocate 536
 // Freeing DataFlowTag536: 1 VGPR (Value: UInt32) (op 739): v13

 // Deleting tag 536
 // (op 1106) Deallocate 450
 // Freeing DataFlowTag450: 1 VGPR (Value: UInt32) (op 451): v10

 // Deleting tag 450
 // (op 1106) Deallocate 451
 // Deleting tag 451
 // (op 1106) Deallocate 455
 // Deleting tag 455
 // (op 1106) Deallocate 469
 // Freeing DataFlowTag469: 2 VGPRs (Value: UInt64) (op 518): v22, v23

 // Deleting tag 469
 // (op 1106) Deallocate 507
 // Freeing DataFlowTag507: 1 VGPR (Value: UInt32) (op 646): v24

 // Deleting tag 507
 // (op 1106) Deallocate 553
 // Deleting tag 553
 // (op 1106) Deallocate 457
 // Deleting tag 457
 // (op 1106) Deallocate 568
 // Freeing DataFlowTag568: 1 VGPR (Value: UInt32) (op 844): v15

 // Deleting tag 568
 // (op 1106) Deallocate 470
 // Deleting tag 470
 // (op 1106) Deallocate 569
 // Deleting tag 569
 // (op 1106) Deallocate 495
 // Deleting tag 495
 // (op 1106) Deallocate 414
 // Freeing DataFlowTag414: 1 SGPR (Value: UInt32) (op 162): s1

 // Deleting tag 414
 // (op 1106) Deallocate 474
 // Deleting tag 474
 // (op 1106) Deallocate 476
 // Deleting tag 476
 // (op 1106) Deallocate 488
 // Freeing DataFlowTag488: 1 VGPR (Value: UInt32) (op 582): v12

 // Deleting tag 488
 // (op 1106) Deallocate 489
 // Deleting tag 489
 // (op 1106) Deallocate 508
 // Deleting tag 508
 // (op 1106) Deallocate 514
 // Deleting tag 514
 // (op 1106) Deallocate 537
 // Deleting tag 537
 // (op 1106) Deallocate 539
 // Deleting tag 539
 // (op 1106) Deallocate 493
 // Deleting tag 493
 // (op 1106) Deallocate 552
 // Freeing DataFlowTag552: 1 SGPR (Value: UInt32) (op 792): s0

 // Deleting tag 552
 // (op 1106) Deallocate 555
 // Deleting tag 555
 // (op 1106) Deallocate 512
 // Deleting tag 512
 // (op 1106) Deallocate 571
 // Deleting tag 571
 // (op 446) Deallocate{}(1106) END
 // (op 446) end: generate({440})
 // (op 446) Unlock Scope 446
 // Unlock None
 // (op 202) Scope(446) END
 // (op 202) end: generate({446})
 // (op 201) SetCoordinate(202) END
 // (op 201) end: generate({202})
 // (op 310) SetCoordinate(201) END
 // (op 310) Deallocate{}(1120) BEGIN
 // (op 1120) Deallocate 72
 // Deleting tag 72
 // (op 1120) Deallocate 138
 // Deleting tag 138
 // (op 1120) Deallocate 395
 // Deleting tag 395
 // (op 1120) Deallocate 472
 // Freeing DataFlowTag472: 4 SGPRs (Buffer: None) (op 543): s44, s45, s46, s47

 // Deleting tag 472
 // (op 1120) Deallocate 16
 // Deleting tag 16
 // (op 1120) Deallocate 397
 // Deleting tag 397
 // (op 1120) Deallocate 562
 // Deleting tag 562
 // (op 1120) Deallocate 578
 // Deleting tag 578
 // (op 1120) Deallocate 453
 // Freeing DataFlowTag453: 4 SGPRs (Buffer: None) (op 479): s40, s41, s42, s43

 // Deleting tag 453
 // (op 1120) Deallocate 200
 // Deleting tag 200
 // (op 1120) Deallocate 234
 // Deleting tag 234
 // (op 1120) Deallocate 546
 // Deleting tag 546
 // (op 1120) Deallocate 32
 // Deleting tag 32
 // (op 1120) Deallocate 87
 // Deleting tag 87
 // (op 1120) Deallocate 249
 // Deleting tag 249
 // (op 1120) Deallocate 1
 // Deleting tag 1
 // (op 1120) Deallocate 4
 // Deleting tag 4
 // (op 1120) Deallocate 24
 // Deleting tag 24
 // (op 1120) Deallocate 153
 // Deleting tag 153
 // (op 1120) Deallocate 3
 // Deleting tag 3
 // (op 1120) Deallocate 510
 // Freeing DataFlowTag510: 4 SGPRs (Buffer: None) (op 671): s52, s53, s54, s55

 // Deleting tag 510
 // (op 1120) Deallocate 399
 // Deleting tag 399
 // (op 1120) Deallocate 393
 // Deleting tag 393
 // (op 1120) Deallocate 491
 // Freeing DataFlowTag491: 4 SGPRs (Buffer: None) (op 607): s48, s49, s50, s51

 // Deleting tag 491
 // (op 310) Deallocate{}(1120) END
 // (op 310) Deallocate{Tensor_2_stride_1_3, Tensor_4_pointer, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_7_pointer, Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer}(1234) BEGIN
 // (op 1234) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 1234) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 1234) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1234) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1234) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1234) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1234) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1234) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1234) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1234) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 310) Deallocate{Tensor_2_stride_1_3, Tensor_4_pointer, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_7_pointer, Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer}(1234) END
 // (op 310) Barrier(1122) BEGIN
s_barrier  // (op 1122) 
 // (op 310) Barrier(1122) END
 // (op 310) end: generate({315})
 // (op 310) Unlock Scope 310
 // Unlock None
 // (op 1) Scope(310) END
 // (op 1) Deallocate{}(1152) BEGIN
 // (op 1152) Deallocate 561
 // Deleting tag 561
 // (op 1152) Deallocate 595
 // Deleting tag 595
 // (op 1152) Deallocate 609
 // Deleting tag 609
 // (op 1152) Deallocate 614
 // Freeing DataFlowTag614: 1 VGPR (Value: UInt32) (op 993): v5

 // Deleting tag 614
 // (op 1152) Deallocate 545
 // Deleting tag 545
 // (op 1152) Deallocate 563
 // Deleting tag 563
 // (op 1152) Deallocate 577
 // Deleting tag 577
 // (op 1152) Deallocate 617
 // Deleting tag 617
 // (op 1152) Deallocate 597
 // Deleting tag 597
 // (op 1152) Deallocate 607
 // Deleting tag 607
 // (op 1152) Deallocate 544
 // Freeing DataFlowTag544: 1 VGPR (Value: UInt32) (op 767): v3

 // Deleting tag 544
 // (op 1152) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 819): v4

 // Deleting tag 560
 // (op 1152) Deallocate 579
 // Deleting tag 579
 // (op 1152) Deallocate 547
 // Deleting tag 547
 // (op 1152) Deallocate 606
 // Freeing DataFlowTag606: 1 VGPR (Value: UInt32) (op 969): v7

 // Deleting tag 606
 // (op 1152) Deallocate 576
 // Freeing DataFlowTag576: 1 VGPR (Value: UInt32) (op 872): v0

 // Deleting tag 576
 // (op 1152) Deallocate 594
 // Freeing DataFlowTag594: 1 VGPR (Value: UInt32) (op 930): v6

 // Deleting tag 594
 // (op 1152) Deallocate 615
 // Deleting tag 615
 // (op 1) Deallocate{}(1152) END
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
s_cbranch_vccz RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123
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
 // (op 318) NOP(882) BEGIN
 // (op 882) generate({})
 // (op 882) end: generate({})
 // (op 318) NOP(882) END
 // (op 318) NOP(879) BEGIN
 // (op 879) generate({})
 // (op 879) end: generate({})
 // (op 318) NOP(879) END
 // (op 318) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(886) BEGIN
 // (op 886) Assign dim(580) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 580: v**UNALLOCATED**
 // (op 886) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag580: VGPR Value: UInt32 x 1: (unallocated)
 // (op 886) reg expression
 // (op 886) reg expression
 // (op 886) reg expression
 // (op 886) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 886) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 886): v0
v_lshrrev_b32 v0, 7, v1 // (op 886) 
 // (op 886) convert
 // (op 886) BEGIN: Tile[1]
 // (op 886) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 886): v3
v_and_b32 v3, 63, v1 // (op 886) 
 // (op 886) END: Tile[1]
 // (op 886) LogicalShiftR({Tile[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 886): v4
v_lshrrev_b32 v4, 4, v3 // (op 886) 
 // (op 886) BEGIN: Flatten
 // (op 886) {Flatten: ShiftL(v4:U32, 2:U32)U32}
 // (op 886) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 886)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 886)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 886): v5
v_lshlrev_b32 v5, 2, v4 // (op 886) 
 // (op 886) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 886): v4

 // (op 886) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 886): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 886) 
 // (op 886) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 886): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 886): v0

 // (op 886) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 886): v0
v_lshl_add_u32 v0, s16, 4, v4 // (op 886) 
 // (op 886) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 886): v4

 // (op 886) LogicalShiftR({Tile[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 886): v4
v_lshrrev_b32 v4, 2, v3 // (op 886) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 886): v3

 // (op 886) BEGIN: Tile[1]
 // (op 886) {Tile[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 886): v3
v_and_b32 v3, 3, v4 // (op 886) 
 // (op 886) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 886): v4

 // (op 886) BEGIN: Tile[1]
 // (op 886) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 886): v4
v_and_b32 v4, 3, v1 // (op 886) 
 // (op 886) END: Tile[1]
 // (op 886) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 886): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 886) 
 // (op 886) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 886): v4

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 886): v3

 // (op 886) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 886): v3
v_lshrrev_b32 v3, 6, v1 // (op 886) 
 // (op 886) BEGIN: Tile[1]
 // (op 886) {Tile[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 886): v4
v_and_b32 v4, 1, v3 // (op 886) 
 // (op 886) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 886): v3

 // (op 886) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 886): v3
v_lshl_add_u32 v3, v4, 4, v5 // (op 886) 
 // (op 886) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 886): v5

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 886): v4

 // (op 886) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 886): v4
v_lshl_add_u32 v4, s17, 5, v3 // (op 886) 
 // (op 886) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 886): v3

 // (op 886) Multiply({Flatten: v4:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 886): v3
v_mul_lo_u32 v3, v4, s30 // (op 886) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 886): v4

 // (op 886) BEGIN: Join
// Allocated DataFlowTag580: 1 VGPR (Value: UInt32) (op 886): v4
v_add_lshl_u32 v4, v0, v3, 2 // (op 886) 
 // (op 886) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 886): v3

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 886): v0

 // (op 318) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(886) END
 // (op 318) Assign VGPR 64:U64(887) BEGIN
 // (op 887) Assign dim(581) = 64:U64
 // (op 887) Assign stride expression
 // (op 318) Assign VGPR 64:U64(887) END
 // (op 318) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(888) BEGIN
 // (op 888) Assign dim(583) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 583: s**UNALLOCATED**
 // (op 888) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag583: SGPR Buffer: None x 1: (unallocated)
 // (op 888) reg expression
 // (op 888) reg expression
 // (op 888) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 888) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 888)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 888)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 888): s14, s15
s_mul_i32 s15, s30, s13 // (op 888) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 888): s0
s_mul_i32 s0, s31, s12 // (op 888) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 888) least significant
// Allocated : 1 SGPR (Value: Int32) (op 888): s1
s_mul_hi_u32 s1, s30, s12 // (op 888) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 888) most significant: sum
s_add_u32 s15, s15, s1 // (op 888) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 888): s1

 // Freeing : 1 SGPR (Value: Int32) (op 888): s0

 // (op 888) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 888) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 888)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 888)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 888): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 888) 
 // Freeing : 2 SGPRs (Value: Int64) (op 888): s14, s15

 // (op 888) reg expression
 // (op 888) reg expression
// Allocated DataFlowTag583: 4 SGPRs (Buffer: None) (op 888): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 888) reg expression
s_mov_b32 s9, s33 // (op 888) reg expression
s_mov_b32 s10, s0 // (op 888) reg expression
s_mov_b32 s11, 131072 // (op 888) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 888): s0, s1

 // (op 318) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(888) END
 // (op 318) NOP(880) BEGIN
 // (op 880) generate({})
 // (op 880) end: generate({})
 // (op 318) NOP(880) END
 // (op 318) Assign VGPR 4:U64(893) BEGIN
 // (op 893) Assign dim(585) = 4:U64
 // (op 893) Assign stride expression
 // (op 318) Assign VGPR 4:U64(893) END
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
// VMEM: Expected complete at 5917 (current 5437)
 // (op 318) StoreTiled Value: Float(52) END
 // (op 318) Deallocate{}(1088) BEGIN
 // (op 1088) Deallocate 580
 // Freeing DataFlowTag580: 1 VGPR (Value: UInt32) (op 886): v4

 // Deleting tag 580
 // (op 1088) Deallocate 584
 // Deleting tag 584
 // (op 1088) Deallocate 581
 // Deleting tag 581
 // (op 1088) Deallocate 585
 // Deleting tag 585
 // (op 1088) Deallocate 55
 // Freeing DataFlowTag55: 4 VGPRs (Value: Float) (op 46): v10, v11, v12, v13

 // Deleting tag 55
 // (op 318) Deallocate{}(1088) END
 // (op 318) end: generate({321})
 // (op 318) Unlock Scope 318
 // Unlock None
 // (op 123) Scope(318) END
 // (op 123) end: generate({318})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 123) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 123) 
s_branch RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_123:
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
 // (op 324) Scope(704) BEGIN
 // (op 704) Lock Scope 704
 // Lock Branch
 // (op 704) generate({701})
 // (op 704) NOP(701) BEGIN
 // (op 701) generate({})
 // (op 701) end: generate({})
 // (op 704) NOP(701) END
 // (op 704) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(709) BEGIN
 // (op 709) Assign dim(526) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}
 // tag 526: v**UNALLOCATED**
 // (op 709) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32} into DataFlowTag526: VGPR Value: UInt32 x 1: (unallocated)
 // (op 709) reg expression
 // (op 709) reg expression
 // (op 709) reg expression
 // (op 709) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 709) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 709): v0
v_lshrrev_b32 v0, 7, v1 // (op 709) 
 // (op 709) convert
 // (op 709) BEGIN: Flatten[1]
 // (op 709) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 709): v3
v_and_b32 v3, 63, v1 // (op 709) 
 // (op 709) END: Flatten[1]
 // (op 709) LogicalShiftR({Flatten[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 709): v4
v_lshrrev_b32 v4, 4, v3 // (op 709) 
 // (op 709) BEGIN: Tile
 // (op 709) {Tile: ShiftL(v4:U32, 2:U32)U32}
 // (op 709) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 709)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 709)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 709): v5
v_lshlrev_b32 v5, 2, v4 // (op 709) 
 // (op 709) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 709): v4

 // (op 709) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 709): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 709) 
 // (op 709) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 709): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 709): v0

 // (op 709) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 709): v0
v_lshl_add_u32 v0, s16, 4, v4 // (op 709) 
 // (op 709) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 709): v4

 // (op 709) LogicalShiftR({Flatten[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 709): v4
v_lshrrev_b32 v4, 2, v3 // (op 709) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 709): v3

 // (op 709) BEGIN: Flatten[1]
 // (op 709) {Flatten[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 709): v3
v_and_b32 v3, 3, v4 // (op 709) 
 // (op 709) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 709): v4

 // (op 709) BEGIN: Flatten[1]
 // (op 709) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 709): v4
v_and_b32 v4, 3, v1 // (op 709) 
 // (op 709) END: Flatten[1]
 // (op 709) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 709): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 709) 
 // (op 709) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 709): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 709): v3

 // (op 709) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 709): v3
v_lshrrev_b32 v3, 6, v1 // (op 709) 
 // (op 709) BEGIN: Tile
 // (op 709) {Tile: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 709): v4
v_and_b32 v4, 1, v3 // (op 709) 
 // (op 709) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 709): v3

 // (op 709) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 709): v3
v_lshl_add_u32 v3, v4, 4, v5 // (op 709) 
 // (op 709) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 709): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 709): v4

 // (op 709) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 709): v4
v_lshl_add_u32 v4, s17, 5, v3 // (op 709) 
 // (op 709) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 709): v3

 // (op 709) Multiply({Tile: v4:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 709): v3
v_mul_lo_u32 v3, v4, s34 // (op 709) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 709): v4

 // (op 709) BEGIN: Split
// Allocated DataFlowTag526: 1 VGPR (Value: UInt32) (op 709): v4
v_add_lshl_u32 v4, v0, v3, 2 // (op 709) 
 // (op 709) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 709): v3

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 709): v0

 // (op 704) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(290)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(290)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(290)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(709) END
 // (op 704) Deallocate{}(1178) BEGIN
 // (op 1178) Deallocate 403
 // Deleting tag 403
 // (op 1178) Deallocate 401
 // Deleting tag 401
 // (op 1178) Deallocate 290
 // Deleting tag 290
 // (op 704) Deallocate{}(1178) END
 // (op 704) Assign VGPR 64:U64(710) BEGIN
 // (op 710) Assign dim(527) = 64:U64
 // (op 710) Assign stride expression
 // (op 704) Assign VGPR 64:U64(710) END
 // (op 704) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(711) BEGIN
 // (op 711) Assign dim(529) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 529: s**UNALLOCATED**
 // (op 711) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag529: SGPR Buffer: None x 1: (unallocated)
 // (op 711) reg expression
 // (op 711) reg expression
 // (op 711) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 711) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 711)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 711)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 711): s14, s15
s_mul_i32 s15, s34, s13 // (op 711) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 711): s0
s_mul_i32 s0, s35, s12 // (op 711) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 711) least significant
// Allocated : 1 SGPR (Value: Int32) (op 711): s1
s_mul_hi_u32 s1, s34, s12 // (op 711) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 711) most significant: sum
s_add_u32 s15, s15, s1 // (op 711) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 711): s1

 // Freeing : 1 SGPR (Value: Int32) (op 711): s0

 // (op 711) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 711) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 711)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 711)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 711): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 711) 
 // Freeing : 2 SGPRs (Value: Int64) (op 711): s14, s15

 // (op 711) reg expression
 // (op 711) reg expression
// Allocated DataFlowTag529: 4 SGPRs (Buffer: None) (op 711): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 711) reg expression
s_mov_b32 s5, s37 // (op 711) reg expression
s_mov_b32 s6, s0 // (op 711) reg expression
s_mov_b32 s7, 131072 // (op 711) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 711): s0, s1

 // (op 704) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(711) END
 // (op 704) Deallocate{Tensor_10_pointer}(1238) BEGIN
 // (op 1238) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 704) Deallocate{Tensor_10_pointer}(1238) END
 // (op 704) NOP(702) BEGIN
 // (op 702) generate({})
 // (op 702) end: generate({})
 // (op 704) NOP(702) END
 // (op 704) Assign VGPR 4:U64(716) BEGIN
 // (op 716) Assign dim(531) = 4:U64
 // (op 716) Assign stride expression
 // (op 704) Assign VGPR 4:U64(716) END
 // (op 704) LoadTiled Value: Float(135) BEGIN
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
// VMEM: Expected complete at 6428 (current 5948)
 // (op 704) LoadTiled Value: Float(135) END
 // (op 704) Deallocate{Tensor_10_stride_1_10}(1102) BEGIN
 // (op 1102) Deallocate 5
 // Deleting tag 5
 // (op 1102) Deallocate 530
 // Deleting tag 530
 // (op 1102) Deallocate 529
 // Freeing DataFlowTag529: 4 SGPRs (Buffer: None) (op 711): s4, s5, s6, s7

 // Deleting tag 529
 // (op 1102) Deallocate 531
 // Deleting tag 531
 // (op 1102) Deallocate 526
 // Freeing DataFlowTag526: 1 VGPR (Value: UInt32) (op 709): v4

 // Deleting tag 526
 // (op 1102) Deallocate 527
 // Deleting tag 527
 // (op 1102) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 704) Deallocate{Tensor_10_stride_1_10}(1102) END
 // (op 704) end: generate({701})
 // (op 704) Unlock Scope 704
 // Unlock None
 // (op 324) Scope(704) END
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
 // (op 324) Deallocate{}(1104) BEGIN
 // (op 1104) Deallocate 385
 // Freeing : 4 VGPRs (Value: Float) (op 135): v10, v11, v12, v13

 // Deleting tag 385
 // (op 324) Deallocate{}(1104) END
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
 // (op 324) Deallocate{}(1098) BEGIN
 // (op 1098) Deallocate 389
 // Freeing DataFlowTag389: 4 VGPRs (Value: Float) (op 132): v14, v15, v16, v17

 // Deleting tag 389
 // (op 1098) Deallocate 387
 // Freeing DataFlowTag387: 4 VGPRs (Value: Float) (op 136): v4, v5, v6, v7

 // Deleting tag 387
 // (op 324) Deallocate{}(1098) END
 // (op 324) NOP(899) BEGIN
 // (op 899) generate({})
 // (op 899) end: generate({})
 // (op 324) NOP(899) END
 // (op 324) NOP(896) BEGIN
 // (op 896) generate({})
 // (op 896) end: generate({})
 // (op 324) NOP(896) END
 // (op 324) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(903) BEGIN
 // (op 903) Assign dim(586) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 586: v**UNALLOCATED**
 // (op 903) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag586: VGPR Value: UInt32 x 1: (unallocated)
 // (op 903) reg expression
 // (op 903) reg expression
 // (op 903) reg expression
 // (op 903) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 903) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 903): v0
v_lshrrev_b32 v0, 7, v1 // (op 903) 
 // (op 903) convert
 // (op 903) BEGIN: Tile[1]
 // (op 903) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 903): v3
v_and_b32 v3, 63, v1 // (op 903) 
 // (op 903) END: Tile[1]
 // (op 903) LogicalShiftR({Tile[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 903): v4
v_lshrrev_b32 v4, 4, v3 // (op 903) 
 // (op 903) BEGIN: Flatten
 // (op 903) {Flatten: ShiftL(v4:U32, 2:U32)U32}
 // (op 903) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 903)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 903)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 903): v5
v_lshlrev_b32 v5, 2, v4 // (op 903) 
 // (op 903) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 903): v4

 // (op 903) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 903): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 903) 
 // (op 903) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 903): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 903): v0

 // (op 903) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 903): v0
v_lshl_add_u32 v0, s16, 4, v4 // (op 903) 
 // (op 903) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 903): v4

 // (op 903) LogicalShiftR({Tile[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 903): v4
v_lshrrev_b32 v4, 2, v3 // (op 903) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 903): v3

 // (op 903) BEGIN: Tile[1]
 // (op 903) {Tile[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 903): v3
v_and_b32 v3, 3, v4 // (op 903) 
 // (op 903) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 903): v4

 // (op 903) BEGIN: Tile[1]
 // (op 903) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 903): v4
v_and_b32 v4, 3, v1 // (op 903) 
 // (op 903) END: Tile[1]
 // (op 903) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 903): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 903) 
 // (op 903) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 903): v4

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 903): v3

 // (op 903) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 903): v3
v_lshrrev_b32 v3, 6, v1 // (op 903) 
 // (op 903) BEGIN: Tile[1]
 // (op 903) {Tile[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 903): v4
v_and_b32 v4, 1, v3 // (op 903) 
 // (op 903) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 903): v3

 // (op 903) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 903): v3
v_lshl_add_u32 v3, v4, 4, v5 // (op 903) 
 // (op 903) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 903): v5

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 903): v4

 // (op 903) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 903): v4
v_lshl_add_u32 v4, s17, 5, v3 // (op 903) 
 // (op 903) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 903): v3

 // (op 903) Multiply({Flatten: v4:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 903): v3
v_mul_lo_u32 v3, v4, s30 // (op 903) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 903): v4

 // (op 903) BEGIN: Join
// Allocated DataFlowTag586: 1 VGPR (Value: UInt32) (op 903): v4
v_add_lshl_u32 v4, v0, v3, 2 // (op 903) 
 // (op 903) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 903): v3

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 903): v0

 // (op 324) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(330)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(330)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(330)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(903) END
 // (op 324) Assign VGPR 64:U64(904) BEGIN
 // (op 904) Assign dim(587) = 64:U64
 // (op 904) Assign stride expression
 // (op 324) Assign VGPR 64:U64(904) END
 // (op 324) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(905) BEGIN
 // (op 905) Assign dim(583) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 905) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag583: SGPR Buffer: None x 1: s[8:11]
 // (op 905) reg expression
 // (op 905) reg expression
 // (op 905) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 905) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 905)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 905)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 905): s14, s15
s_mul_i32 s15, s30, s13 // (op 905) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 905): s0
s_mul_i32 s0, s31, s12 // (op 905) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 905) least significant
// Allocated : 1 SGPR (Value: Int32) (op 905): s1
s_mul_hi_u32 s1, s30, s12 // (op 905) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 905) most significant: sum
s_add_u32 s15, s15, s1 // (op 905) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 905): s1

 // Freeing : 1 SGPR (Value: Int32) (op 905): s0

 // (op 905) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 905) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 905)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 905)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 905): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 905) 
 // Freeing : 2 SGPRs (Value: Int64) (op 905): s14, s15

 // (op 905) reg expression
 // (op 905) reg expression
s_mov_b32 s8, s32 // (op 905) reg expression
s_mov_b32 s9, s33 // (op 905) reg expression
s_mov_b32 s10, s0 // (op 905) reg expression
s_mov_b32 s11, 131072 // (op 905) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 905): s0, s1

 // (op 324) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(905) END
 // (op 324) NOP(897) BEGIN
 // (op 897) generate({})
 // (op 897) end: generate({})
 // (op 324) NOP(897) END
 // (op 324) Assign VGPR 4:U64(910) BEGIN
 // (op 910) Assign dim(589) = 4:U64
 // (op 910) Assign stride expression
 // (op 324) Assign VGPR 4:U64(910) END
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
// VMEM: Expected complete at 6945 (current 6465)
 // (op 324) StoreTiled Value: Float(134) END
 // (op 324) Deallocate{}(1100) BEGIN
 // (op 1100) Deallocate 586
 // Freeing DataFlowTag586: 1 VGPR (Value: UInt32) (op 903): v4

 // Deleting tag 586
 // (op 1100) Deallocate 587
 // Deleting tag 587
 // (op 1100) Deallocate 391
 // Freeing DataFlowTag391: 4 VGPRs (Value: Float) (op 133): v10, v11, v12, v13

 // Deleting tag 391
 // (op 1100) Deallocate 588
 // Deleting tag 588
 // (op 1100) Deallocate 589
 // Deleting tag 589
 // (op 324) Deallocate{}(1100) END
 // (op 324) end: generate({327})
 // (op 324) Unlock Scope 324
 // Unlock None
 // (op 123) Scope(324) END
 // (op 123) Deallocate{Tensor_21_stride_1_16, Tensor_21_pointer}(1154) BEGIN
 // (op 1154) Deallocate 405
 // Deleting tag 405
 // (op 1154) Deallocate 407
 // Deleting tag 407
 // (op 1154) Deallocate 330
 // Deleting tag 330
 // (op 1154) Deallocate 583
 // Freeing DataFlowTag583: 4 SGPRs (Buffer: None) (op 888): s8, s9, s10, s11

 // Deleting tag 583
 // (op 1154) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 1154) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 123) Deallocate{Tensor_21_stride_1_16, Tensor_21_pointer}(1154) END
 // (op 123) end: generate({324})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 123) 
RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_123:
 // (op 123) 
 // (op 123) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(35)S)BL(123) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1096) BEGIN
 // (op 1096) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 39): v9

 // Deleting tag 44
 // (op 1096) Deallocate 51
 // Freeing DataFlowTag51: 4 ACCVGPRs (Value: Float) (op 59): a0, a1, a2, a3

 // Deleting tag 51
 // (op 1096) Deallocate 35
 // Freeing : 1 VGPR (Value: Float) (op 2): v8

 // Deleting tag 35
 // (op 1096) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1096) END
 // (op 1) end: generate({763, 815, 868, 926, 965, 989})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 11520 // lds bytes
  .amdhsa_accum_offset 52
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
    DataTypeA: F6
    DataTypeB: F8
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
  - .name: RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TN_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 11520
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 50
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

